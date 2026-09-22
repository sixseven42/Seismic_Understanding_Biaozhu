# -*- coding: utf-8 -*-
"""smoke_web.py — web_app 标注 handler 层的多用户冒烟（不起 HTTP，避免 flaky）。

直接用 web_app 的 handler（claim_next / save_anno）模拟两个用户各自领取/保存，
断言：进度正确、不重复标注、记录带 annotated_by、云上传入队（dry-run）。
运行：python tests/smoke_web.py
"""
import json
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 直接 `python tests/smoke_web.py` 时：先加项目根，再加 tests（segy_factory 所在）
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import jobmanager
# 同 tests/test_jobmanager.py：测试里把增强图渲染切回同步，断言才稳定。
# 必须在 import web_app 之前置好 —— web_app 在导入时就建了自己的 JobManager。
jobmanager.AUG_ASYNC_DEFAULT = False
import web_app
from segy_factory import write_sgy
from jobmanager import JobManager
from cloudsync import Uploader
from users import Accounts
from web_core import fmt_option

CFG = web_app.CFG


class _Req:
    """替身 gr.Request：只提供 username。"""

    def __init__(self, username):
        self.username = username


def full_selection():
    """每个特征选合法项；bbox 特征置「不存在」免画框（与 test_jobmanager 同裁决）。"""
    return {f.name: ("不存在" if f.bbox and f.option_by_label("不存在") else f.options[0].label)
            for f in CFG.features}


def radio_values_for(selection):
    vals = []
    for f in CFG.features:
        opt = f.option_by_label(selection.get(f.name, ""))
        vals.append(fmt_option(opt) if opt else None)
    return vals


class SmokeWebFilter(unittest.TestCase):
    """面波区「⚙ 滤波」单键开关：应用/还原、参数按作业记忆、换道集失效、非法参数拒绝。"""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.sgy = os.path.join(self.root, "demo.sgy")
        write_sgy(self.sgy, n_gathers=3, traces_per=8, ns=64)
        self._orig_jm, self._orig_up = web_app.JM, web_app.UP
        web_app.JM = web_app.make_jm(os.path.join(self.root, "jobs"))
        web_app.UP = Uploader({}, dry_run=True)
        self.job = web_app.JM.create_job(self.sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2, 3],
                                         clip=99, title="demo", created_by="boss")

    def tearDown(self):
        web_app.JM, web_app.UP = self._orig_jm, self._orig_up
        web_app.WORK.pop("ann1", None)

    def _claimed(self):
        ann = _Req("ann1")
        web_app.claim_next(ann, self.job.job_id)
        return ann, web_app.wstate("ann1")["gid"]

    @staticmethod
    def _bytes(path):
        with open(path, "rb") as fh:
            return fh.read()

    def _anno_arity(self):
        """anno_outputs 的应有长度：img+info+sentence + 各 radio + 各 bbox 状态 + 4 参数。"""
        return 4 + len(CFG.features) + len(web_app.bbox_feats()) + 4

    def test_toggle_applies_then_reverts(self):
        ann, gid = self._claimed()
        clean = web_app.render_display(ann)
        on = web_app.toggle_filter(ann, 20, 30, 80, 100)          # 第一次点 → 应用
        self.assertIn("已应用滤波", on[1])
        self.assertNotEqual(on[0], clean)                        # 换成了滤波图
        self.assertNotEqual(self._bytes(on[0]), self._bytes(clean))
        self.assertIn("__flt", os.path.basename(on[0]))

        off = web_app.toggle_filter(ann, 20, 30, 80, 100)         # 再点一次 → 还原
        self.assertIn("已还原", off[1])
        self.assertEqual(off[0], clean)                          # 命中干净图缓存
        self.assertEqual(self._bytes(off[0]), self._bytes(clean))

    def test_toggle_output_arity_matches_anno_outputs(self):
        """滤波按钮与 anno_outputs 同序同长 —— 少一个值就会在点击时崩。"""
        ann, _ = self._claimed()
        self.assertEqual(len(web_app.toggle_filter(ann, 20, 30, 80, 100)),
                         self._anno_arity())
        self.assertEqual(len(web_app.toggle_filter(ann, 20, 30, 80, 100)),
                         self._anno_arity())                     # 还原分支同样长度
        self.assertEqual(len(web_app.toggle_filter(_Req("ann1"), 20, 30, 80, 100)),
                         self._anno_arity())                     # 未领取分支同样长度

    def test_params_remembered_for_job_and_refilled(self):
        """第一次设参数并应用；之后字段被回填成该作业的统一参数，不用重填。"""
        ann, gid = self._claimed()
        self.assertEqual(web_app._filter_fields("ann1"), (gr.skip(),) * 4)  # 还没设过
        web_app.toggle_filter(ann, 20, 30, 80, 100)
        self.assertEqual(web_app._filter_fields("ann1"), (20.0, 30.0, 80.0, 100.0))

    def test_not_applied_on_next_gather(self):
        """换到下一张必须重新点一次：滤波状态不跟着走，参数则记住。"""
        ann, gid = self._claimed()
        web_app.toggle_filter(ann, 20, 30, 80, 100)
        web_app.save_anno(ann, *radio_values_for(full_selection()))
        nxt = web_app.wstate("ann1")["gid"]
        self.assertNotEqual(nxt, gid)
        self.assertIsNone(web_app.active_filter(web_app.wstate("ann1"), nxt))
        self.assertNotIn("__flt", os.path.basename(web_app.render_display(ann)))
        self.assertEqual(web_app._filter_fields("ann1"), (20.0, 30.0, 80.0, 100.0))
        # 再点一次即用这套参数应用
        again = web_app.toggle_filter(ann, 20, 30, 80, 100)
        self.assertIn("已应用滤波", again[1])

    def test_invalid_corners_leave_image_untouched(self):
        ann, gid = self._claimed()
        clean = web_app.render_display(ann)
        out = web_app.toggle_filter(ann, 80, 20, 100, 150)        # f1 > f2
        self.assertIn("须满足", out[1])
        self.assertIsNone(web_app.active_filter(web_app.wstate("ann1"), gid))
        self.assertEqual(web_app.render_display(ann), clean)

    def test_toggle_without_claim_is_guided(self):
        out = web_app.toggle_filter(_Req("ann1"), 20, 30, 80, 100)
        self.assertIn("请先领取道集", out[1])

    def test_unknown_dt_refuses_filter(self):
        ann, gid = self._claimed()
        self.job.dt_ms = 0.0                                     # 模拟缺 dt_ms 的旧作业
        out = web_app.toggle_filter(ann, 20, 30, 80, 100)
        self.assertIn("无法滤波", out[1])
        self.assertIsNone(web_app.active_filter(web_app.wstate("ann1"), gid))

    def test_box_status_says_skipped_when_absent(self):
        """选「不存在」→ 该拉框项状态改说「无需画框（本步自动跳过）」，且不再是 Ctrl 目标。"""
        ann, gid = self._claimed()
        sel = full_selection()
        sel["面波"] = "不存在"
        sel["近炮点强能量噪声"] = "存在"        # 只让面波「不存在」
        web_app.on_radio_change(ann, *radio_values_for(sel))
        statuses = web_app.box_statuses_of(ann)
        self.assertIn("无需画框", statuses[0], statuses)          # 面波那条
        self.assertIn("自动跳过", statuses[0])
        self.assertNotIn("当前 Ctrl+左键目标", statuses[0])
        # 面波不再需要框 → Ctrl 目标顺延到近炮点
        self.assertEqual(web_app._boxes_payload("ann1")["target"], "near_shot_noise")

    def test_absent_feature_cannot_get_a_second_box(self):
        """用户实测回归：面波=存在（已画框）、近炮点=不存在 → 不该还能给近炮点拉框。

        旧规则「都框好了就返回最后一个」把目标落在了「不存在」的近炮点上，于是还能拉出
        第二个（多余的）框。现在目标只能是面波（Ctrl 再点＝重画它）。
        """
        ann, gid = self._claimed()
        sel = full_selection()
        sel["面波"] = "存在"
        sel["近炮点强能量噪声"] = "不存在"
        web_app.on_radio_change(ann, *radio_values_for(sel))
        ok, key = web_app.apply_drag_box("ann1", "surface_wave", 10, 20, 100, 200)
        self.assertTrue(ok, key)
        pay = web_app._boxes_payload("ann1")
        self.assertIn("surface_wave", pay["boxes"])
        self.assertEqual(pay["target"], "surface_wave")           # 不再落到近炮点
        # 两个都「不存在」时更是没有可框目标
        sel["面波"] = "不存在"
        web_app.on_radio_change(ann, *radio_values_for(sel))
        self.assertIsNone(web_app._boxes_payload("ann1")["target"])

    def test_boxes_payload_exposes_target(self):
        """Ctrl 两点框选的目标由服务端算好给前端，前端不猜 DOM。"""
        ann, gid = self._claimed()
        pay = web_app._boxes_payload("ann1")
        self.assertEqual(set(pay), {"boxes", "target", "note"})
        self.assertEqual(pay["target"], "abnormal_amplitude")     # 第一个需包络未完成
        self.assertEqual(pay["boxes"], {})
        self.assertEqual(pay["note"], "")                         # 有目标 → 不需要兜底文案

    def test_export_images_unaffected_by_filter(self):
        """滤波只影响显示：保存导出的增强图仍由原始数据渲染。"""
        ann, gid = self._claimed()
        web_app.toggle_filter(ann, 20, 30, 80, 100)
        ok, rec, augs, err = web_app.JM.save(
            self.job.job_id, "ann1", gid, full_selection(),
            {f.key: {"xyxy": [1, 1, 2, 2], "traces": [0, 2], "samples": [0, 4]}
             for f in CFG.features if f.bbox})
        self.assertTrue(ok, err)
        img_dir = os.path.join(self.job.output_dir, "images")
        for a in augs:                                          # 导出图都不带滤波后缀
            self.assertNotIn("__flt", a["image_path"])
            self.assertTrue(os.path.isfile(os.path.join(img_dir,
                                                        os.path.basename(a["image_path"]))))


class SmokeJobPool(unittest.TestCase):
    """多选作业 → 从作业池随机抽道集；保存/跳过后继续从池里抽（可跨作业）。"""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.sgy_a = os.path.join(self.root, "a.sgy")
        self.sgy_b = os.path.join(self.root, "b.sgy")
        write_sgy(self.sgy_a, n_gathers=2, traces_per=8, ns=64)
        write_sgy(self.sgy_b, n_gathers=2, traces_per=8, ns=64)
        self._orig_jm, self._orig_up = web_app.JM, web_app.UP
        web_app.JM = web_app.make_jm(os.path.join(self.root, "jobs"))
        web_app.UP = Uploader({}, dry_run=True)
        self.job_a = web_app.JM.create_job(self.sgy_a, [(95, 96), (13, 16)], [(95, 96)],
                                           [1, 2], clip=99, title="A", created_by="boss")
        self.job_b = web_app.JM.create_job(self.sgy_b, [(95, 96), (13, 16)], [(95, 96)],
                                           [1, 2], clip=99, title="B", created_by="boss")

    def tearDown(self):
        web_app.JM, web_app.UP = self._orig_jm, self._orig_up
        web_app.WORK.pop("ann1", None)

    def _select(self, *job_ids):
        ann = _Req("ann1")
        out = web_app.select_jobs(ann, list(job_ids))
        return ann, out

    def test_selecting_two_jobs_claims_from_either(self):
        ann, out = self._select(self.job_a.job_id, self.job_b.job_id)
        st = web_app.wstate("ann1")
        self.assertEqual(st["job_ids"], [self.job_a.job_id, self.job_b.job_id])
        self.assertIn(st["job_id"], st["job_ids"])
        self.assertIsNotNone(st["gid"])
        # 顶部进度里应同时有「本作业」与「作业池」汇总
        self.assertIn("本作业", out[1])
        self.assertIn("作业池", out[1])
        self.assertIn("共 2 个作业", out[1])

    def test_single_job_has_no_pool_summary(self):
        ann, out = self._select(self.job_a.job_id)
        self.assertIn("本作业", out[1])
        self.assertNotIn("作业池", out[1])          # 单选时不显示池汇总

    def test_save_draws_next_from_pool_and_can_cross_jobs(self):
        """保存后在池里接着抽：跨作业交替是允许且被点名的行为。"""
        ann, _ = self._select(self.job_a.job_id, self.job_b.job_id)
        seen = set()
        for _ in range(6):
            st = web_app.wstate("ann1")
            if not st.get("gid"):
                break
            seen.add(st["job_id"])
            out = web_app.save_anno(ann, *radio_values_for(full_selection()))
            self.assertIn("已保存", out[1])
        self.assertEqual(len(seen), 2, f"6 次保存应至少跨到两个作业，实际只到过 {seen}")

    def test_skip_draws_from_pool(self):
        ann, _ = self._select(self.job_a.job_id, self.job_b.job_id)
        skipped_gid = web_app.wstate("ann1")["gid"]
        out = web_app.skip_current(ann)
        self.assertIn("已跳过", out[1])
        st = web_app.wstate("ann1")
        self.assertIsNotNone(st["gid"])
        self.assertNotEqual(st["gid"], skipped_gid)   # 换了一张

    def test_reselect_same_pool_keeps_current_gather(self):
        ann, _ = self._select(self.job_a.job_id, self.job_b.job_id)
        gid = web_app.wstate("ann1")["gid"]
        web_app.select_jobs(ann, [self.job_a.job_id, self.job_b.job_id])   # 同样两个
        self.assertEqual(web_app.wstate("ann1")["gid"], gid)

    def test_shrinking_pool_without_current_job_drops_gather(self):
        """把当前道集所属作业取消勾选 → 当前道集作废（不该继续标一个没选的作业）。"""
        ann, _ = self._select(self.job_a.job_id, self.job_b.job_id)
        cur_job = web_app.wstate("ann1")["job_id"]
        other = self.job_b.job_id if cur_job == self.job_a.job_id else self.job_a.job_id
        web_app.wstate("ann1")["job_ids"] = [other]      # 模拟用户取消勾选了一个
        web_app.refresh_page(ann)
        st = web_app.wstate("ann1")
        self.assertEqual(st["job_ids"], [other])
        self.assertIsNone(st.get("gid"), "当前道集所属作业已被取消选择，应清空当前道集")

    def test_results_land_in_each_job_folder(self):
        """多作业一起标，产物仍按**当前道集所属作业**落到各自的 jobs/<job_id>/ 里。

        具体核对每个作业目录：labels.jsonl 只含本作业的道集、images/ 与 npy/ 里的文件名
        都属于本作业 —— 不允许串到别的作业目录。
        """
        ann, _ = self._select(self.job_a.job_id, self.job_b.job_id)
        saved = {}                                   # job_id -> [gid, ...]
        for _ in range(6):
            st = web_app.wstate("ann1")
            if not st.get("gid"):
                break
            jid, gid = st["job_id"], st["gid"]
            out = web_app.save_anno(ann, *radio_values_for(full_selection()))
            self.assertIn("已保存", out[1])
            saved.setdefault(jid, []).append(gid)
        self.assertEqual(len(saved), 2, f"应在两个作业里都存过，实际 {saved}")

        for jid, gids in saved.items():
            job = web_app.JM.get(jid)
            own = set(job.gather_ids())
            # 1) 记录都在本作业的 store 里，且属于本作业
            for gid in gids:
                self.assertIsNotNone(job.store.get(gid), f"{gid} 应记录在 {jid}")
                self.assertIn(gid, own)
            # 2) labels.jsonl 里的记录不越界
            with open(os.path.join(job.output_dir, "labels.jsonl"), encoding="utf-8") as f:
                recs = [json.loads(l) for l in f if l.strip()]
            self.assertTrue(recs)
            for r in recs:
                base = r.get("augmented_from") or r["gather_id"]
                self.assertIn(base, own, f"{jid} 的 labels.jsonl 混入了别的作业的道集")
            # 3) images/ 与 npy/ 的文件名都属于本作业
            for sub, pat in (("images", "images"), ("npy", "npy")):
                d = os.path.join(job.output_dir, sub)
                for name in (os.listdir(d) if os.path.isdir(d) else []):
                    base = name.split("__clip")[0].removesuffix(".npy")
                    self.assertIn(base, own, f"{jid}/{pat} 里混入了别的作业的产物: {name}")

    def test_mine_choices_union_across_jobs(self):
        ann, _ = self._select(self.job_a.job_id, self.job_b.job_id)
        web_app.save_anno(ann, *radio_values_for(full_selection()))
        choices = web_app._mine_choices("ann1", [self.job_a.job_id, self.job_b.job_id])
        self.assertTrue(choices, "「我标注的」应含刚保存的记录")
        # 每个 gid 都能定位回它所属的作业
        for gid in choices:
            self.assertIn(web_app._job_of_gid("ann1", gid,
                                              [self.job_a.job_id, self.job_b.job_id]),
                          (self.job_a.job_id, self.job_b.job_id))


class SmokeNoReassignAndHint(unittest.TestCase):
    """用户实测反馈：标注者看到旧图上浮着「框选已完成」，怀疑已完成的任务被重发。

    实测结论：**不会重发**（已标注道集永远领不到）；那是「池子标完 → 旧图留在屏幕上 +
    提示语误写成『已完成』」造成的错觉。这里把两件事都钉住。
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.sgy = os.path.join(self.root, "demo.sgy")
        write_sgy(self.sgy, n_gathers=2, traces_per=8, ns=64)
        self._orig_jm, self._orig_up = web_app.JM, web_app.UP
        web_app.JM = web_app.make_jm(os.path.join(self.root, "jobs"))
        web_app.UP = Uploader({}, dry_run=True)
        self.job = web_app.JM.create_job(self.sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2],
                                         clip=99, title="d", created_by="boss")

    def tearDown(self):
        web_app.JM, web_app.UP = self._orig_jm, self._orig_up
        web_app.WORK.pop("ann1", None)

    def _label_all(self):
        """把作业全部标完，返回标注过的 gid 集合。"""
        ann = _Req("ann1")
        web_app.select_jobs(ann, [self.job.job_id])
        done = set()
        while True:
            gid = web_app.wstate("ann1").get("gid")
            if not gid:
                break
            out = web_app.save_anno(ann, *radio_values_for(full_selection()))
            self.assertIn("已保存", out[1])
            done.add(gid)
        return ann, done

    def test_labeled_gathers_are_never_handed_out_again(self):
        """核心断言：全部标完后反复领取/归还，绝不会再拿到已标注的道集。"""
        ann, done = self._label_all()
        self.assertEqual(len(done), 2)
        for _ in range(20):
            out = web_app.claim_next(ann, [self.job.job_id])
            st = web_app.wstate("ann1")
            self.assertIsNone(st.get("gid"), "已标完的作业不该再发出道集")
            self.assertIn("全部标注", out[1])

    def test_labeled_gather_not_returned_after_reload(self):
        """重启（load_all）后同样不会重发已标注的。"""
        ann, done = self._label_all()
        web_app.JM = web_app.make_jm(os.path.join(self.root, "jobs"))
        web_app.JM.load_all()
        web_app.WORK.pop("ann1", None)
        ann2 = _Req("ann2")
        web_app.select_jobs(ann2, [self.job.job_id])
        self.assertIsNone(web_app.wstate("ann2").get("gid"))
        for _ in range(10):
            web_app.claim_next(ann2, [self.job.job_id])
            self.assertIsNone(web_app.wstate("ann2").get("gid"))

    def test_idle_state_clears_image_so_no_stale_picture(self):
        """池子标完后图片必须被清掉 —— 否则旧图留在屏幕上会像"又发了一张"。"""
        ann, _done = self._label_all()
        out = web_app.claim_next(ann, [self.job.job_id])
        self.assertNotEqual(out[0], gr.skip(), "空态应清空图片，而不是保留上一张")
        # 空态输出也要与 anno_outputs 同长
        n = 4 + len(CFG.features) + len(web_app.bbox_feats()) + 4
        self.assertEqual(len(out), n)

    def test_payload_note_empty_when_no_gather(self):
        """没有当前道集时 note 为空 → 前端不画任何提示（不会再冒出「已完成」）。"""
        web_app.WORK.pop("ann1", None)
        pay = web_app._boxes_payload("ann1")
        self.assertEqual(set(pay), {"boxes", "target", "note"})
        self.assertIsNone(pay["target"])
        self.assertEqual(pay["note"], "")

    def test_payload_note_explains_all_absent(self):
        """当前这张两项都选「不存在」→ 说明是「无需画框」，而不是「已完成」。"""
        ann = _Req("ann1")
        web_app.select_jobs(ann, [self.job.job_id])
        sel = full_selection()                       # bbox 特征都是「不存在」
        web_app.on_radio_change(ann, *radio_values_for(sel))
        pay = web_app._boxes_payload("ann1")
        self.assertIsNone(pay["target"])
        self.assertIn("无需画框", pay["note"])

    def test_frontend_never_says_completed(self):
        """前端不该再出现「框选已完成」这种会误导的措辞。"""
        js = web_app.ANNO_JS
        self.assertNotIn("框选已完成", js)
        self.assertIn("targetNote", js)


class SmokeUserTotals(unittest.TestCase):
    """本用户标注总量（标注区）+ 各用户标注总量表（管理员区）。"""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.sgy = os.path.join(self.root, "demo.sgy")
        write_sgy(self.sgy, n_gathers=3, traces_per=8, ns=64)
        self._orig_jm, self._orig_up, self._orig_acc = web_app.JM, web_app.UP, web_app.ACC
        web_app.JM = web_app.make_jm(os.path.join(self.root, "jobs"))
        web_app.UP = Uploader({}, dry_run=True)
        # 也换掉账号表：_staff_rows 会列出 users.yaml 里的账号，用真实表会与仓库内容耦合
        upath = os.path.join(self.root, "users.yaml")
        with open(upath, "w", encoding="utf-8") as f:
            f.write("users:\n"
                    "- {username: boss, password: b, role: admin}\n"
                    "- {username: ann1, password: a, role: annotator}\n"
                    "- {username: ann2, password: c, role: annotator}\n")
        web_app.ACC = Accounts(upath)
        self.job = web_app.JM.create_job(self.sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2, 3],
                                         clip=99, title="demo", created_by="boss")

    def tearDown(self):
        web_app.JM, web_app.UP, web_app.ACC = self._orig_jm, self._orig_up, self._orig_acc
        web_app.WORK.pop("ann1", None)

    def test_my_total_counts_and_updates_in_anno_area(self):
        ann = _Req("ann1")
        self.assertIn("0 张", web_app._my_total_text("ann1"))
        web_app.select_jobs(ann, [self.job.job_id])
        out = web_app.save_anno(ann, *radio_values_for(full_selection()))
        self.assertIn("已保存", out[1])
        # anno_outputs 末位就是「我的累计标注」，保存后立刻变成 1
        self.assertIn("我的累计标注：1 张", out[-1])
        self.assertEqual(web_app.JM.user_count("ann1"), 1)

    def test_my_total_keeps_counting(self):
        ann = _Req("ann1")
        web_app.select_jobs(ann, [self.job.job_id])
        for _ in range(2):
            out = web_app.save_anno(ann, *radio_values_for(full_selection()))
            self.assertIn("已保存", out[1])
        self.assertEqual(web_app.JM.user_count("ann1"), 2)
        self.assertIn("我的累计标注：2 张", web_app._my_total_text("ann1"))

    def test_idle_state_leaves_total_untouched(self):
        """空态不重算也不清空（数值没变，gr.skip 保留原值即可）——别把计数洗成 0。"""
        ann = _Req("ann1")
        idle = web_app.refresh_anno(ann)             # 还没领任何道集
        self.assertEqual(idle[-1], gr.skip())

    def test_staff_rows_list_all_users_with_counts(self):
        ann = _Req("ann1")
        web_app.select_jobs(ann, [self.job.job_id])
        web_app.save_anno(ann, *radio_values_for(full_selection()))
        rows = dict((n, c) for n, _r, c in web_app._staff_rows())
        self.assertEqual(set(rows), {"boss", "ann1", "ann2"})   # 全部账号都在
        self.assertEqual(rows["ann1"], 1)
        self.assertEqual(rows["boss"], 0)             # 没标的算 0，不是缺行
        html = web_app.staff_counts_html(web_app._staff_rows())
        self.assertIn("ann1", html)
        self.assertIn(">1<", html)
        self.assertIn("管理员", html)                 # 角色列

    def test_admin_table_is_in_admin_only_section(self):
        """各用户总量表必须挂在管理员可见的 ④ 管理区，不能进全体可见的 ③ 标注量区。"""
        import json as _json
        cfg = _json.dumps(web_app.build_app().get_config_file(), ensure_ascii=False)
        i_admin = cfg.index("④ 管理")
        i_staff = cfg.index("staff-counts") if "staff-counts" in cfg else -1
        # 初始 HTML 在 config 里（服务启动时渲染的那份）应位于 ④ 之后
        self.assertGreater(i_staff, i_admin, "各用户总量表应渲染在 ④ 管理区内")


class SmokeProgressVisibleToAll(unittest.TestCase):
    """标注量/作业进度区：管理员与标注者**都能看到**；建作业/管理操作仍仅管理员。"""

    @classmethod
    def setUpClass(cls):
        cls.cfg = json.dumps(web_app.build_app().get_config_file(), ensure_ascii=False)

    def _visible(self, title_pat):
        m = re.search(r'\{"label": "' + title_pat + r'[^"]*"[^}]*?"visible": (true|false)',
                      self.cfg)
        self.assertIsNotNone(m, f"界面里找不到「{title_pat}」区")
        return m.group(1)

    def test_progress_section_visible_to_everyone(self):
        self.assertEqual(self._visible("③ 标注量"), "true", "标注量区必须对全体可见")

    def test_admin_only_sections_stay_admin_only(self):
        self.assertEqual(self._visible("① 建作业"), "false")
        self.assertEqual(self._visible("④ 管理"), "false")

    def test_init_page_pushes_fresh_progress_table(self):
        """页面加载时按最新数据重渲进度表（组件初值只是"服务启动那一刻"的快照）。"""

        class R:
            username = "ann1"

        out = web_app.init_page(R())
        # 输出个数必须与 demo.load 的输出列表一致（对不上会在页面加载时报错）
        self.assertEqual(len(out), 8, "init_page 输出个数须与 demo.load 的输出列表一致")
        self.assertIn("jobs-progress", out[-2], "init_page 应输出最新的进度表 HTML")
        self.assertIn("我的累计标注", out[-1], "页面加载时应带上本人的累计标注数")


class SmokeAutoClaimOnJobSelect(unittest.TestCase):
    """选中作业即自动领取第一张（不必再点「领取下一张」）。"""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        # 用不同文件名的 sgy → 不同 job_id（job_id 含秒级时间戳，同文件同秒建两个会撞名）
        self.sgy_a = os.path.join(self.root, "a.sgy")
        self.sgy_b = os.path.join(self.root, "b.sgy")
        self.sgy_c = os.path.join(self.root, "c.sgy")
        write_sgy(self.sgy_a, n_gathers=3, traces_per=8, ns=64)
        write_sgy(self.sgy_b, n_gathers=2, traces_per=8, ns=64)
        write_sgy(self.sgy_c, n_gathers=1, traces_per=8, ns=64)
        self._orig_jm, self._orig_up = web_app.JM, web_app.UP
        web_app.JM = web_app.make_jm(os.path.join(self.root, "jobs"))
        web_app.UP = Uploader({}, dry_run=True)
        self.job_a = web_app.JM.create_job(self.sgy_a, [(95, 96), (13, 16)], [(95, 96)],
                                           [1, 2, 3], clip=99, title="A", created_by="boss")
        self.job_b = web_app.JM.create_job(self.sgy_b, [(95, 96), (13, 16)], [(95, 96)],
                                           [1, 2], clip=99, title="B", created_by="boss")
        self.job_c = web_app.JM.create_job(self.sgy_c, [(95, 96), (13, 16)], [(95, 96)],
                                           [1], clip=99, title="C", created_by="boss")

    def tearDown(self):
        web_app.JM, web_app.UP = self._orig_jm, self._orig_up
        web_app.WORK.pop("ann1", None)

    def test_selecting_job_auto_claims_first_gather(self):
        ann = _Req("ann1")
        out = web_app.select_jobs(ann, [self.job_a.job_id])
        st = web_app.wstate("ann1")
        self.assertIsNotNone(st.get("gid"), "选中作业应自动领到一张")
        self.assertIn(st["gid"], out[1])                   # 提示行里报出当前道集
        self.assertNotIn("请选择作业后点", out[1])

    def test_reselect_same_job_keeps_unsaved_work(self):
        """重复选中同一作业：不换张、也不清掉未保存的选择。"""
        ann = _Req("ann1")
        web_app.select_jobs(ann, [self.job_a.job_id])
        gid = web_app.wstate("ann1")["gid"]
        web_app.wstate("ann1")["partial"] = {"集合类型": "炮集"}
        web_app.select_jobs(ann, [self.job_a.job_id])
        self.assertEqual(web_app.wstate("ann1")["gid"], gid)
        self.assertEqual(web_app.wstate("ann1")["partial"], {"集合类型": "炮集"})

    def test_switching_job_claims_there(self):
        ann = _Req("ann1")
        web_app.select_jobs(ann, [self.job_a.job_id])
        web_app.select_jobs(ann, [self.job_b.job_id])
        st = web_app.wstate("ann1")
        self.assertEqual(st["job_id"], self.job_b.job_id)
        self.assertIsNotNone(st.get("gid"))

    def test_completed_job_reports_all_done_without_gid(self):
        ann = _Req("ann1")
        web_app.select_jobs(ann, [self.job_c.job_id])       # 只有一张，先领到
        gid = web_app.wstate("ann1")["gid"]
        ok, rec, augs, err = web_app.JM.save(
            self.job_c.job_id, "ann1", gid, full_selection(),
            {f.key: {"xyxy": [1, 1, 2, 2], "traces": [0, 2], "samples": [0, 4]}
             for f in CFG.features if f.bbox})
        self.assertTrue(ok, err)
        web_app.WORK.pop("ann1", None)                     # 当作重新进页面
        out = web_app.select_jobs(ann, [self.job_c.job_id])
        self.assertIsNone(web_app.wstate("ann1").get("gid"))
        self.assertIn("全部标注", out[1])

    def test_claim_button_still_works_after_release(self):
        """归还后仍可继续（自动领取不是唯一入口）。"""
        ann = _Req("ann1")
        web_app.select_jobs(ann, [self.job_a.job_id])
        web_app.release_current(ann)
        self.assertIsNone(web_app.wstate("ann1").get("gid"))
        out = web_app.claim_next(ann, self.job_a.job_id)
        self.assertIsNotNone(web_app.wstate("ann1").get("gid"))
        self.assertEqual(len(out), 4 + len(CFG.features) + len(web_app.bbox_feats()) + 4)


class SmokeAnnoOutputArity(unittest.TestCase):
    """anno_outputs 契约：每个事件返回的值个数必须与输出列表一致。

    多一个/少一个只会在浏览器点击时才崩，故在此把每个 handler 都过一遍。
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.sgy = os.path.join(self.root, "demo.sgy")
        write_sgy(self.sgy, n_gathers=3, traces_per=8, ns=64)
        self._orig_jm, self._orig_up = web_app.JM, web_app.UP
        web_app.JM = web_app.make_jm(os.path.join(self.root, "jobs"))
        web_app.UP = Uploader({}, dry_run=True)
        self.job = web_app.JM.create_job(self.sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2, 3],
                                         clip=99, title="demo", created_by="boss")

    def tearDown(self):
        web_app.JM, web_app.UP = self._orig_jm, self._orig_up
        web_app.WORK.pop("ann1", None)

    def _n_anno(self):
        # img + info + sentence + 各 radio + 各 bbox 状态 + 4 个滤波参数
        return 4 + len(CFG.features) + len(web_app.bbox_feats()) + 4

    def test_anno_events_return_anno_outputs_length(self):
        ann = _Req("ann1")
        n = self._n_anno()
        self.assertEqual(len(web_app.claim_next(ann, self.job.job_id)), n, "claim_next")
        self.assertEqual(len(web_app.refresh_page(ann)), 2 + n, "refresh_page")
        self.assertEqual(len(web_app.refresh_anno(ann)), n, "refresh_anno")
        self.assertEqual(len(web_app.toggle_filter(ann, 20, 30, 80, 100)), n, "toggle_filter")
        self.assertEqual(len(web_app.toggle_filter(ann, 20, 30, 80, 100)), n, "toggle_filter/还原")
        n_clear = 2 + len(web_app.bbox_feats())        # img + info + 各 bbox 状态
        for f in web_app.bbox_feats():
            self.assertEqual(len(web_app.clear_box(ann, f.key)), n_clear,
                             f"clear_box({f.key})")
        # 保存并自动领取下一张
        self.assertEqual(len(web_app.save_anno(ann, *radio_values_for(full_selection()))), n,
                         "save_anno")
        self.assertEqual(len(web_app.skip_current(ann)), n, "skip_current")
        self.assertEqual(len(web_app.release_current(ann)), n, "release_current")
        # 改选项 → 句子 + 各框状态（Ctrl 目标随选项变）
        self.assertEqual(len(web_app.on_radio_change(ann, *radio_values_for(full_selection()))),
                         1 + len(web_app.bbox_feats()), "on_radio_change")
        # 继承输出句子 + 各 radio + **各框状态**：继承是程序化改选项，不产生 DOM change
        # 事件，框状态标记是前端唯一能拿到的"该回读 Ctrl 目标了"的信号
        self.assertEqual(len(web_app.inherit_previous(ann)),
                         1 + len(CFG.features) + len(web_app.bbox_feats()),
                         "inherit_previous")
        # 返回上一张走 anno_outputs 全套（图/提示/句子/radio/框状态/滤波参数/累计）
        self.assertEqual(len(web_app.back_previous(ann)), n, "back_previous")
        # open_job_for 额外多一个「我标注的」下拉
        self.assertEqual(len(web_app.select_jobs(ann, [self.job.job_id])),
                         n + 1, "open_job_for")
        # 空态/无作业分支同样要对齐
        web_app.WORK.pop("ann1", None)
        self.assertEqual(len(web_app.refresh_anno(ann)), n, "refresh_anno 空态")
        self.assertEqual(len(web_app.toggle_filter(ann, 20, 30, 80, 100)), n,
                         "toggle_filter 空态")


class SmokeBackPrevious(unittest.TestCase):
    """「↩ 返回上一张」：连续往回翻、修正保存覆盖原结果且**不**自动前进。

    历史用浏览栈而不是「我标注的」列表顺序 —— 那张表按首次标注时间排序
    （storage.LabelStore.upsert 对已存在的 gid 不挪位），修正保存不会让记录回到末尾，
    靠它会翻到别处去。
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.sgy = os.path.join(self.root, "demo.sgy")
        write_sgy(self.sgy, n_gathers=5, traces_per=8, ns=64)
        self._orig_jm, self._orig_up = web_app.JM, web_app.UP
        web_app.JM = web_app.make_jm(os.path.join(self.root, "jobs"))
        web_app.UP = Uploader({}, dry_run=True)
        self.job = web_app.JM.create_job(self.sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2, 3],
                                         clip=99, title="demo", created_by="boss")
        self.ann = _Req("ann1")

    def tearDown(self):
        web_app.JM, web_app.UP = self._orig_jm, self._orig_up
        web_app.WORK.pop("ann1", None)

    # ---- 小工具 ----
    def _gid(self):
        return web_app.wstate("ann1").get("gid")

    def _hist(self):
        return list(web_app.wstate("ann1").get("hist") or [])

    def _save(self, sel=None):
        out = web_app.save_anno(self.ann, *radio_values_for(sel or full_selection()))
        self.assertEqual(len(out), 4 + len(CFG.features) + len(web_app.bbox_feats()) + 4,
                         "save_anno 输出长度")
        return self._gid(), out[1]

    def _back(self):
        out = web_app.back_previous(self.ann)
        self.assertEqual(len(out), 4 + len(CFG.features) + len(web_app.bbox_feats()) + 4,
                         "back_previous 输出长度")
        return out[1]

    def _claim(self):
        web_app.claim_next(self.ann, self.job.job_id)
        return self._gid()

    # ---- 主流程 ----
    def test_walks_back_one_step_at_a_time_then_stops(self):
        """标完 G1、G2（每次自动领下一张）后连点「返回上一张」：回到 G2、G1，再点明确提示。

        用户实测回归：保存完自动领到的那张**一个字都没填**，点「返回上一张」必须真的回去
        —— 早先一律拦下问「尚未保存」，按钮在最常用的场景里等于摆设。
        """
        g1 = self._claim()
        g2, _ = self._save()               # 标完 G1 → 自动领 G2；历史 [G1]
        self.assertNotEqual(g1, g2)
        self.assertEqual(self._hist(), [g1])

        g3, _ = self._save()               # 标完 G2 → 自动领 G3；历史 [G1, G2]
        self.assertNotIn(g3, (g1, g2))
        self.assertEqual(self._hist(), [g1, g2])

        # 站在没动过的 G3 上直接点返回：应把 G3 归还回池并回到 G2
        self.assertIsNone(web_app.JM.record(self.job.job_id, g3))
        msg = self._back()
        self.assertIn("已返回", msg)
        self.assertIn(g2, msg)
        self.assertIn("归还", msg)
        self.assertEqual(self._gid(), g2)
        self.assertEqual(self._hist(), [g1], "回退后目标要出栈，否则原地打转")
        # G3 已归还回池（G2 是已标注记录，本就不持租约 —— 与 reopen_mine 一致）
        self.assertNotEqual(web_app.JM.current(self.job.job_id, "ann1"), g3,
                            "G3 应已归还，不该还被自己占着")

        self.assertIn(g1, self._back())
        self.assertEqual(self._gid(), g1)
        self.assertEqual(self._hist(), [])

        msg = self._back()
        self.assertIn("已是最早一张", msg)
        self.assertEqual(self._gid(), g1, "到最早一张应原地不动")

    def test_back_refuses_when_current_gather_already_answered(self):
        """手上那张**已经作答**（改了选项）→ 拦下来问一句，别把真实工作静默丢掉。"""
        self._claim()
        self._save()                       # 自动领到一张
        unsaved = self._gid()
        self.assertIsNone(web_app.JM.record(self.job.job_id, unsaved))
        web_app.on_radio_change(self.ann, *radio_values_for(full_selection()))   # 作答
        self.assertTrue(web_app._has_unsaved_input(web_app.wstate("ann1")))
        self.assertIn("已作答", self._back())
        self.assertEqual(self._gid(), unsaved, "不该换张")
        self.assertTrue(self._hist(), "历史不该被清")

    def test_back_also_refuses_when_only_a_box_was_drawn(self):
        """画了框也算已作答（不能只看选项）。"""
        self._claim()
        self._save()
        unsaved = self._gid()
        ok, _ = web_app.apply_drag_box("ann1", "surface_wave", 10, 20, 100, 200)
        self.assertTrue(ok)
        self.assertIn("已作答", self._back())
        self.assertEqual(self._gid(), unsaved)

    def test_correction_save_overwrites_original_and_still_advances(self):
        """返回上一张 → 改 → 保存：覆盖原记录，且**照常自动领下一张**。

        用户实测回归：v3.13 让修正保存「原地停住」，但按钮就叫「保存并释放」——
        只覆盖不前进会被当成卡住了。「覆盖」由提示语讲清楚，不靠"停住"表达。
        """
        g1 = self._claim()
        self._save()                       # 标完 G1 → 自动领 G2（未作答）
        self.assertIn("已返回", self._back())      # G2 一个字没填 → 直接归还并回到 G1
        self.assertEqual(self._gid(), g1, "先回到 G1")

        before = web_app.JM.record(self.job.job_id, g1)
        self.assertEqual(before["labels"]["gather_type"], full_selection()["集合类型"])

        sel = full_selection()
        sel["集合类型"] = "残差"            # 只改一项
        after_gid, msg = self._save(sel)
        self.assertIn("已覆盖", msg)                 # 提示语点明是覆盖而非新增
        self.assertIn("已自动领取下一张", msg)        # …同时照常前进
        self.assertNotEqual(after_gid, g1, "修正保存后应自动领下一张")
        self.assertEqual(self._gid(), after_gid)
        after = web_app.JM.record(self.job.job_id, g1)
        self.assertEqual(after["labels"]["gather_type"], "残差", "原记录应被覆盖")
        self.assertEqual(after["annotated_by"], "ann1", "覆盖后仍记原作者")
        self.assertEqual(web_app.JM.user_count("ann1"), 1, "修正不该多算一张")
        # 刚改完那张进了历史，往回点还能回到它
        self.assertIn(g1, self._back())

    def test_back_without_job_is_guided(self):
        self.assertIn("请先选择作业", self._back())

    def test_clear_drag_box_clears_a_committed_box(self):
        """右键清框走的服务端口子：把已落定的框清掉，且 Ctrl 目标随之顺延回来。

        用户实测回归：右键原先只清得掉「还没闭合的半成品」，已落定的框清不掉 ——
        因为它去程序化点 Gradio 的「✕ 清除此框」按钮，而那个按钮在 DOM 里长什么样
        各版本不一致。现在直连这个接口，不再依赖 DOM。
        """
        self._claim()
        ok, msg = web_app.apply_drag_box("ann1", "surface_wave", 10, 20, 100, 200)
        self.assertTrue(ok, msg)
        pay = web_app._boxes_payload("ann1")
        self.assertIn("surface_wave", pay["boxes"])
        self.assertEqual(pay["target"], "abnormal_amplitude", "异常振幅仍是首个未完成包络")

        ok, msg = web_app.clear_drag_box("ann1", "surface_wave")
        self.assertTrue(ok, msg)
        pay = web_app._boxes_payload("ann1")
        self.assertNotIn("surface_wave", pay["boxes"], "已落定的框应被清掉")
        self.assertEqual(pay["target"], "abnormal_amplitude", "清掉后仍应先完成异常振幅")

    def test_clear_drag_box_rejects_unknown_key(self):
        self._claim()
        ok, msg = web_app.clear_drag_box("ann1", "not_a_feature")
        self.assertFalse(ok)
        self.assertIn("非框选特征", msg)

    def test_clear_drag_box_without_claim_is_refused(self):
        ok, msg = web_app.clear_drag_box("ann1", "surface_wave")
        self.assertFalse(ok)
        self.assertIn("没有正在标注的道集", msg)

    def test_switching_job_selection_resets_history(self):
        """换作业选择是显式换了上下文 → 历史清空，不会退回上一套选择的道集。"""
        self._claim()
        self._save()
        self.assertTrue(self._hist())
        # 作业 id 取自 sgy 文件名 + 时间戳 → 同文件建第二个作业会撞 id，换一个文件
        sgy2 = os.path.join(self.root, "demo2.sgy")
        write_sgy(sgy2, n_gathers=3, traces_per=8, ns=64)
        other = web_app.JM.create_job(sgy2, [(95, 96), (13, 16)], [(95, 96)], [1, 2, 3],
                                      clip=99, title="demo2", created_by="boss")
        self.assertNotEqual(other.job_id, self.job.job_id, "两个作业应有不同 id")
        web_app.select_jobs(self.ann, [other.job_id])
        self.assertEqual(self._hist(), [])


class SmokeInheritRefreshesBoxes(unittest.TestCase):
    """继承最近已标注：程序化改选项后必须把框状态一起回吐（前端据此回读 Ctrl 目标）。"""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.sgy = os.path.join(self.root, "demo.sgy")
        write_sgy(self.sgy, n_gathers=3, traces_per=8, ns=64)
        self._orig_jm, self._orig_up = web_app.JM, web_app.UP
        web_app.JM = web_app.make_jm(os.path.join(self.root, "jobs"))
        web_app.UP = Uploader({}, dry_run=True)
        self.job = web_app.JM.create_job(self.sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2, 3],
                                         clip=99, title="demo", created_by="boss")
        self.ann = _Req("ann1")

    def tearDown(self):
        web_app.JM, web_app.UP = self._orig_jm, self._orig_up
        web_app.WORK.pop("ann1", None)

    def test_inherit_outputs_box_statuses_and_clears_target(self):
        web_app.claim_next(self.ann, self.job.job_id)
        sel = full_selection()
        for f in web_app.bbox_feats():
            sel[f.name] = "不存在"                 # 两项都「不存在」
        web_app.save_anno(self.ann, *radio_values_for(sel))
        web_app.claim_next(self.ann, self.job.job_id)   # 站到一张新道集上

        out = web_app.inherit_previous(self.ann)
        n_feat = len(CFG.features)
        statuses = out[1 + n_feat:]
        self.assertEqual(len(statuses), len(web_app.bbox_feats()),
                         "继承必须把框状态一起回吐")
        for s in statuses:
            self.assertIn("无需画框", s, statuses)
        # 服务端据此算出的目标也必须是空 —— 前端回读后就不该再能拉框
        self.assertIsNone(web_app._boxes_payload("ann1")["target"])

    def test_inherit_without_claim_is_guided(self):
        out = web_app.inherit_previous(self.ann)
        self.assertEqual(len(out), 1 + len(CFG.features) + len(web_app.bbox_feats()))
        self.assertIn("请先领取道集", out[0])


class SmokeWebMultiUser(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.sgy = os.path.join(self.root, "demo.sgy")
        write_sgy(self.sgy, n_gathers=3, traces_per=8, ns=64)
        # 替换模块级单例为隔离实例（test 结束还原）
        self._orig_jm, self._orig_up = web_app.JM, web_app.UP
        web_app.JM = web_app.make_jm(os.path.join(self.root, "jobs"))
        web_app.UP = Uploader({}, dry_run=True)
        self.job = web_app.JM.create_job(self.sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2],
                                         clip=99, title="demo", created_by="boss")

    def tearDown(self):
        web_app.JM, web_app.UP = self._orig_jm, self._orig_up
        # 清理测试用户的工作态，避免污染其他用例/真实运行
        for u in ("ann1", "boss"):
            web_app.WORK.pop(u, None)

    def test_two_users_claim_save_no_duplicate(self):
        ann, boss = _Req("ann1"), _Req("boss")
        web_app.claim_next(ann, self.job.job_id)
        gid_a = web_app.wstate("ann1")["gid"]
        self.assertIsNotNone(gid_a)

        out_a = web_app.save_anno(ann, *radio_values_for(full_selection()))
        self.assertIn("已保存", out_a[1])
        # 保存后应自动领取下一张（连续标注体验）
        self.assertIn("已自动领取下一张", out_a[1])
        gid_a2 = web_app.wstate("ann1")["gid"]
        self.assertIsNotNone(gid_a2)
        self.assertNotEqual(gid_a2, gid_a)
        web_app.release_current(ann)      # 释放自动领取的下一张，让 boss 领取

        web_app.claim_next(boss, self.job.job_id)
        gid_b = web_app.wstate("boss")["gid"]
        self.assertIsNotNone(gid_b)
        self.assertNotEqual(gid_a, gid_b)

        web_app.save_anno(boss, *radio_values_for(full_selection()))

        n_labeled, total = web_app.JM.progress(self.job.job_id)
        self.assertEqual((n_labeled, total), (2, 2))

        rec_a = web_app.JM.record(self.job.job_id, gid_a)
        rec_b = web_app.JM.record(self.job.job_id, gid_b)
        self.assertEqual(rec_a["annotated_by"], "ann1")
        self.assertEqual(rec_b["annotated_by"], "boss")

        # 保存成功后应入队 图片 + labels.jsonl（dry-run 记入 dry_log）
        keys = {k for k, _ in web_app.UP.dry_log}
        self.assertTrue(any(k.endswith("labels.jsonl") for k in keys))
        self.assertTrue(any("/images/" in k for k in keys))

    def test_claim_while_unsaved_keeps_gather_and_guides(self):
        """回归（“领取下一张不跳”）：持有未保存任务时再点领取，不得静默续领同一张，
        应返回明确指引且仍停留在当前张（gid 不变、不清空）。"""
        ann = _Req("ann1")
        web_app.claim_next(ann, self.job.job_id)
        gid = web_app.wstate("ann1")["gid"]
        self.assertIsNotNone(gid)
        out = web_app.claim_next(ann, self.job.job_id)   # 未保存，再点一次
        self.assertEqual(web_app.wstate("ann1")["gid"], gid)     # 未前进
        self.assertIn("尚未保存", out[1])                        # 有指引
        self.assertIn("保存并释放", out[1])

    def test_reopen_own_record_saves_via_owner(self):
        """重开自己标过的记录（无租约）仍可保存：renew 返回 None 但 record 已存在，
        save_anno 应走 owner 归属，而非误判为「已被他人领取」。"""
        ann = _Req("ann1")
        web_app.claim_next(ann, self.job.job_id)
        gid = web_app.wstate("ann1")["gid"]
        web_app.save_anno(ann, *radio_values_for(full_selection()))

        # 重开（此时无租约）
        web_app.reopen_mine(ann, [self.job.job_id], gid)
        self.assertEqual(web_app.wstate("ann1")["gid"], gid)

        # 再次保存：应成功（owner 归属），不是「已被他人领取」。
        # 提示语是「已覆盖」：重开已有记录再保存属于**修正**（照常自动领下一张）
        out = web_app.save_anno(ann, *radio_values_for(full_selection()))
        self.assertIn("已覆盖", out[1])
        self.assertNotIn("已被他人领取", out[1])
        self.assertEqual(web_app.JM.record(self.job.job_id, gid)["annotated_by"], "ann1")

    def test_reopen_with_box_then_radio_edit_resave(self):
        """回归：bbox 特征画了框后保存 → 重开（boxes 被 pop）→ 只改非 bbox 项 → 再保存，
        应成功且 regions 保留（save_anno 用 _current_boxes 回退到已存 regions）。"""
        ann = _Req("ann1")
        web_app.claim_next(ann, self.job.job_id)
        gid = web_app.wstate("ann1")["gid"]

        # 面波(bbox)=「存在」并画一个真实框
        sel = full_selection()
        sel["面波"] = "存在"
        web_app.wstate("ann1")["boxes"] = {
            "surface_wave": {"xyxy": [10, 20, 100, 200], "traces": [0, 2], "samples": [0, 8]},
        }
        out = web_app.save_anno(ann, *radio_values_for(sel))
        self.assertIn("已保存", out[1])

        # 重开：boxes 被 pop，应回退到已存 regions
        web_app.reopen_mine(ann, [self.job.job_id], gid)
        self.assertNotIn("boxes", web_app.wstate("ann1"))

        # 只改一个非 bbox 特征（集合类型→残差），再保存。
        # 提示语是「已覆盖」（重开已有记录再保存＝修正，原地停住不自动领下一张）
        sel2 = dict(sel)
        sel2["集合类型"] = "残差"
        out2 = web_app.save_anno(ann, *radio_values_for(sel2))
        self.assertIn("已覆盖", out2[1])

        rec = web_app.JM.record(self.job.job_id, gid)
        self.assertNotIn("surface_wave", rec["regions"])              # 残差分支不保存包络
        self.assertEqual(rec["labels"].get("gather_type"), "残差")  # 非 bbox 改动生效
        self.assertEqual(rec["annotated_by"], "ann1")                 # 原标注者保留


if __name__ == "__main__":
    unittest.main()
