# -*- coding: utf-8 -*-
"""smoke_web.py — web_app 标注 handler 层的多用户冒烟（不起 HTTP，避免 flaky）。

直接用 web_app 的 handler（claim_next / save_anno）模拟两个用户各自领取/保存，
断言：进度正确、不重复标注、记录带 annotated_by、云上传入队（dry-run）。
运行：python tests/smoke_web.py
"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 直接 `python tests/smoke_web.py` 时：先加项目根，再加 tests（segy_factory 所在）
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import web_app
from segy_factory import write_sgy
from jobmanager import JobManager
from cloudsync import Uploader
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
        web_app.JM = JobManager(os.path.join(self.root, "jobs"), CFG)
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
        return 3 + len(CFG.features) + len(web_app.bbox_feats()) + 4

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
        self.assertEqual(set(pay), {"boxes", "target"})
        self.assertEqual(pay["target"], "surface_wave")           # 第一个需框未框
        self.assertEqual(pay["boxes"], {})

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
        web_app.JM = JobManager(os.path.join(self.root, "jobs"), CFG)
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
        out = web_app.open_job_for(ann, self.job_a.job_id)
        st = web_app.wstate("ann1")
        self.assertIsNotNone(st.get("gid"), "选中作业应自动领到一张")
        self.assertIn(st["gid"], out[1])                   # 提示行里报出当前道集
        self.assertNotIn("请选择作业后点", out[1])

    def test_reselect_same_job_keeps_unsaved_work(self):
        """重复选中同一作业：不换张、也不清掉未保存的选择。"""
        ann = _Req("ann1")
        web_app.open_job_for(ann, self.job_a.job_id)
        gid = web_app.wstate("ann1")["gid"]
        web_app.wstate("ann1")["partial"] = {"集合类型": "炮集"}
        web_app.open_job_for(ann, self.job_a.job_id)
        self.assertEqual(web_app.wstate("ann1")["gid"], gid)
        self.assertEqual(web_app.wstate("ann1")["partial"], {"集合类型": "炮集"})

    def test_switching_job_claims_there(self):
        ann = _Req("ann1")
        web_app.open_job_for(ann, self.job_a.job_id)
        web_app.open_job_for(ann, self.job_b.job_id)
        st = web_app.wstate("ann1")
        self.assertEqual(st["job_id"], self.job_b.job_id)
        self.assertIsNotNone(st.get("gid"))

    def test_completed_job_reports_all_done_without_gid(self):
        ann = _Req("ann1")
        web_app.open_job_for(ann, self.job_c.job_id)       # 只有一张，先领到
        gid = web_app.wstate("ann1")["gid"]
        ok, rec, augs, err = web_app.JM.save(
            self.job_c.job_id, "ann1", gid, full_selection(),
            {f.key: {"xyxy": [1, 1, 2, 2], "traces": [0, 2], "samples": [0, 4]}
             for f in CFG.features if f.bbox})
        self.assertTrue(ok, err)
        web_app.WORK.pop("ann1", None)                     # 当作重新进页面
        out = web_app.open_job_for(ann, self.job_c.job_id)
        self.assertIsNone(web_app.wstate("ann1").get("gid"))
        self.assertIn("全部标注", out[1])

    def test_claim_button_still_works_after_release(self):
        """归还后仍可继续（自动领取不是唯一入口）。"""
        ann = _Req("ann1")
        web_app.open_job_for(ann, self.job_a.job_id)
        web_app.release_current(ann)
        self.assertIsNone(web_app.wstate("ann1").get("gid"))
        out = web_app.claim_next(ann, self.job_a.job_id)
        self.assertIsNotNone(web_app.wstate("ann1").get("gid"))
        self.assertEqual(len(out), 3 + len(CFG.features) + len(web_app.bbox_feats()) + 4)


class SmokeAnnoOutputArity(unittest.TestCase):
    """anno_outputs 契约：每个事件返回的值个数必须与输出列表一致。

    多一个/少一个只会在浏览器点击时才崩，故在此把每个 handler 都过一遍。
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.sgy = os.path.join(self.root, "demo.sgy")
        write_sgy(self.sgy, n_gathers=3, traces_per=8, ns=64)
        self._orig_jm, self._orig_up = web_app.JM, web_app.UP
        web_app.JM = JobManager(os.path.join(self.root, "jobs"), CFG)
        web_app.UP = Uploader({}, dry_run=True)
        self.job = web_app.JM.create_job(self.sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2, 3],
                                         clip=99, title="demo", created_by="boss")

    def tearDown(self):
        web_app.JM, web_app.UP = self._orig_jm, self._orig_up
        web_app.WORK.pop("ann1", None)

    def _n_anno(self):
        # img + info + sentence + 各 radio + 各 bbox 状态 + 4 个滤波参数
        return 3 + len(CFG.features) + len(web_app.bbox_feats()) + 4

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
        # 继承只输出句子 + 各 radio
        self.assertEqual(len(web_app.inherit_previous(ann)), 1 + len(CFG.features),
                         "inherit_previous")
        # open_job_for 额外多一个「我标注的」下拉
        self.assertEqual(len(web_app.open_job_for(ann, self.job.job_id)),
                         n + 1, "open_job_for")
        # 空态/无作业分支同样要对齐
        web_app.WORK.pop("ann1", None)
        self.assertEqual(len(web_app.refresh_anno(ann)), n, "refresh_anno 空态")
        self.assertEqual(len(web_app.toggle_filter(ann, 20, 30, 80, 100)), n,
                         "toggle_filter 空态")


class SmokeWebMultiUser(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.sgy = os.path.join(self.root, "demo.sgy")
        write_sgy(self.sgy, n_gathers=3, traces_per=8, ns=64)
        # 替换模块级单例为隔离实例（test 结束还原）
        self._orig_jm, self._orig_up = web_app.JM, web_app.UP
        web_app.JM = JobManager(os.path.join(self.root, "jobs"), CFG)
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
        web_app.reopen_mine(ann, self.job.job_id, gid)
        self.assertEqual(web_app.wstate("ann1")["gid"], gid)

        # 再次保存：应成功（owner 归属），不是「已被他人领取」
        out = web_app.save_anno(ann, *radio_values_for(full_selection()))
        self.assertIn("已保存", out[1])
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
        web_app.reopen_mine(ann, self.job.job_id, gid)
        self.assertNotIn("boxes", web_app.wstate("ann1"))

        # 只改一个非 bbox 特征（集合类型→CMP道集），再保存
        sel2 = dict(sel)
        sel2["集合类型"] = "CMP道集"
        out2 = web_app.save_anno(ann, *radio_values_for(sel2))
        self.assertIn("已保存", out2[1])

        rec = web_app.JM.record(self.job.job_id, gid)
        self.assertIsNotNone(rec["regions"].get("surface_wave"))      # 框保留
        self.assertEqual(rec["labels"].get("gather_type"), "CMP道集")  # 非 bbox 改动生效
        self.assertEqual(rec["annotated_by"], "ann1")                 # 原标注者保留


if __name__ == "__main__":
    unittest.main()
