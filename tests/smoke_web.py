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
