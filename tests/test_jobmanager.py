# -*- coding: utf-8 -*-
import os, sys, tempfile, time, unittest
import numpy as np
# 让本文件能导入同目录的 segy_factory（python -m unittest tests.test_jobmanager 时 tests/ 不在 sys.path）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segy_factory import write_sgy
from gather import Gather
from labels import LabelConfig
from jobmanager import open_job, JobManager, finalize_regions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = LabelConfig(os.path.join(ROOT, "label_config.yaml"))

def meta_for(job_id, out):
    return {"job_id": job_id, "title": "测试", "source_file": "x.sgy", "endian": "auto",
            "sort_keys": [[95, 96]], "extract_keys": [[95, 96]], "values": [1, 2, 3],
            "clip_percentile": 99.0, "created_by": "boss",
            "created_at": "2026-01-01T00:00:00", "state": "open", "output_dir": out}

def fake_gathers():
    gs = []
    for v in (1, 2, 3):
        g = Gather(key="95-96", value=v, trace_indices=np.arange(40)); g.prefix = "fake__"
        gs.append(g)
    return gs

def fake_provider(ns=200):
    def prov(g):
        rng = np.random.default_rng(abs(hash(g.gather_id)) % 2**32)
        return rng.normal(size=(g.n_traces, ns)).astype(np.float32)
    return prov

class TestJobAndManager(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.manager = JobManager(self.root, CFG)
        self.out = os.path.join(self.root, "job1")
        os.makedirs(self.out)
        meta = meta_for("job1", self.out)
        self.gs = fake_gathers()
        self.job = self.manager.register_job(meta, self.gs, 200, fake_provider())

    def test_gather_ids_and_meta(self):
        self.assertEqual(self.job.gather_ids(), ["fake__95-96_1", "fake__95-96_2", "fake__95-96_3"])
        m = self.job.gather_meta("fake__95-96_1")
        self.assertEqual(m["n_traces"], 40)

    def test_progress_zero_then_after_upsert(self):
        self.assertEqual(self.manager.progress("job1"), (0, 3))
        rec = {"gather_id": "fake__95-96_1"}
        self.job.store.upsert(rec)
        self.assertEqual(self.manager.progress("job1"), (1, 3))

    def test_finalize_regions_requires_box_when_not_absent(self):
        selection = {f.name: "存在" for f in CFG.features}
        regions, errs = finalize_regions(CFG, selection, {})
        self.assertTrue(any("面波" in e for e in errs))   # bbox 特征缺框
        selection["面波"] = "不存在"
        selection["近炮点强能量噪声"] = "不存在"
        regions, errs = finalize_regions(CFG, selection, {})
        self.assertEqual(errs, [])
        self.assertIsNone(regions["surface_wave"])
        self.assertIsNone(regions["near_shot_noise"])

    def test_list_jobs_info(self):
        info = self.manager.list_jobs()[0]
        self.assertEqual(info["job_id"], "job1")
        self.assertEqual(info["total"], 3)

class TestClaimLease(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.manager = JobManager(self.root, CFG, ttl_seconds=0.4)
        self.out = os.path.join(self.root, "job1"); os.makedirs(self.out)
        meta = meta_for("job1", self.out)
        self.job = self.manager.register_job(meta, fake_gathers(), 200, fake_provider())

    def test_claim_gives_distinct_and_resume(self):
        a = self.manager.claim("job1", "ann1"); b = self.manager.claim("job1", "ann2")
        self.assertIsNotNone(a["gid"]); self.assertIsNotNone(b["gid"])
        self.assertNotEqual(a["gid"], b["gid"])
        again = self.manager.claim("job1", "ann1")     # 已持有 → resume 同一张
        self.assertTrue(again["resume"]); self.assertEqual(again["gid"], a["gid"])

    def test_release_returns_to_pool(self):
        a = self.manager.claim("job1", "ann1")
        self.assertTrue(self.manager.release("job1", "ann1"))
        b = self.manager.claim("job1", "ann2")
        self.assertEqual(b["gid"], a["gid"])           # 归还能被另一个人领到

    def test_ttl_expiry_frees_lease(self):
        a = self.manager.claim("job1", "ann1")
        time.sleep(0.6)                                # 超过 ttl
        b = self.manager.claim("job1", "ann2")
        self.assertEqual(b["gid"], a["gid"])           # 过期后回池

    def test_no_double_claim_concurrently(self):
        # 单用户并发 claim 不得超过池容量去重；两个用户各自 save 后互不重复
        got = [self.manager.claim("job1", u) for u in ("ann1", "ann2") for _ in range(3)]
        gids = [x["gid"] for x in got if x["gid"]]
        self.assertLessEqual(len(set(gids)), 3)

class TestSave(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.manager = JobManager(self.root, CFG)
        self.out = os.path.join(self.root, "job1"); os.makedirs(self.out)
        meta = meta_for("job1", self.out)
        self.job = self.manager.register_job(meta, fake_gathers(), 200, fake_provider())

    def full_selection(self):
        # 每个特征取合法选项；bbox 特征取「不存在」以免画框（部分特征无「不存在」选项则取首项）
        return {f.name: ("不存在" if f.option_by_label("不存在") else f.options[0].label)
                for f in CFG.features}   # 无 bbox 需求
    def good_boxes(self):
        return {f.key: {"xyxy": [1,1,2,2], "traces": [0,2], "samples": [0,4]} for f in CFG.features if f.bbox}

    def test_save_requires_valid_selection(self):
        claim = self.manager.claim("job1", "ann1")
        ok, rec, err = self.manager.save("job1", "ann1", claim["gid"], {}, {}, is_admin=False)
        self.assertFalse(ok); self.assertIn("未选择", err)

    def test_save_success_sets_annotated_by_and_image(self):
        claim = self.manager.claim("job1", "ann1")
        ok, rec, err = self.manager.save("job1", "ann1", claim["gid"],
                                         self.full_selection(), self.good_boxes())
        self.assertTrue(ok, err); self.assertEqual(rec["annotated_by"], "ann1")
        img = os.path.join(self.out, rec["image_path"])
        self.assertTrue(os.path.isfile(img))            # 渲染缓存已生成
        self.assertTrue(os.path.isfile(os.path.join(self.out, rec["npy_path"])))
        self.assertTrue(self.manager.progress("job1")[0] >= 1)

    def test_other_user_cannot_save_someone_elses_claimed(self):
        a = self.manager.claim("job1", "ann1")
        ok, rec, err = self.manager.save("job1", "ann2", a["gid"],
                                         self.full_selection(), self.good_boxes())
        self.assertFalse(ok); self.assertIn("已被", err)

    def test_owner_can_resave_and_admin_can_override(self):
        a = self.manager.claim("job1", "ann1")
        self.manager.save("job1", "ann1", a["gid"], self.full_selection(), self.good_boxes())
        ok, rec, err = self.manager.save("job1", "ann1", a["gid"],
                                         self.full_selection(), self.good_boxes())
        self.assertTrue(ok, err)                        # 本人回改
        self.assertEqual(rec["annotated_by"], "ann1")   # 回改保留原标注者
        ok, rec, err = self.manager.save("job1", "boss", a["gid"],
                                         self.full_selection(), self.good_boxes(), is_admin=True)
        self.assertTrue(ok, err)                        # admin 覆盖
        self.assertEqual(rec["annotated_by"], "ann1")   # 覆盖时仍保留原标注者（spec §5.4）

class TestImageExportOnlyOnSave(unittest.TestCase):
    """回归：显示/领取 不得向导出 images/ 目录写图；只有保存才生成导出图。

    bug: web_app.render_display(领取、保存后自动领下一张都会触发显示) 走 ensure_image，
    把尚未标注的道集直接写成导出图 images/<gid>.png —— 造成「标一张却出现两张结果图」、
    images/ 出现无任何 jsonl 记录的孤立图、导出图与标注记录永远对不上。
    """
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.manager = JobManager(self.root, CFG)
        self.out = os.path.join(self.root, "job1"); os.makedirs(self.out)
        meta = meta_for("job1", self.out)
        self.job = self.manager.register_job(meta, fake_gathers(), 200, fake_provider())

    def pngs_in_images(self):
        d = os.path.join(self.out, "images")
        if not os.path.isdir(d):
            return []
        return sorted(f for f in os.listdir(d) if f.endswith(".png"))

    def sel_boxes(self):
        sel = {f.name: ("不存在" if f.option_by_label("不存在") else f.options[0].label)
               for f in CFG.features}
        boxes = {f.key: {"xyxy": [1, 1, 2, 2], "traces": [0, 2], "samples": [0, 4]}
                 for f in CFG.features if f.bbox}
        return sel, boxes

    def test_display_of_claimed_unlabeled_gather_does_not_export(self):
        g1 = self.manager.claim("job1", "ann1")["gid"]
        self.assertTrue(self.job.display_image(g1))      # 显示一张未标注道集
        self.assertEqual(self.pngs_in_images(), [])      # 不得生成导出图

    def test_one_save_exports_only_its_own_image(self):
        g1 = self.manager.claim("job1", "ann1")["gid"]
        self.job.display_image(g1)                       # 领取即显示
        sel, boxes = self.sel_boxes()
        ok, rec, err = self.manager.save("job1", "ann1", g1, sel, boxes)
        self.assertTrue(ok, err)
        # 保存后自动领取并显示下一张（save_anno 尾部行为）
        g2 = self.manager.claim("job1", "ann1")["gid"]
        self.job.display_image(g2)
        self.assertNotEqual(g1, g2)
        self.assertEqual(self.pngs_in_images(),
                         [os.path.basename(rec["image_path"])])   # 只有刚保存那张

    def test_display_and_export_render_same_content(self):
        g1 = self.manager.claim("job1", "ann1")["gid"]
        shown = self.job.display_image(g1)
        self.assertTrue(os.path.isfile(shown))
        self.assertFalse(os.path.abspath(shown).startswith(
            os.path.abspath(os.path.join(self.out, "images"))))   # 显示图在 .cache，不在导出目录


class TestClaimShuffledPool(unittest.TestCase):
    """回归：任务领取按打乱后的池下发，任一标注者拿到的是乱序道集，
    而不是按数据值连续的顺序段（避免标注疲劳）。"""
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.manager = JobManager(self.root, CFG)

    def _job_with(self, n_values: int):
        out = os.path.join(self.root, "job1"); os.makedirs(out)
        meta = meta_for("job1", out)
        gs = []
        for v in range(1, n_values + 1):
            g = Gather(key="95-96", value=v, trace_indices=np.arange(40))
            g.prefix = "fake__"
            gs.append(g)
        return self.manager.register_job(meta, gs, 200, fake_provider()), out

    def test_one_user_gets_shuffled_not_ascending_order(self):
        n = 12
        job, out = self._job_with(n)
        sel = {f.name: ("不存在" if f.option_by_label("不存在") else f.options[0].label)
               for f in CFG.features}
        boxes = {f.key: {"xyxy": [1, 1, 2, 2], "traces": [0, 2], "samples": [0, 4]}
                 for f in CFG.features if f.bbox}
        got = []
        for _ in range(n):
            c = self.manager.claim("job1", "ann1")
            self.assertIsNotNone(c["gid"], "应能一直领取到剩余道集")
            v = int(c["gid"].rsplit("_", 1)[1])
            got.append(v)
            ok, rec, err = self.manager.save("job1", "ann1", c["gid"], sel, boxes)
            self.assertTrue(ok, err)
        self.assertEqual(sorted(got), list(range(1, n + 1)))   # 每张恰好一次（全集排列）
        self.assertNotEqual(got, list(range(1, n + 1)))        # 不再按原始顺序连续下发


class TestCreateAndRestore(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.sgy = os.path.join(self.root, "demo.sgy")
        write_sgy(self.sgy, n_gathers=3, traces_per=8, ns=64)

    def test_create_job_persists_and_restores(self):
        m = JobManager(os.path.join(self.root, "jobs"), CFG)
        # 注: create_job 的 extract_keys 契约是「区间列表」(如 [[95,96]]/[(95,96)])，
        # 与 load_all/save 读 meta["extract_keys"] 的形态一致；单字段不能传裸元组 (95,96)。
        job = m.create_job(self.sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2], clip=99,
                           title="demo", created_by="boss")
        self.assertEqual(len(job.gather_ids()), 2)
        jf = os.path.join(job.output_dir, "job.json")
        self.assertTrue(os.path.isfile(jf))
        # 用全新 manager 走启动恢复
        m2 = JobManager(os.path.join(self.root, "jobs"), CFG)
        m2.load_all()
        info = m2.list_jobs()[0]
        self.assertEqual(info["job_id"], job.job_id)
        self.assertEqual(info["total"], 2)
        # 恢复后仍可领取/保存（回归渲染需读取源数据）
        claim = m2.claim(job.job_id, "ann1")
        self.assertEqual(m2.progress(job.job_id), (0, 2))
        # 注: 集合类型/静校正 无「不存在」选项，仅对含该选项的特征置「不存在」，其余取合法首项
        # （与 TestSave.full_selection / Task4 裁决一致；bbox 特征置「不存在」免画框）
        sel = {f.name: ("不存在" if f.option_by_label("不存在") else f.options[0].label)
               for f in CFG.features}
        boxes = {f.key: {"xyxy": [0, 0, 3, 5], "traces": [0, 3], "samples": [0, 6]} for f in CFG.features if f.bbox}
        ok, rec, err = m2.save(job.job_id, "ann1", claim["gid"], sel, boxes)
        self.assertTrue(ok, err)
        self.assertTrue(os.path.isfile(os.path.join(job.output_dir, rec["image_path"])))

if __name__ == "__main__":
    unittest.main()
