# -*- coding: utf-8 -*-
import json, os, sys, tempfile, time, unittest
import numpy as np
# 让本文件能导入同目录的 segy_factory（python -m unittest tests.test_jobmanager 时 tests/ 不在 sys.path）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segy_factory import write_sgy, _bin_header, HDR
from gather import Gather
from labels import LabelConfig
from jobmanager import open_job, JobManager, JobError, finalize_regions
from preprocess import clip_bound

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = LabelConfig(os.path.join(ROOT, "label_config.yaml"))

def meta_for(job_id, out, aug=True):
    """默认带新版增强字段（aug_lo/aug_hi）；aug=False 模拟旧版作业（需重建）。"""
    m = {"job_id": job_id, "title": "测试", "source_file": "x.sgy", "endian": "auto",
         "sort_keys": [[95, 96]], "extract_keys": [[95, 96]], "values": [1, 2, 3],
         "clip_percentile": 99.0, "dt_ms": 2.0, "created_by": "boss",
         "created_at": "2026-01-01T00:00:00", "state": "open", "output_dir": out}
    if aug:
        m.update({"aug_lo": 90.0, "aug_hi": 99.9})
    return m

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

    def pngs_in_images(self):
        d = os.path.join(self.out, "images")
        if not os.path.isdir(d):
            return []
        return sorted(f for f in os.listdir(d) if f.endswith(".png"))

    def test_save_requires_valid_selection(self):
        claim = self.manager.claim("job1", "ann1")
        ok, rec, augs, err = self.manager.save("job1", "ann1", claim["gid"], {}, {},
                                               is_admin=False)
        self.assertFalse(ok); self.assertIsNone(rec); self.assertEqual(augs, [])
        self.assertIn("未选择", err)

    def test_save_writes_five_aug_images_and_records(self):
        claim = self.manager.claim("job1", "ann1")
        gid = claim["gid"]
        ok, rec, augs, err = self.manager.save("job1", "ann1", gid,
                                               self.full_selection(), self.good_boxes())
        self.assertTrue(ok, err)
        # 基础记录：标注者归属、无导出图、保留 npy
        self.assertEqual(rec["annotated_by"], "ann1")
        self.assertEqual(rec["gather_id"], gid)
        self.assertIsNone(rec["image_path"])
        self.assertTrue(os.path.isfile(os.path.join(self.out, rec["npy_path"])))
        # 5 条增强记录 + 5 张图，clip 各不同、labels 与基础一致
        self.assertEqual(len(augs), 5)
        clips = []
        for a in augs:
            self.assertTrue(os.path.isfile(os.path.join(self.out, a["image_path"])))
            self.assertEqual(a["augmented_from"], gid)
            self.assertEqual(a["labels"], rec["labels"])
            self.assertEqual(a["sentence"], rec["sentence"])
            self.assertEqual(a["regions"], rec["regions"])
            self.assertEqual(a["annotated_by"], "ann1")
            clips.append(a["clip_percentile"])
            self.assertTrue(a["image_path"].startswith("images/" + gid + "__clip"))
        self.assertEqual(len(set(clips)), 5)               # 5 个 clip 互不相同
        self.assertTrue(all(90.0 <= c <= 99.9 for c in clips))
        self.assertEqual(len(self.pngs_in_images()), 5)    # 只多这 5 张导出图
        self.assertTrue(self.manager.progress("job1")[0] >= 1)   # 进度只按基础计数
        # 6 条记录 = 1 基础 + 5 增强
        self.assertEqual(len(self.job.store.records), 6)

    def test_resave_does_not_accumulate_aug(self):
        claim = self.manager.claim("job1", "ann1")
        gid = claim["gid"]
        for _ in range(2):
            ok, rec, augs, err = self.manager.save("job1", "ann1", gid,
                                                   self.full_selection(), self.good_boxes())
            self.assertTrue(ok, err)
            self.assertEqual(len(augs), 5)
        self.assertEqual(len(self.job.store.records), 6)     # 重存后仍 1 基础 + 5 增强
        self.assertEqual(len(self.pngs_in_images()), 5)

    def test_save_rejects_legacy_job_without_aug(self):
        # 旧版作业（job.json 无 aug_lo/hi）：一律重建，禁止保存
        out2 = os.path.join(self.root, "job_legacy"); os.makedirs(out2)
        meta = meta_for("job_legacy", out2, aug=False)
        self.manager.register_job(meta, fake_gathers(), 200, fake_provider())
        c = self.manager.claim("job_legacy", "ann1")
        ok, rec, augs, err = self.manager.save("job_legacy", "ann1", c["gid"],
                                               self.full_selection(), self.good_boxes())
        self.assertFalse(ok); self.assertIn("重新创建", err)

    def test_save_narrow_clip_range_fails(self):
        out3 = os.path.join(self.root, "job_narrow"); os.makedirs(out3)
        meta = meta_for("job_narrow", out3)
        meta.update({"aug_lo": 99.5, "aug_hi": 99.5})        # 只 1 个可选值，采不出 5 个
        self.manager.register_job(meta, fake_gathers(), 200, fake_provider())
        c = self.manager.claim("job_narrow", "ann1")
        ok, rec, augs, err = self.manager.save("job_narrow", "ann1", c["gid"],
                                               self.full_selection(), self.good_boxes())
        self.assertFalse(ok); self.assertIn("过窄", err)

    def test_other_user_cannot_save_someone_elses_claimed(self):
        a = self.manager.claim("job1", "ann1")
        ok, rec, augs, err = self.manager.save("job1", "ann2", a["gid"],
                                               self.full_selection(), self.good_boxes())
        self.assertFalse(ok); self.assertIn("已被", err)

    def test_owner_can_resave_and_admin_can_override(self):
        a = self.manager.claim("job1", "ann1")
        self.manager.save("job1", "ann1", a["gid"], self.full_selection(), self.good_boxes())
        ok, rec, augs, err = self.manager.save("job1", "ann1", a["gid"],
                                               self.full_selection(), self.good_boxes())
        self.assertTrue(ok, err)                        # 本人回改
        self.assertEqual(rec["annotated_by"], "ann1")   # 回改保留原标注者
        ok, rec, augs, err = self.manager.save("job1", "boss", a["gid"],
                                               self.full_selection(), self.good_boxes(),
                                               is_admin=True)
        self.assertTrue(ok, err)                        # admin 覆盖
        self.assertEqual(rec["annotated_by"], "ann1")   # 覆盖时仍保留原标注者（spec §5.4）

class TestImageExportOnlyOnSave(unittest.TestCase):
    """回归：显示/领取 不得向导出 images/ 目录写图；只有保存才生成增强导出图。

    bug: web_app.render_display(领取、保存后自动领下一张都会触发显示) 走导出图，
    把尚未标注的道集直接写成导出图 —— 造成「标一张却出现多张结果图」、
    images/ 出现无任何 jsonl 记录的孤立图、导出图与标注记录永远对不上。
    v3.6 起保存导出 N_AUG=5 张 <gid>__clip<值>.png，显示仍只走 .cache。
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

    def test_save_exports_only_own_gather_aug_images(self):
        g1 = self.manager.claim("job1", "ann1")["gid"]
        self.job.display_image(g1)                       # 领取即显示
        sel, boxes = self.sel_boxes()
        ok, rec, augs, err = self.manager.save("job1", "ann1", g1, sel, boxes)
        self.assertTrue(ok, err)
        # 保存后自动领取并显示下一张（save_anno 尾部行为）
        g2 = self.manager.claim("job1", "ann1")["gid"]
        self.job.display_image(g2)
        self.assertNotEqual(g1, g2)
        self.assertEqual(len(self.pngs_in_images()), 5)                 # 刚保存那张的 5 张增强图
        names = {os.path.basename(a["image_path"]) for a in augs}
        self.assertTrue(names <= set(self.pngs_in_images()))           # 记录与图一一对应
        self.assertFalse(any(not n.startswith(g1 + "__clip") for n in self.pngs_in_images()))

    def test_display_image_lives_in_cache_not_export(self):
        g1 = self.manager.claim("job1", "ann1")["gid"]
        shown = self.job.display_image(g1)
        self.assertTrue(os.path.isfile(shown))
        self.assertFalse(os.path.abspath(shown).startswith(
            os.path.abspath(os.path.join(self.out, "images"))))   # 显示图在 .cache，不在导出目录


class TestSkip(unittest.TestCase):
    """回归：跳过 = 当前道集回池（不写记录）+ 自动领取下一张（非刚跳过的那张）。"""
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.manager = JobManager(self.root, CFG)
        self.out = os.path.join(self.root, "job1"); os.makedirs(self.out)
        meta = meta_for("job1", self.out)
        self.job = self.manager.register_job(meta, fake_gathers(), 200, fake_provider())

    def test_skip_releases_and_claims_another(self):
        first = self.manager.claim("job1", "ann1")["gid"]
        out = self.manager.skip("job1", "ann1")
        self.assertIsNotNone(out["gid"])
        self.assertNotEqual(out["gid"], first)          # 自动领的是另一张
        self.assertEqual(out["skipped"], first)
        # 被跳过的那张回池、未标注：别人能领到
        other = self.manager.claim("job1", "ann2")
        self.assertEqual(other["gid"], first)
        self.assertIsNone(self.manager.record("job1", first))   # 没有写记录

    def test_skip_single_remaining_returns_it_back(self):
        # 先标完两张，只剩第三张；ann2 持有它，跳过只能退回同一张（仍有任务可做）
        sel = {f.name: ("不存在" if f.option_by_label("不存在") else f.options[0].label)
               for f in CFG.features}
        boxes = {f.key: {"xyxy": [1, 1, 2, 2], "traces": [0, 2], "samples": [0, 4]}
                 for f in CFG.features if f.bbox}
        for _ in range(2):
            c = self.manager.claim("job1", "ann1")
            ok, rec, augs, err = self.manager.save("job1", "ann1", c["gid"], sel, boxes)
            self.assertTrue(ok, err)
        hold = self.manager.claim("job1", "ann2")      # 只剩最后一张，ann2 拿到
        self.assertEqual(self.manager.progress("job1"), (2, 3))
        out = self.manager.skip("job1", "ann2")
        self.assertEqual(out["gid"], hold["gid"])      # 只有它，退回继续做
        self.assertIsNone(self.manager.record("job1", hold["gid"]))


    def test_repeated_skips_do_not_oscillate(self):
        # 4 个未标注道集、单人连点跳过：应在不同的道集间推进，而不是 A<->B 循环
        root = tempfile.mkdtemp()
        mgr = JobManager(root, CFG)
        out = os.path.join(root, "job1"); os.makedirs(out)
        meta = meta_for("job1", out)
        gs = []
        for v in range(1, 5):
            g = Gather(key="95-96", value=v, trace_indices=np.arange(40))
            g.prefix = "fake__"
            gs.append(g)
        mgr.register_job(meta, gs, 200, fake_provider())
        c0 = mgr.claim("job1", "ann1")["gid"]
        s1 = mgr.skip("job1", "ann1")
        s2 = mgr.skip("job1", "ann1")
        self.assertEqual(s1["skipped"], c0)
        self.assertIsNotNone(s1["gid"])
        self.assertIsNotNone(s2["gid"])
        self.assertNotEqual(s1["gid"], c0)
        self.assertNotIn(s2["gid"], {c0, s1["gid"]})   # 第 3 次不再回到前两张

    def test_skip_then_manual_claim_avoids_recently_skipped(self):
        root = tempfile.mkdtemp()
        mgr = JobManager(root, CFG)
        out = os.path.join(root, "job1"); os.makedirs(out)
        meta = meta_for("job1", out)
        gs = []
        for v in range(1, 5):
            g = Gather(key="95-96", value=v, trace_indices=np.arange(40))
            g.prefix = "fake__"
            gs.append(g)
        mgr.register_job(meta, gs, 200, fake_provider())
        c0 = mgr.claim("job1", "ann1")["gid"]
        mgr.skip("job1", "ann1")                        # 跳过 c0 → 拿到另一张
        mgr.release("job1", "ann1")                     # 放掉当前这张后手动再领
        again = mgr.claim("job1", "ann1")               # 不应马上领回刚跳过的 c0
        self.assertIsNotNone(again["gid"])
        self.assertNotEqual(again["gid"], c0)


class TestDeleteJob(unittest.TestCase):
    """软删除：从列表隐藏、不可领取；恢复后回到 open；本地文件保留。"""
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.manager = JobManager(self.root, CFG)
        self.out = os.path.join(self.root, "job1"); os.makedirs(self.out)
        meta = meta_for("job1", self.out)
        self.manager.register_job(meta, fake_gathers(), 200, fake_provider())

    def test_delete_hides_from_list_and_blocks_claim(self):
        self.assertEqual([j["job_id"] for j in self.manager.list_jobs()], ["job1"])
        self.manager.delete_job("job1")
        self.assertEqual(self.manager.list_jobs(), [])                    # 列表隐藏
        self.assertEqual([j["job_id"] for j in self.manager.deleted_jobs()], ["job1"])
        out = self.manager.claim("job1", "ann1")
        self.assertIsNone(out["gid"])                                     # 不可领取
        self.assertTrue(os.path.isdir(self.out))                          # 本地文件保留

    def test_restore_returns_job_and_allows_claim(self):
        self.manager.delete_job("job1")
        self.manager.restore_job("job1")
        self.assertEqual([j["job_id"] for j in self.manager.list_jobs()], ["job1"])
        self.assertEqual(self.manager.deleted_jobs(), [])
        out = self.manager.claim("job1", "ann1")
        self.assertIsNotNone(out["gid"])                                  # 恢复后可领取


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
            ok, rec, augs, err = self.manager.save("job1", "ann1", c["gid"], sel, boxes)
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
        ok, rec, augs, err = m2.save(job.job_id, "ann1", claim["gid"], sel, boxes)
        self.assertTrue(ok, err)
        self.assertIsNone(rec["image_path"])                     # 基础记录无导出图
        self.assertEqual(len(augs), 5)
        for a in augs:
            self.assertTrue(os.path.isfile(os.path.join(job.output_dir, a["image_path"])))
        # 旧版作业（job.json 无 aug_lo/hi）在启动恢复时一律标 broken、不可领取
        legacy_out = os.path.join(self.root, "jobs", "legacy__x"); os.makedirs(legacy_out)
        legacy_meta = meta_for("legacy__x", legacy_out, aug=False)
        legacy_meta.update({"source_file": self.sgy, "extract_keys": [[95, 96]],
                            "values": [1, 2]})
        with open(os.path.join(legacy_out, "job.json"), "w", encoding="utf-8") as f:
            json.dump(legacy_meta, f, ensure_ascii=False)
        m3 = JobManager(os.path.join(self.root, "jobs"), CFG)
        m3.load_all()
        info = next(j for j in m3.list_jobs() if j["job_id"] == "legacy__x")
        self.assertEqual(info["state"], "broken")
        self.assertIsNone(m3.claim("legacy__x", "ann1")["gid"])

class TestCreateMinTraces(unittest.TestCase):
    """建作业时按「道数下限」滤去道数过少的道集。"""

    @staticmethod
    def _write_var_sgy(path, counts, ns=64):
        """造一个每道集道数不同的小端 sgy（95-96=炮号, 9-12=FFID, 13-16=道号）。"""
        e = "<"
        rows = []
        gid = 0
        for gv, c in enumerate(counts, start=1):
            for t in range(c):
                hdr = bytearray(HDR)
                hdr[94:96] = np.array([gv], dtype=e + "i2").tobytes()
                hdr[8:12] = np.array([37000 + gid], dtype=e + "i4").tobytes()
                hdr[12:16] = np.array([t + 1], dtype=e + "i4").tobytes()
                data = np.random.default_rng(gid).normal(size=ns).astype(np.float32)
                rows.append(bytes(hdr) + data.tobytes())
                gid += 1
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\x00" * 3200)
            f.write(_bin_header(ns, sum(counts), "little"))
            f.write(b"".join(rows))

    def test_min_traces_filters_low_count_gathers(self):
        root = tempfile.mkdtemp()
        sgy = os.path.join(root, "v.sgy")
        self._write_var_sgy(sgy, [2, 8])            # 炮1=2道，炮2=8道
        jobs_root = os.path.join(root, "jobs")
        m = JobManager(jobs_root, CFG)
        job = m.create_job(sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2],
                           clip=99, title="v", created_by="boss", min_traces=5)
        ids = job.gather_ids()
        self.assertEqual(len(ids), 1)               # 2道的炮1被滤去，只剩炮2
        self.assertIn("_95-96_2", ids[0])
        self.assertEqual(job.meta["min_traces"], 5)
        # 重启恢复后仍按 min_traces 过滤
        m2 = JobManager(jobs_root, CFG)
        m2.load_all()
        self.assertEqual(m2.list_jobs()[0]["total"], 1)

    def test_min_traces_zero_keeps_all(self):
        root = tempfile.mkdtemp()
        sgy = os.path.join(root, "v.sgy")
        self._write_var_sgy(sgy, [2, 8])
        m = JobManager(os.path.join(root, "jobs"), CFG)
        job = m.create_job(sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2],
                           clip=99, title="v", created_by="boss", min_traces=0)
        self.assertEqual(len(job.gather_ids()), 2)

    def test_min_traces_too_high_raises(self):
        root = tempfile.mkdtemp()
        sgy = os.path.join(root, "v.sgy")
        self._write_var_sgy(sgy, [2, 8])
        m = JobManager(os.path.join(root, "jobs"), CFG)
        with self.assertRaises(JobError):
            m.create_job(sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2],
                         clip=99, title="v", created_by="boss", min_traces=50)

    def test_filter_min_traces_unit(self):
        from jobmanager import JobManager as _J
        gs = []
        for k in (4, 8, 12, 3):
            g = Gather(key="95-96", value=k, trace_indices=np.arange(k))
            gs.append(g)
        kept = _J._filter_min_traces(gs, 5)
        self.assertEqual([g.n_traces for g in kept], [8, 12])   # 4、3 被滤
        self.assertEqual(len(_J._filter_min_traces(gs, 0)), 4)  # 0=不过滤


class TestDecimate(unittest.TestCase):
    """建作业「抽稀间隔 N」：按抽取顺序每 N 个保留 1 个，再进任务池。"""

    @staticmethod
    def _write_even_sgy(path, n_gathers=6, traces_per=8, ns=64):
        write_sgy(path, n_gathers=n_gathers, traces_per=traces_per, ns=ns)

    def test_decimate_unit(self):
        from jobmanager import JobManager as _J
        gs = [Gather(key="95-96", value=k, trace_indices=np.arange(8)) for k in range(6)]
        self.assertEqual([g.value for g in _J._decimate(gs, 2)], [0, 2, 4])
        self.assertEqual([g.value for g in _J._decimate(gs, 3)], [0, 3])
        self.assertEqual(len(_J._decimate(gs, 1)), 6)      # 1 = 不抽稀
        self.assertEqual(len(_J._decimate(gs, 0)), 6)      # 0 = 不抽稀
        self.assertEqual(len(_J._decimate(gs, None)), 6)   # 缺省 = 不抽稀

    def test_create_job_decimates_and_persists(self):
        root = tempfile.mkdtemp()
        sgy = os.path.join(root, "demo.sgy")
        self._write_even_sgy(sgy, n_gathers=6)
        jobs_root = os.path.join(root, "jobs")
        m = JobManager(jobs_root, CFG)
        job = m.create_job(sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2, 3, 4, 5, 6],
                           clip=99, title="d", created_by="boss", decimate_n=2)
        ids = job.gather_ids()
        self.assertEqual(len(ids), 3)                      # 6 → 留第 1/3/5 个
        self.assertTrue(ids[0].endswith("_95-96_1"))
        self.assertTrue(ids[1].endswith("_95-96_3"))
        self.assertTrue(ids[2].endswith("_95-96_5"))
        self.assertEqual(job.meta["decimate_n"], 2)
        # 重启恢复后池子必须与建作业时一致
        m2 = JobManager(jobs_root, CFG)
        m2.load_all()
        self.assertEqual(m2.list_jobs()[0]["total"], 3)
        self.assertEqual(sorted(m2.get(job.job_id).gather_ids()), sorted(ids))

    def test_create_job_decimate_one_keeps_all(self):
        root = tempfile.mkdtemp()
        sgy = os.path.join(root, "demo.sgy")
        self._write_even_sgy(sgy, n_gathers=6)
        m = JobManager(os.path.join(root, "jobs"), CFG)
        job = m.create_job(sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2, 3, 4, 5, 6],
                           clip=99, created_by="boss", decimate_n=1)
        self.assertEqual(len(job.gather_ids()), 6)

    def test_create_job_decimate_more_than_pool_keeps_first(self):
        root = tempfile.mkdtemp()
        sgy = os.path.join(root, "demo.sgy")
        self._write_even_sgy(sgy, n_gathers=6)
        m = JobManager(os.path.join(root, "jobs"), CFG)
        job = m.create_job(sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2, 3, 4, 5, 6],
                           clip=99, created_by="boss", decimate_n=50)
        self.assertEqual(len(job.gather_ids()), 1)          # gs[::50] 仍留第 1 个

    def test_min_traces_filtered_before_decimate(self):
        """顺序必须「先按道数下限过滤、再抽稀」：劣质道集先出局，抽稀只削好道集。"""
        root = tempfile.mkdtemp()
        sgy = os.path.join(root, "v.sgy")
        # 炮1=2道（劣质），炮2/3/4=8道
        TestCreateMinTraces._write_var_sgy(sgy, [2, 8, 8, 8])
        m = JobManager(os.path.join(root, "jobs"), CFG)
        job = m.create_job(sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2, 3, 4],
                           clip=99, created_by="boss", min_traces=5, decimate_n=2)
        ids = job.gather_ids()
        # 先滤掉炮1 → [炮2,炮3,炮4]，再每 2 个留 1 个 → [炮2,炮4]
        self.assertEqual(len(ids), 2)
        self.assertTrue(ids[0].endswith("_95-96_2"))
        self.assertTrue(ids[1].endswith("_95-96_4"))

    def test_decimate_all_removed_raises(self):
        """抽稀后一个不剩（理论上不可达）也要给明确报错而非空作业。"""
        from jobmanager import JobManager as _J
        # 空池抽稀仍为空，create_job 的「没有抽到任何道集」分支覆盖该情形
        self.assertEqual(_J._decimate([], 3), [])
        root = tempfile.mkdtemp()
        sgy = os.path.join(root, "v.sgy")
        TestCreateMinTraces._write_var_sgy(sgy, [2, 8])
        m = JobManager(os.path.join(root, "jobs"), CFG)
        with self.assertRaises(JobError):
            m.create_job(sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2],
                         clip=99, created_by="boss", min_traces=50, decimate_n=2)


class TestBandpassDisplay(unittest.TestCase):
    """显示图滤波：仅影响 Job.display/display_image，绝不碰导出用的 raw。"""

    FLT = {"f1": 20.0, "f2": 30.0, "f3": 80.0, "f4": 100.0}

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.manager = JobManager(self.root, CFG)
        self.out = os.path.join(self.root, "job1"); os.makedirs(self.out)
        self.job = self.manager.register_job(meta_for("job1", self.out),
                                             fake_gathers(), 200, fake_provider())
        self.gid = self.job.gather_ids()[0]

    def test_dt_ms_read_from_meta(self):
        self.assertAlmostEqual(self.job.dt_ms, 2.0)

    def test_unknown_dt_ms_refuses_to_filter(self):
        """dt_ms 缺失时宁可不给滤波，也不能按猜的采样间隔画错通带位置。"""
        m = meta_for("j2", self.out); m.pop("dt_ms")
        job = self.manager.register_job(m, fake_gathers(), 200, fake_provider())
        self.assertEqual(job.dt_ms, 0.0)                    # 0 = 未知
        self.assertEqual(job.display(job.gather_ids()[0]).shape, (40, 200))   # 干净图照常
        with self.assertRaises(JobError):
            job.display(job.gather_ids()[0], bandpass=self.FLT)

    def test_garbage_dt_ms_treated_as_unknown(self):
        m = meta_for("j3", self.out); m["dt_ms"] = "abc"
        job = self.manager.register_job(m, fake_gathers(), 200, fake_provider())
        self.assertEqual(job.dt_ms, 0.0)
        m2 = meta_for("j4", self.out); m2["dt_ms"] = -1.0
        job2 = self.manager.register_job(m2, fake_gathers(), 200, fake_provider())
        self.assertEqual(job2.dt_ms, 0.0)

    def test_filtered_display_differs_from_clean(self):
        clean = self.job.display(self.gid)
        filt = self.job.display(self.gid, bandpass=self.FLT)
        self.assertEqual(filt.shape, clean.shape)
        self.assertFalse(np.allclose(clean, filt))

    def test_filtered_display_does_not_mutate_raw(self):
        """滤波不得就地改写 provider 的数据（否则干净视图被污染）。"""
        before = self.job.display(self.gid)
        self.job.display(self.gid, bandpass=self.FLT)
        after = self.job.display(self.gid)
        np.testing.assert_allclose(before, after)

    def test_filtered_image_uses_separate_cache_file(self):
        clean = self.job.display_image(self.gid)
        filt = self.job.display_image(self.gid, bandpass=self.FLT)
        self.assertNotEqual(clean, filt)
        self.assertTrue(os.path.isfile(clean))
        self.assertTrue(os.path.isfile(filt))
        # 还原（再取干净图）必须仍命中干净缓存、不被滤波图顶掉
        self.assertEqual(self.job.display_image(self.gid), clean)

    def test_same_filter_reuses_cache(self):
        a = self.job.display_image(self.gid, bandpass=self.FLT)
        b = self.job.display_image(self.gid, bandpass=dict(self.FLT))
        self.assertEqual(a, b)

    def test_incomplete_filter_raises_instead_of_clean_fallback(self):
        """参数不全必须报错，绝不能悄悄给回未滤波的图（点了滤波却看到干净图）。"""
        self.job.display_image(self.gid)                  # 先建好干净图缓存
        with self.assertRaises(JobError):
            self.job.display_image(self.gid, bandpass={"f1": 10.0})
        with self.assertRaises(JobError):
            self.job.display(self.gid, bandpass={"f1": 10.0, "f2": 20.0})


class TestDtMsPersistedOnCreate(unittest.TestCase):
    def test_create_job_writes_dt_ms_to_job_json(self):
        root = tempfile.mkdtemp()
        sgy = os.path.join(root, "demo.sgy")
        write_sgy(sgy, n_gathers=2, traces_per=8, ns=64)   # dt_us=2000 → 2.0 ms
        jobs_root = os.path.join(root, "jobs")
        m = JobManager(jobs_root, CFG)
        job = m.create_job(sgy, [(95, 96), (13, 16)], [(95, 96)], [1, 2],
                           clip=99, created_by="boss")
        self.assertAlmostEqual(float(job.meta["dt_ms"]), 2.0)
        m2 = JobManager(jobs_root, CFG)
        m2.load_all()
        self.assertAlmostEqual(m2.get(job.job_id).dt_ms, 2.0)   # 恢复后仍可用于滤波

    def test_legacy_job_without_dt_ms_still_loads(self):
        """旧作业（job.json 无 dt_ms）必须照常恢复，并补出可用采样间隔。"""
        root = tempfile.mkdtemp()
        sgy = os.path.join(root, "demo.sgy")
        write_sgy(sgy, n_gathers=2, traces_per=8, ns=64)
        jobs_root = os.path.join(root, "jobs")
        out = os.path.join(jobs_root, "legacy__x"); os.makedirs(out)
        meta = meta_for("legacy__x", out)
        meta.update({"source_file": sgy, "values": [1, 2]})
        meta.pop("dt_ms")
        with open(os.path.join(out, "job.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        m = JobManager(jobs_root, CFG)
        m.load_all()
        job = m.get("legacy__x")
        self.assertEqual(job.state, "open")        # 不因缺 dt_ms 被标 broken
        self.assertAlmostEqual(job.dt_ms, 2.0)     # 从源文件 reader 补齐


class TestSharedColorScale(unittest.TestCase):
    """滤波前后必须共用同一色标：界限只从原始数据算一次，滤波图不被重新拉满。"""

    NS, DT_MS = 256, 2.0
    # 带外强能量（k=4 → 7.8 Hz，幅值 10）+ 带内弱信号（k=20 → 39.1 Hz，幅值 1）
    FLT = {"f1": 20.0, "f2": 30.0, "f3": 80.0, "f4": 100.0}

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.manager = JobManager(self.root, CFG)
        self.out = os.path.join(self.root, "job1"); os.makedirs(self.out)
        dt = self.DT_MS / 1000.0               # 秒：f = k / (NS * dt)
        t = np.arange(self.NS) * dt
        slow = 10.0 * np.sin(2 * np.pi * (4 / (self.NS * dt)) * t)    # 7.8 Hz，带外强能量
        fast = 1.0 * np.sin(2 * np.pi * (20 / (self.NS * dt)) * t)    # 39.1 Hz，带内弱信号
        self.data = np.tile((slow + fast), (6, 1)).astype(np.float32)
        meta = meta_for("job1", self.out)
        self.job = self.manager.register_job(
            meta, fake_gathers(), self.NS, lambda g, d=self.data: d)
        self.gid = self.job.gather_ids()[0]

    @staticmethod
    def _img_span(path):
        from PIL import Image
        a = np.asarray(Image.open(path).convert("L"), dtype=float)
        return float(a.max() - a.min())

    def test_vlim_comes_from_raw_data_only(self):
        v = self.job.display_vlim(self.gid)
        self.assertAlmostEqual(v, clip_bound(self.data, self.job.clip), places=4)

    def test_clean_view_reaches_bound_filtered_does_not_rescale(self):
        vlim = self.job.display_vlim(self.gid)
        clean = self.job.display(self.gid)
        filt = self.job.display(self.gid, bandpass=self.FLT)
        self.assertAlmostEqual(float(np.abs(clean).max()), vlim, places=4)  # 干净图用满色标
        # 滤波只剩弱得多的带内信号，绝不能又被拉伸到满量程
        self.assertLess(float(np.abs(filt).max()), 0.25 * vlim)

    def test_filtered_png_span_much_smaller_than_clean(self):
        """像素级验证：共用色标 → 滤波图对比度显著变低，而不是重新拉满。"""
        clean = self.job.display_image(self.gid)
        filt = self.job.display_image(self.gid, bandpass=self.FLT)
        self.assertLess(self._img_span(filt), 0.5 * self._img_span(clean))

    def test_repeat_filtered_render_is_deterministic(self):
        """同一色标下重复渲染结果一致（色标不随数据变化漂移）。"""
        p1 = self.job.display_image(self.gid, bandpass=self.FLT)
        with open(p1, "rb") as fh:
            first = fh.read()
        os.remove(p1)
        p2 = self.job.display_image(self.gid, bandpass=self.FLT)
        with open(p2, "rb") as fh:
            self.assertEqual(first, fh.read())


if __name__ == "__main__":
    unittest.main()
