# -*- coding: utf-8 -*-
import json
import os
import tempfile
import unittest

import numpy as np

from jobmanager import JobError, JobManager
from labels import LabelConfig
from preprocess import clip_bound
from segy_reader import FILE_HEADER_SIZE, HEADER_SIZE
from tests.segy_factory import write_sgy


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = LabelConfig(os.path.join(ROOT, "label_config.yaml"))


class TestResidualJobs(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.paths = [os.path.join(self.root, name)
                      for name in ("before.sgy", "after.sgy", "residual.sgy")]
        for path in self.paths:
            write_sgy(path, n_gathers=2, traces_per=4, ns=16)
        # Keep headers identical but make each source's samples distinguishable.
        for i, path in enumerate(self.paths):
            with open(path, "r+b") as fh:
                fh.seek(FILE_HEADER_SIZE + HEADER_SIZE)
                fh.write(np.array([10.0 + i], dtype="<f4").tobytes())
        self.jobs_root = os.path.join(self.root, "jobs")

    def create(self, manager=None):
        manager = manager or JobManager(self.jobs_root, CFG, aug_async=False)
        job = manager.create_job(
            self.paths[0], [(95, 96), (13, 16)], [(95, 96)], [1, 2],
            job_type="residual", source_files=self.paths, min_traces=0)
        return manager, job

    @staticmethod
    def selection():
        return {
            "集合类型": "残差",
            "噪声类型": ["异常振幅"],
            "去噪质量": ["去噪完成"],
        }

    def test_three_aligned_sources_are_available_as_layers(self):
        _, job = self.create()
        gid = next(g for g in job.gather_ids() if g.endswith("_1"))
        self.assertEqual(job.job_type, "residual")
        self.assertEqual(job.layer_names, ("去噪前", "去噪后", "噪声残差"))
        self.assertEqual([job.raw(gid, i)[0, 0] for i in range(3)], [10.0, 11.0, 12.0])
        with open(os.path.join(job.output_dir, "job.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        self.assertEqual(meta["job_type"], "residual")
        self.assertEqual(meta["source_files"], [os.path.abspath(p) for p in self.paths])

    def test_trace_count_mismatch_is_rejected(self):
        write_sgy(self.paths[2], n_gathers=2, traces_per=3, ns=16)
        with self.assertRaisesRegex(JobError, "道数不一致"):
            self.create()

    def test_any_trace_header_mismatch_is_rejected(self):
        with open(self.paths[1], "r+b") as fh:
            fh.seek(FILE_HEADER_SIZE + 30)
            fh.write(b"\x01")
        with self.assertRaisesRegex(JobError, "道头不一致"):
            self.create()

    def test_reload_preserves_three_source_provider(self):
        manager, job = self.create()
        job_id = job.job_id
        manager.aug_queue.stop()
        restored = JobManager(self.jobs_root, CFG, aug_async=False)
        restored.load_all()
        loaded = restored.get(job_id)
        gid = next(g for g in loaded.gather_ids() if g.endswith("_1"))
        self.assertEqual([loaded.raw(gid, i)[0, 0] for i in range(3)], [10.0, 11.0, 12.0])

    def test_all_residual_layers_use_before_display_scale(self):
        _, job = self.create()
        gid = next(g for g in job.gather_ids() if g.endswith("_1"))
        expected = job.display_vlim(gid, 0)
        self.assertEqual(job.display_vlim(gid, 1), expected)
        self.assertEqual(job.display_vlim(gid, 2), expected)

    def test_save_exports_fifteen_layer_images_and_three_npys(self):
        manager, job = self.create()
        claim = manager.claim(job.job_id, "ann1")
        gid = claim["gid"]
        render_calls = []
        original_render = job._render_png

        def capture_render(data, abs_path, vlim=None):
            render_calls.append((os.path.basename(abs_path), vlim))
            return original_render(data, abs_path, vlim=vlim)

        job._render_png = capture_render
        ok, rec, layers, err = manager.save(
            job.job_id, "ann1", gid, self.selection(), {}, aug_sync=True)
        self.assertTrue(ok, err)
        self.assertEqual(len(layers), 15)
        self.assertEqual(set(rec["image_paths"]), {"before", "after", "residual"})
        self.assertTrue(all(len(paths) == 5 for paths in rec["image_paths"].values()))
        self.assertEqual(len({path for paths in rec["image_paths"].values() for path in paths}), 15)
        self.assertEqual(set(rec["npy_paths"]), {"before", "after", "residual"})
        self.assertEqual(len({layer["npy_path"] for layer in layers}), 3)
        self.assertEqual(len(render_calls), 15)
        before = job.raw(gid, 0)
        for clip in {layer["clip_percentile"] for layer in layers}:
            expected_vlim = clip_bound(before, clip)
            clip_calls = [vlim for name, vlim in render_calls
                          if f"__clip{clip:g}.png" in name]
            self.assertEqual(len(clip_calls), 3)
            self.assertEqual(clip_calls, [expected_vlim] * 3)
        for layer_index in range(3):
            layer_records = [layer for layer in layers if layer["layer_index"] == layer_index]
            self.assertEqual(len(layer_records), 5)
            self.assertEqual({layer["clip_percentile"] for layer in layer_records},
                             {layer["clip_percentile"] for layer in layers if layer["layer_index"] == 0})
            for layer in layer_records:
                self.assertTrue(os.path.isfile(os.path.join(job.output_dir, layer["image_path"])))
                self.assertTrue(os.path.isfile(os.path.join(job.output_dir, layer["npy_path"])))
        self.assertEqual(
            rec["sentence"],
            "这是一个残差，去噪类型是异常振幅，去噪状况为：去噪完成。",
        )


if __name__ == "__main__":
    unittest.main()
