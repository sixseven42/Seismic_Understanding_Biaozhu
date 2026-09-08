# -*- coding: utf-8 -*-
import os, tempfile, unittest
from cloudsync import Uploader, object_key, load_config

class TestCloudSync(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.img = os.path.join(self.d, "images"); os.makedirs(self.img)
        for n in ("a.png", "b.png"):
            with open(os.path.join(self.img, n), "wb") as f: f.write(b"x")
        with open(os.path.join(self.d, "labels.jsonl"), "w") as f: f.write("{}\n")

    def test_object_key_slashes(self):
        self.assertEqual(object_key("job1", "images\\a.png"), "seismic/job1/images/a.png")

    def test_dry_run_records_not_upload(self):
        up = Uploader({}, dry_run=True)                 # enabled false
        up.enqueue("job1", self.d, ["images/a.png", "labels.jsonl"])
        self.assertEqual(len(up.dry_log), 2)
        self.assertEqual(up.dry_log[0][0], "seismic/job1/images/a.png")

    def test_missing_local_skipped(self):
        up = Uploader({}, dry_run=True)
        up.enqueue("job1", self.d, ["images/nope.png"])
        self.assertEqual(up.dry_log, [])

    def test_resync_all_walks_images_and_labels(self):
        up = Uploader({}, dry_run=True)
        up.resync_all("job1", self.d)
        keys = [k for k, _ in up.dry_log]
        self.assertEqual(sorted(keys),
                         sorted(["seismic/job1/images/a.png", "seismic/job1/images/b.png",
                                 "seismic/job1/labels.jsonl"]))

class TestRetryAndLog(unittest.TestCase):
    def test_final_failure_writes_upload_log(self):
        import unittest.mock as mock
        self.d = tempfile.mkdtemp()
        img = os.path.join(self.d, "images"); os.makedirs(img)
        with open(os.path.join(self.d, "labels.jsonl"), "w") as f: f.write("{}\n")
        class _FakeFail:
            def put_object(self, **kw): raise OSError("boom")
        cfg = {"enabled": True, "secret_id": "s", "secret_key": "k",
               "bucket": "b", "region": "r"}
        up = Uploader(cfg, dry_run=False)     # enabled True (secret_id set)
        up._client = _FakeFail()
        up.max_attempts = 2
        with mock.patch("cloudsync.time.sleep"):
            up.enqueue("job1", self.d, ["labels.jsonl"])
            item = up._q.get_nowait()
            up._upload_once(item)
        self.assertEqual(len(up.failures), 1)
        self.assertTrue(os.path.isfile(os.path.join(self.d, "upload.log")))
        with open(os.path.join(self.d, "upload.log"), encoding="utf-8") as f:
            self.assertIn("seismic/job1/labels.jsonl", f.read())

if __name__ == "__main__":
    unittest.main()
