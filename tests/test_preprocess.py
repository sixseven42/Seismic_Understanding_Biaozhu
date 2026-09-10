# -*- coding: utf-8 -*-
"""preprocess.bandpass 步骤：复用 bandpass.py 的 SeiSee 四角频率零相位带通。"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import apply_pipeline, available, clip_abs, clip_bound


def _tone(k, ns, dt):
    """在 FFT 长度 ns 上恰为 k 个整周期的正弦（无频谱泄漏）。"""
    t = np.arange(ns) * dt
    return np.sin(2 * np.pi * (k / (ns * dt)) * t)


class TestBandpassRegistered(unittest.TestCase):
    def test_registered(self):
        self.assertIn("bandpass", available())

    def test_declares_params(self):
        p = available()["bandpass"]["params"]
        for k in ("f1", "f2", "f3", "f4", "dt_ms"):
            self.assertIn(k, p)


class TestBandpassCorners(unittest.TestCase):
    """四角频率必须满足 0 <= f1 < f2 < f3 < f4，否则报错而非静默返回。"""

    def setUp(self):
        self.d = np.zeros((2, 256), np.float32)

    def _run(self, **kw):
        return apply_pipeline(self.d, [{"name": "bandpass", "params": kw}])

    def test_equal_corners_rejected(self):
        with self.assertRaises(ValueError):
            self._run(f1=20, f2=20, f3=80, f4=100, dt_ms=2.0)

    def test_unordered_corners_rejected(self):
        with self.assertRaises(ValueError):
            self._run(f1=80, f2=20, f3=100, f4=150, dt_ms=2.0)

    def test_negative_f1_rejected(self):
        with self.assertRaises(ValueError):
            self._run(f1=-1, f2=20, f3=80, f4=100, dt_ms=2.0)

    def test_bad_dt_rejected(self):
        with self.assertRaises(ValueError):
            self._run(f1=20, f2=30, f3=80, f4=100, dt_ms=0)


class TestBandpassFrequencyResponse(unittest.TestCase):
    """带内保留、带外压制（ns=256、dt=2ms → 频率分辨率 1.953 Hz）。"""

    NS, DT = 256, 0.002
    PARAMS = {"f1": 20.0, "f2": 30.0, "f3": 80.0, "f4": 100.0, "dt_ms": 2.0}

    def _filtered(self, k):
        sig = _tone(k, self.NS, self.DT)[None, :].astype(np.float32)
        return apply_pipeline(sig, [{"name": "bandpass", "params": self.PARAMS}])[0]

    def test_shape_and_dtype_preserved(self):
        data = np.random.default_rng(0).normal(size=(3, self.NS)).astype(np.float32)
        out = apply_pipeline(data, [{"name": "bandpass", "params": self.PARAMS}])
        self.assertEqual(out.shape, data.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_passband_tone_kept(self):
        # k=20 → 39.06 Hz，落在 [f2, f3) 通带内，幅值应基本不变
        out = self._filtered(20)
        self.assertAlmostEqual(float(np.std(out)), float(np.std(_tone(20, self.NS, self.DT))),
                               delta=0.02)

    def test_below_f1_tone_removed(self):
        # k=5 → 9.77 Hz < f1=20，应被完全压掉
        self.assertLess(float(np.std(self._filtered(5))), 1e-4)

    def test_above_f4_tone_removed(self):
        # k=102 → 199.2 Hz > f4=100，应被完全压掉
        self.assertLess(float(np.std(self._filtered(102))), 1e-4)

    def test_in_band_kept_while_out_of_band_removed(self):
        """同一道集里混三个频率：只有通带那个活下来。"""
        sig = (_tone(5, self.NS, self.DT) + _tone(20, self.NS, self.DT)
               + _tone(102, self.NS, self.DT))[None, :].astype(np.float32)
        out = apply_pipeline(sig, [{"name": "bandpass", "params": self.PARAMS}])[0]
        expect = _tone(20, self.NS, self.DT)
        np.testing.assert_allclose(out, expect, atol=0.02)

    def test_does_not_mutate_input(self):
        data = np.random.default_rng(1).normal(size=(2, self.NS)).astype(np.float32)
        orig = data.copy()
        apply_pipeline(data, [{"name": "bandpass", "params": self.PARAMS}])
        np.testing.assert_array_equal(data, orig)


class TestClipBoundAndAbs(unittest.TestCase):
    """滤波前后共用同一色标：clip_bound 只从原始数据算一次，clip_abs 按绝对界限截断。"""

    def test_clip_abs_clamps_to_absolute_bound(self):
        d = np.array([[-5.0, -0.5, 0.0, 0.5, 5.0]], dtype=np.float32)
        out = clip_abs(d, vlim=1.0)
        np.testing.assert_allclose(out, [[-1.0, -0.5, 0.0, 0.5, 1.0]])
        self.assertEqual(out.dtype, np.float32)

    def test_clip_abs_registered(self):
        self.assertIn("clip_abs", available())

    def test_clip_abs_zero_vlim_is_noop(self):
        d = np.array([[3.0, -7.0]], dtype=np.float32)
        np.testing.assert_allclose(clip_abs(d, vlim=0), d)

    def test_clip_bound_matches_clip_percentile(self):
        from preprocess import clip_percentile
        rng = np.random.default_rng(0)
        d = rng.normal(size=(4, 128)).astype(np.float32) * 3
        v = clip_bound(d, 99.0)
        np.testing.assert_allclose(clip_abs(d, v), clip_percentile(d, 99.0))

    def test_clip_bound_full_percentile_is_max(self):
        d = np.array([[-4.0, 2.0]], dtype=np.float32)
        self.assertEqual(clip_bound(d, 100.0), 4.0)

    def test_clip_bound_all_zero_is_one(self):
        self.assertEqual(clip_bound(np.zeros((2, 8), np.float32), 99.0), 1.0)

    def test_clip_bound_rejects_bad_percentile(self):
        with self.assertRaises(ValueError):
            clip_bound(np.zeros((1, 4), np.float32), 0)


if __name__ == "__main__":
    unittest.main()
