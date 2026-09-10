# -*- coding: utf-8 -*-
import unittest

from web_core import (IMG_W, IMG_H, active_filter, box_target, fmt_option,
                      jobs_progress_html, label_of, parse_filter, pixel_box_to_data)


class _Opt:
    """最小 Option 替身：只需 hotkey / label 两个属性。"""

    def __init__(self, hotkey, label):
        self.hotkey = hotkey
        self.label = label


class TestOptionText(unittest.TestCase):
    def test_fmt_option_prefix(self):
        self.assertEqual(fmt_option(_Opt("1", "炮集")), "1 炮集")

    def test_label_of_strips_prefix(self):
        self.assertEqual(label_of("1 炮集"), "炮集")

    def test_label_of_no_prefix(self):
        self.assertEqual(label_of("炮集"), "炮集")


class TestPixelBox(unittest.TestCase):
    def test_pixel_box_to_data_range(self):
        box = pixel_box_to_data((0, 0), (IMG_W - 1, IMG_H - 1), 8, 200)
        self.assertEqual(box["traces"], [0, 8])
        self.assertEqual(box["samples"], [0, 200])
        self.assertEqual(box["xyxy"], [0, 0, IMG_W - 1, IMG_H - 1])

    def test_pixel_box_orders_corners(self):
        # 两个角点顺序颠倒时，应归一化为左上→右下
        box = pixel_box_to_data((100, 200), (10, 20), 512, 1024)
        self.assertEqual(box["xyxy"], [10, 20, 100, 200])

    def test_pixel_box_clamps_negative(self):
        box = pixel_box_to_data((-5, -5), (50, 50), 512, 1024)
        self.assertEqual(box["xyxy"], [0, 0, 50, 50])


class TestJobsProgressHtml(unittest.TestCase):
    def _jobs(self):
        return [
            {"job_id": "j1", "title": "A 作业", "state": "open",
             "labeled": 3, "total": 10, "created_by": "boss"},
            {"job_id": "j2", "title": "<inject>", "state": "closed",
             "labeled": 5, "total": 5, "created_by": "boss"},
        ]

    def test_shows_counts_and_percent(self):
        html = jobs_progress_html(self._jobs())
        self.assertIn("3 / 10", html)
        self.assertIn("30%", html)
        self.assertIn("j1", html)

    def test_full_progress_shows_100(self):
        html = jobs_progress_html(self._jobs())
        self.assertIn("5 / 5", html)
        self.assertIn("100%", html)

    def test_zero_total_does_not_crash(self):
        jobs = [{"job_id": "j3", "title": "空", "state": "broken",
                 "labeled": 0, "total": 0, "created_by": "boss"}]
        html = jobs_progress_html(jobs)
        self.assertIn("0 / 0", html)      # 防除零，仍能渲染
        self.assertNotIn("NaN", html)

    def test_escapes_title(self):
        html = jobs_progress_html(self._jobs())
        self.assertNotIn("<inject>", html)
        self.assertIn("&lt;inject&gt;", html)


class TestParseFilter(unittest.TestCase):
    """面波区「滤波」按钮的参数校验：0 <= f1 < f2 < f3 < f4。"""

    def test_valid_returns_params_and_no_error(self):
        p, err = parse_filter(10, 20, 100, 150)
        self.assertEqual(err, "")
        self.assertEqual(p, {"f1": 10.0, "f2": 20.0, "f3": 100.0, "f4": 150.0})

    def test_valid_zero_f1(self):
        p, err = parse_filter(0, 20, 100, 150)
        self.assertEqual(err, "")
        self.assertEqual(p["f1"], 0.0)

    def test_accepts_string_numbers(self):
        p, err = parse_filter("10", "20", "100", "150")
        self.assertEqual(err, "")
        self.assertEqual(p["f4"], 150.0)

    def test_unordered_rejected(self):
        p, err = parse_filter(50, 20, 100, 150)
        self.assertIsNone(p)
        self.assertTrue(err)

    def test_equal_corners_rejected(self):
        p, err = parse_filter(20, 20, 100, 150)
        self.assertIsNone(p)
        self.assertTrue(err)

    def test_negative_rejected(self):
        p, err = parse_filter(-5, 20, 100, 150)
        self.assertIsNone(p)
        self.assertTrue(err)

    def test_non_numeric_rejected(self):
        p, err = parse_filter("abc", 20, 100, 150)
        self.assertIsNone(p)
        self.assertTrue(err)

    def test_none_rejected(self):
        p, err = parse_filter(None, 20, 100, 150)
        self.assertIsNone(p)
        self.assertTrue(err)


class TestActiveFilter(unittest.TestCase):
    """滤波状态与 gid 绑定：换到别的道集时自动失效（无需各 handler 手动清除）。"""

    def test_no_filter_by_default(self):
        self.assertIsNone(active_filter({}, "g1"))

    def test_active_for_matching_gid(self):
        st = {"filter": {"gid": "g1", "f1": 10.0, "f2": 20.0, "f3": 100.0, "f4": 150.0}}
        self.assertEqual(active_filter(st, "g1")["f1"], 10.0)

    def test_inactive_after_gather_changes(self):
        st = {"filter": {"gid": "g1", "f1": 10.0, "f2": 20.0, "f3": 100.0, "f4": 150.0}}
        self.assertIsNone(active_filter(st, "g2"))

    def test_malformed_entry_ignored(self):
        self.assertIsNone(active_filter({"filter": "on"}, "g1"))
        self.assertIsNone(active_filter({"filter": {"f1": 10.0}}, "g1"))


class _Feat:
    """最小 Feature 替身：box_target 只用 name / key。"""

    def __init__(self, name, key):
        self.name, self.key = name, key


class TestBoxTarget(unittest.TestCase):
    """Ctrl+两点框选的目标特征：第一个「需框未框」的 bbox 特征。"""

    def setUp(self):
        self.feats = [_Feat("面波", "surface_wave"), _Feat("近炮点强能量噪声", "near_shot_noise")]

    def t(self, sel, boxes):
        return box_target(self.feats, sel, boxes)

    def test_no_bbox_features_returns_none(self):
        self.assertIsNone(box_target([], {}, {}))

    def test_first_needing_box_not_yet_drawn(self):
        sel = {"面波": "存在", "近炮点强能量噪声": "存在"}
        self.assertEqual(self.t(sel, {}), "surface_wave")

    def test_advances_after_first_box_drawn(self):
        sel = {"面波": "存在", "近炮点强能量噪声": "存在"}
        self.assertEqual(self.t(sel, {"surface_wave": {"xyxy": [1, 1, 2, 2]}}),
                         "near_shot_noise")

    def test_absent_is_skipped(self):
        """选「不存在」不需要框，Ctrl 不能卡在它上面。"""
        sel = {"面波": "不存在", "近炮点强能量噪声": "存在"}
        self.assertEqual(self.t(sel, {}), "near_shot_noise")

    def test_all_absent_returns_none(self):
        """两个都选「不存在」→ 不该再画出多余的框（曾经会返回最后一个，导致仍能拉框）。"""
        sel = {"面波": "不存在", "近炮点强能量噪声": "不存在"}
        self.assertIsNone(self.t(sel, {}))
        self.assertIsNone(self.t(sel, {"surface_wave": {"xyxy": [1, 1, 2, 2]}}))

    def test_absent_feature_never_becomes_target(self):
        """「不存在」的特征绝不能成为 Ctrl 目标 —— 用户实测：近炮点选「不存在」却还能拉框。

        面波=存在、近炮点=不存在时：无论面波画没画，目标都只能是面波（已画就是重画它），
        绝不会落到近炮点上。
        """
        sel = {"面波": "存在", "近炮点强能量噪声": "不存在"}
        self.assertEqual(self.t(sel, {"surface_wave": {"xyxy": [1, 1, 2, 2]}}),
                         "surface_wave")
        self.assertEqual(self.t(sel, {}), "surface_wave")

    def test_one_absent_other_needs_box(self):
        """面波=存在且已画框、近炮点=存在未画 → 目标仍是近炮点。"""
        sel = {"面波": "存在", "近炮点强能量噪声": "存在"}
        self.assertEqual(self.t(sel, {"surface_wave": {"xyxy": [1, 1, 2, 2]}}),
                         "near_shot_noise")

    def test_unanswered_can_be_drawn_first(self):
        self.assertEqual(self.t({}, {}), "surface_wave")

    def test_unanswered_skipped_when_box_already_drawn(self):
        self.assertEqual(self.t({}, {"surface_wave": {"xyxy": [1, 1, 2, 2]}}),
                         "near_shot_noise")

    def test_all_drawn_returns_last_for_redraw(self):
        sel = {"面波": "存在", "近炮点强能量噪声": "存在"}
        boxes = {"surface_wave": {"xyxy": [1, 1, 2, 2]},
                 "near_shot_noise": {"xyxy": [3, 3, 4, 4]}}
        self.assertEqual(self.t(sel, boxes), "near_shot_noise")

    def test_absent_with_box_is_skipped(self):
        """选了「不存在」但残留一个框：仍跳过它，不抢占目标。"""
        sel = {"面波": "不存在", "近炮点强能量噪声": "存在"}
        self.assertEqual(self.t(sel, {"surface_wave": {"xyxy": [1, 1, 2, 2]}}),
                         "near_shot_noise")

    def test_none_boxes_tolerated(self):
        self.assertEqual(box_target(self.feats, None, None), "surface_wave")


if __name__ == "__main__":
    unittest.main()
