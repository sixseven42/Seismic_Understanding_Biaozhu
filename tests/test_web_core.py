# -*- coding: utf-8 -*-
import unittest

from web_core import IMG_W, IMG_H, fmt_option, label_of, pixel_box_to_data


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


if __name__ == "__main__":
    unittest.main()
