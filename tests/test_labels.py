# -*- coding: utf-8 -*-
"""标签配置回归：v3.7 删除「静校正 / 直达波 / 50Hz工业噪声」三个特征。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from labels import LabelConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = LabelConfig(os.path.join(ROOT, "label_config.yaml"))


class TestRemovedFeatures(unittest.TestCase):
    """三个已删特征不得再出现在配置里（键与显示名都不许）。"""

    REMOVED_KEYS = ("statics", "direct_wave", "industrial_50hz")
    REMOVED_NAMES = ("静校正", "直达波", "50Hz工业噪声")

    def test_removed_keys_absent(self):
        keys = {f.key for f in CFG.features}
        for k in self.REMOVED_KEYS:
            self.assertNotIn(k, keys)

    def test_removed_names_absent(self):
        names = {f.name for f in CFG.features}
        for n in self.REMOVED_NAMES:
            self.assertNotIn(n, names)

    def test_remaining_features_order(self):
        # v3.8：带框特征排最后（键盘先答完选择题，再鼠标 Ctrl 框选）
        self.assertEqual([f.key for f in CFG.features],
                         ["gather_type", "abnormal_amplitude", "aliasing_noise",
                          "surface_wave", "near_shot_noise"])

    def test_bbox_features_are_last(self):
        """带框特征必须连续排在末尾 —— 界面/键盘顺序即标签顺序，先选择题后框选。"""
        flags = [f.bbox for f in CFG.features]
        first_box = flags.index(True)
        self.assertTrue(all(flags[first_box:]), flags)
        self.assertTrue(all(not f for f in flags[:first_box]), flags)


class TestSentenceTemplate(unittest.TestCase):
    def test_template_drops_removed_placeholders(self):
        s = CFG.sentence_template
        for n in TestRemovedFeatures.REMOVED_NAMES:
            self.assertNotIn("{" + n + "}", s)

    def test_render_sentence_has_no_removed_wording(self):
        sel = {f.name: f.options[0].label for f in CFG.features}
        s = CFG.render_sentence(sel)
        self.assertIn("这是一条炮集", s)
        for n in TestRemovedFeatures.REMOVED_NAMES:
            self.assertNotIn(n, s)
        self.assertNotIn("（未标注）", s)

    def test_render_sentence_uses_phrases(self):
        # 面波取「已压制但有残留」时，句子应原样带出该措辞
        sel = {f.name: f.options[0].label for f in CFG.features}
        sel["面波"] = "已压制但有残留"
        self.assertIn("面波已压制但有残留", CFG.render_sentence(sel))


class TestBBoxFeatures(unittest.TestCase):
    def test_surface_wave_still_bbox(self):
        sw = next(f for f in CFG.features if f.key == "surface_wave")
        self.assertTrue(sw.bbox)
        self.assertTrue(sw.bbox_color)

    def test_bbox_features_are_surface_wave_and_near_shot(self):
        self.assertEqual([f.key for f in CFG.features if f.bbox],
                         ["surface_wave", "near_shot_noise"])

    def test_validate_selection_ignores_legacy_keys(self):
        """旧记录里残留的 statics/direct_wave/industrial_50hz 不参与校验（向后兼容）。"""
        sel = {f.name: f.options[0].label for f in CFG.features}
        legacy = dict(sel, **{"静校正": "需要", "直达波": "存在", "50Hz工业噪声": "不存在"})
        self.assertEqual(CFG.validate_selection(legacy), [])

    def test_to_record_labels_only_current_keys(self):
        sel = {f.name: f.options[0].label for f in CFG.features}
        rec = CFG.to_record_labels(sel)
        self.assertEqual(set(rec), {f.key for f in CFG.features})
        for k in TestRemovedFeatures.REMOVED_KEYS:
            self.assertNotIn(k, rec)


if __name__ == "__main__":
    unittest.main()
