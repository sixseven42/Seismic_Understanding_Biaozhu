# -*- coding: utf-8 -*-
"""
labels.py — 标签体系加载与句子渲染

标签体系由 label_config.yaml 定义，增删特征/选项只改配置文件。
"""

from __future__ import annotations

import os
import yaml


class LabelConfigError(Exception):
    pass


class Option:
    def __init__(self, d: dict, hotkey_default: int):
        self.label: str = d["label"]
        self.phrase: str = d.get("phrase", self.label)
        self.hotkey: str = str(d.get("hotkey", hotkey_default))
        self.exclusive: bool = bool(d.get("exclusive", False))


class Feature:
    def __init__(self, d: dict):
        self.name: str = d["name"]
        self.key: str = d.get("key", d["name"])
        self.bbox: bool = bool(d.get("bbox", False))
        self.bbox_color: str = d.get("bbox_color", "#ff0000")
        self.region_shape: str = d.get("region_shape", "rectangle")
        self.input: str = d.get("input", "radio")
        self.when: dict = dict(d.get("when") or {})
        opts = d.get("options") or []
        if not opts:
            raise LabelConfigError(f"特征「{self.name}」没有可选项")
        self.options = [Option(o, i + 1) for i, o in enumerate(opts)]

    def option_by_label(self, label: str) -> Option | None:
        for o in self.options:
            if o.label == label:
                return o
        return None

    def active_for(self, selection: dict) -> bool:
        """Whether this conditional feature applies to the current selection."""
        if not self.when:
            return True
        parent = selection.get(self.when.get("feature"))
        if isinstance(parent, list):
            parent = parent[0] if len(parent) == 1 else None
        if "equals" in self.when:
            return parent == self.when["equals"]
        if "not_equals" in self.when:
            return parent != self.when["not_equals"]
        return True


class LabelConfig:
    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise LabelConfigError(f"标签配置不存在: {path}")
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.features = [Feature(d) for d in cfg.get("features", [])]
        if not self.features:
            raise LabelConfigError("配置中没有任何特征")
        self.sentence_template: str = cfg.get("sentence_template", "")
        self.sentence_template_residual: str = cfg.get("sentence_template_residual", "")
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise LabelConfigError("存在重名特征")

    def validate_selection(self, selection: dict[str, str]) -> list[str]:
        """selection: {特征名: 选项label}。返回缺失/非法项说明，空列表表示合法。"""
        errs = []
        for f in self.active_features(selection):
            v = selection.get(f.name)
            if v is None:
                errs.append(f"「{f.name}」未选择")
            elif f.input == "checkbox":
                vals = v if isinstance(v, list) else [v]
                if not vals:
                    errs.append(f"「{f.name}」至少选择一项")
                    continue
                bad = [x for x in vals if f.option_by_label(x) is None]
                if bad:
                    errs.append(f"「{f.name}」的选项「{bad[0]}」不在配置中")
                exclusive = [o.label for o in f.options if o.exclusive]
                if any(x in exclusive for x in vals) and len(vals) != 1:
                    errs.append(f"「{f.name}」的独占选项不能与其他选项同时选择")
            elif f.option_by_label(v) is None:
                errs.append(f"「{f.name}」的选项「{v}」不在配置中")
        return errs

    def active_features(self, selection: dict) -> list[Feature]:
        return [f for f in self.features if f.active_for(selection or {})]

    def render_sentence(self, selection: dict[str, str]) -> str:
        """把选择渲染成句子（用各选项的 phrase）。"""
        mapping = {}
        for f in self.features:
            value = selection.get(f.name, "")
            if f.input == "checkbox":
                vals = value if isinstance(value, list) else ([value] if value else [])
                mapping[f.name] = "、".join(
                    (f.option_by_label(v).phrase if f.option_by_label(v) else str(v))
                    for v in vals
                ) or "（未标注）"
            else:
                opt = f.option_by_label(value)
                mapping[f.name] = opt.phrase if opt else "（未标注）"
        template = self.sentence_template
        if selection.get("集合类型") == "残差" and self.sentence_template_residual:
            template = self.sentence_template_residual
        try:
            return template.format(**mapping)
        except KeyError as e:
            raise LabelConfigError(
                f"句子模板引用了不存在的特征: {e}（现有: {list(mapping)}）"
            )

    def to_record_labels(self, selection: dict[str, str]) -> dict[str, str]:
        """转为 JSONL 存储字段：{feature_key: option_label}。"""
        return {f.key: selection.get(f.name, "") for f in self.active_features(selection)}
