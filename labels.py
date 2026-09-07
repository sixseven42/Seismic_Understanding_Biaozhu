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


class Feature:
    def __init__(self, d: dict):
        self.name: str = d["name"]
        self.key: str = d.get("key", d["name"])
        self.bbox: bool = bool(d.get("bbox", False))
        self.bbox_color: str = d.get("bbox_color", "#ff0000")
        opts = d.get("options") or []
        if not opts:
            raise LabelConfigError(f"特征「{self.name}」没有可选项")
        self.options = [Option(o, i + 1) for i, o in enumerate(opts)]

    def option_by_label(self, label: str) -> Option | None:
        for o in self.options:
            if o.label == label:
                return o
        return None


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
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise LabelConfigError("存在重名特征")

    def validate_selection(self, selection: dict[str, str]) -> list[str]:
        """selection: {特征名: 选项label}。返回缺失/非法项说明，空列表表示合法。"""
        errs = []
        for f in self.features:
            v = selection.get(f.name)
            if v is None:
                errs.append(f"「{f.name}」未选择")
            elif f.option_by_label(v) is None:
                errs.append(f"「{f.name}」的选项「{v}」不在配置中")
        return errs

    def render_sentence(self, selection: dict[str, str]) -> str:
        """把选择渲染成句子（用各选项的 phrase）。"""
        mapping = {}
        for f in self.features:
            opt = f.option_by_label(selection.get(f.name, ""))
            mapping[f.name] = opt.phrase if opt else "（未标注）"
        try:
            return self.sentence_template.format(**mapping)
        except KeyError as e:
            raise LabelConfigError(
                f"句子模板引用了不存在的特征: {e}（现有: {list(mapping)}）"
            )

    def to_record_labels(self, selection: dict[str, str]) -> dict[str, str]:
        """转为 JSONL 存储字段：{feature_key: option_label}。"""
        return {f.key: selection.get(f.name, "") for f in self.features}
