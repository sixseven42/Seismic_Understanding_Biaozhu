# -*- coding: utf-8 -*-
"""
migrate_remove_features.py — v1.9 标签体系迁移：删除「线性噪声」「多次波」

按当前 label_config.yaml 重建 output/labels.jsonl 每条记录：
- labels 只保留配置中仍存在的特征键（按配置顺序），多余键（linear_noise、multiples）删除
- sentence 用 labels.render_sentence 重新生成
- 幂等：重复运行结果不变
- 首次运行前把原文件备份为 labels.jsonl.v18.bak（已存在则跳过）

用法：
    python migrate_remove_features.py [jsonl路径]   # 默认 output/labels.jsonl
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

from labels import LabelConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSONL = os.path.join(HERE, "output", "labels.jsonl")
BACKUP_SUFFIX = ".v18.bak"


def migrate(jsonl_path: str, cfg: LabelConfig) -> None:
    with open(jsonl_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    key2feat = {feat.key: feat for feat in cfg.features}
    n_changed, n_dropped, missing = 0, 0, {}

    for rec in records:
        old_labels = rec.get("labels", {})
        dropped = sorted(set(old_labels) - set(key2feat))
        if dropped:
            n_dropped += 1
        new_labels = {}
        selection = {}
        for feat in cfg.features:
            v = old_labels.get(feat.key)
            if v is None:
                missing.setdefault(rec["gather_id"], []).append(feat.key)
                v = ""
            new_labels[feat.key] = v
            selection[feat.name] = v
        new_sentence = cfg.render_sentence(selection)
        if new_labels != old_labels or new_sentence != rec.get("sentence"):
            n_changed += 1
            rec["labels"] = new_labels
            rec["sentence"] = new_sentence

    if missing:
        for gid, keys in list(missing.items())[:10]:
            print(f"⚠️ {gid} 缺少配置键: {keys}（已置空）")
        if len(missing) > 10:
            print(f"⚠️ …共 {len(missing)} 条记录缺键")

    backup = jsonl_path + BACKUP_SUFFIX
    if not os.path.exists(backup):
        shutil.copy2(jsonl_path, backup)
        print(f"已备份原文件 → {backup}")

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(jsonl_path)),
                               suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, jsonl_path)

    print(f"共 {len(records)} 条记录：{n_changed} 条有改动，"
          f"{n_dropped} 条删除了旧特征键（linear_noise/multiples）")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSONL
    cfg = LabelConfig(os.path.join(HERE, "label_config.yaml"))
    print(f"当前配置特征（{len(cfg.features)} 个）: "
          + ", ".join(f.name for f in cfg.features))
    migrate(path, cfg)
