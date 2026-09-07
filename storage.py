# -*- coding: utf-8 -*-
"""
storage.py — 标注结果存储（JSONL）与断点续标

每行一条记录：
{
  "gather_id": "95-96_1373",
  "gather_key": "95-96",
  "gather_value": 1373,
  "n_traces": 528,
  "labels": {"gather_type": "炮集", "statics": "需要", ...},
  "sentence": "这是一条炮集，……。",
  "image_path": "images/shot_1373.png",
  "source_file": "/path/to/xxx.sgy",
  "timestamp": "2026-08-21T00:40:00"
}
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime


class LabelStore:
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.records: dict[str, dict] = {}   # gather_id -> record
        self.order: list[str] = []           # 保持写入顺序
        self._load()

    def _load(self):
        if not os.path.isfile(self.jsonl_path):
            return
        with open(self.jsonl_path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f"{self.jsonl_path} 第 {ln} 行不是合法 JSON")
                gid = rec.get("gather_id")
                if gid is None:
                    raise ValueError(f"{self.jsonl_path} 第 {ln} 行缺少 gather_id")
                if gid not in self.records:
                    self.order.append(gid)
                self.records[gid] = rec

    # ------------------------------------------------------------------
    def upsert(self, record: dict) -> dict:
        """新增或覆盖一条记录，并落盘（全量重写，保证修改生效）。"""
        record = dict(record)
        record["timestamp"] = datetime.now().isoformat(timespec="seconds")
        gid = record["gather_id"]
        if gid not in self.records:
            self.order.append(gid)
        self.records[gid] = record
        self._flush()
        return record

    def get(self, gather_id: str) -> dict | None:
        return self.records.get(gather_id)

    def is_labeled(self, gather_id: str) -> bool:
        return gather_id in self.records

    def remove_augmented(self, base_gather_id: str) -> list[str]:
        """删除某道集的所有增强记录（augmented_from == base_gather_id）。
        返回被删的 gather_id 列表。"""
        removed = [gid for gid, rec in self.records.items()
                   if rec.get("augmented_from") == base_gather_id]
        if removed:
            for gid in removed:
                del self.records[gid]
            self.order = [g for g in self.order if g not in set(removed)]
            self._flush()
        return removed

    def labeled_ids(self) -> set[str]:
        return set(self.records)

    def next_unlabeled(self, gather_ids: list[str], start: int = 0) -> int | None:
        """在 gather_ids 序列中，从 start 起找第一个未标注的位置。"""
        for i in range(start, len(gather_ids)):
            if gather_ids[i] not in self.records:
                return i
        return None

    # ------------------------------------------------------------------
    def _flush(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.jsonl_path)), exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(os.path.abspath(self.jsonl_path)), suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for gid in self.order:
                f.write(json.dumps(self.records[gid], ensure_ascii=False) + "\n")
        os.replace(tmp, self.jsonl_path)   # 原子替换，防中途崩溃丢数据
