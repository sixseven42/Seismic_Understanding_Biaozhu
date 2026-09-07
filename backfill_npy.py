# -*- coding: utf-8 -*-
"""
backfill_npy.py — 为已标注的 labels.jsonl 回补原始数据 npy

逐条记录：按 source_file + sort_keys + gather_key/value 重新抽道，
把原始数据（float32, (n_traces, ns)，无预处理）存到 <输出目录>/npy/<gather_id>.npy，
并回填 npy_path / sort_keys / extract_key 字段。

已确认的排序键映射（用户拍板 2026-08-21）：
  95-96   -> 95-96, 13-16
  197-200 -> 197-200, 25-28

用法：python backfill_npy.py output/labels.jsonl
幂等：npy 已存在且字段已回填的记录自动跳过。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

from segy_reader import SegyReader
from gather import extract_gathers, parse_key, key_str

# gather_key -> sort_keys（记录里没有 sort_keys 字段时按此映射）
SORT_KEYS_MAP = {
    "95-96": [(95, 96), (13, 16)],
    "197-200": [(197, 200), (25, 28)],
}


def parse_sort_keys(s: str):
    return [parse_key(t) for t in s.split(",") if t.strip()]


def main(jsonl_path: str):
    out_dir = os.path.dirname(os.path.abspath(jsonl_path))
    npy_dir = os.path.join(out_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)

    with open(jsonl_path, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    print(f"共 {len(recs)} 条记录")

    readers: dict[str, SegyReader] = {}
    ok = skip = fail = 0
    try:
        for i, rec in enumerate(recs):
            gid = rec["gather_id"]
            npy_path = os.path.join(npy_dir, f"{gid}.npy")
            rel_npy = os.path.join("npy", f"{gid}.npy")
            if os.path.isfile(npy_path) and rec.get("npy_path") == rel_npy:
                skip += 1
                continue
            try:
                src = rec["source_file"]
                gkey = rec["gather_key"]
                sort_keys = (parse_sort_keys(rec["sort_keys"])
                             if rec.get("sort_keys") else SORT_KEYS_MAP[gkey])
                if src not in readers:
                    readers[src] = SegyReader(src, endian="auto")
                reader = readers[src]
                gathers = extract_gathers(reader, sort_keys, parse_key(gkey),
                                          values=[int(rec["gather_value"])])
                if len(gathers) != 1:
                    raise RuntimeError(f"抽到 {len(gathers)} 个道集")
                g = gathers[0]
                # 一致性校验：道数与 gather_id 必须吻合
                stem = os.path.splitext(os.path.basename(src))[0]
                expect_gid = f"{stem}__{g.key}_{g.value}"
                if expect_gid != gid:
                    raise RuntimeError(f"gather_id 不符: 期望 {expect_gid} 实际 {gid}")
                if g.n_traces != rec["n_traces"]:
                    raise RuntimeError(f"道数不符: 记录 {rec['n_traces']} 抽出 {g.n_traces}")
                raw = reader.get_traces(g.trace_indices).astype(np.float32)
                np.save(npy_path, raw)
                rec["npy_path"] = rel_npy
                rec["sort_keys"] = ", ".join(key_str(k) for k in sort_keys)
                rec["extract_key"] = gkey
                ok += 1
                if (ok + 1) % 20 == 0 or i == len(recs) - 1:
                    print(f"  进度 {i+1}/{len(recs)} ...")
            except Exception as e:
                fail += 1
                print(f"  [失败] {gid}: {e}")
    finally:
        for r in readers.values():
            r.close()

    # 原子重写 jsonl（保留原字段顺序与时间戳）
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, jsonl_path)
    print(f"完成: 成功 {ok}，跳过 {skip}，失败 {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python backfill_npy.py <labels.jsonl 路径>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
