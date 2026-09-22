# -*- coding: utf-8 -*-
"""
check_field_constant.py — 检查「某个道头字节字段在每个道集内是否完全一致」

用途：判断某字节段能否作为该道集的定义字段 / 稳定标签。
判据：按你给的抽道集键把道分好组后，统计该字段在每个道集内的不同值个数：
      - 所有道集都只有 1 个不同值  → 该字段在每个道集内完全一致（可作为道集级字段）
      - 存在 >1 个值的道集         → 字段在道集内部跳变，不能用来切/标注道集

用法示例：
    python check_field_constant.py --sgy E:\\...\\03_noisy.sgy \
        --gather-keys "9-12,197-200" \
        --check-fields "189-192,13-16,181-184"

    python check_field_constant.py --sgy E:\\...\\15_noisy.sgy \
        --gather-keys "9-12" \
        --check-fields "189-192,17-20,8-9"

参数：
    --sgy PATH          输入 SEG-Y（自动检测端序，支持 IBM/IEEE/整数格式）
    --gather-keys       抽道集键，逗号分隔多段，如 "9-12,197-200"（1-based）
    --sort-keys         可选：排序键（默认 = 抽道集键本身，保证同组道连续）
    --check-fields      要检查的字节段，逗号分隔多段，如 "189-192,13-16"
    --endian            big/little/auto（默认 auto）
    --show-bad N        非一致道集最多打印 N 个（默认 5，0=不打印）
"""

from __future__ import annotations

import argparse
import numpy as np

from segy_reader import SegyReader
from gather import extract_gathers, parse_ranges, ranges_label


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sgy", required=True)
    ap.add_argument("--gather-keys", required=True,
                    help='抽道集键，如 "9-12,197-200"')
    ap.add_argument("--sort-keys", default=None,
                    help='排序键，如 "9-12,197-200,13-16"；缺省取抽道集键')
    ap.add_argument("--check-fields", required=True,
                    help='要检查的字段，如 "189-192,13-16"')
    ap.add_argument("--endian", default="auto", choices=["auto", "big", "little"])
    ap.add_argument("--show-bad", type=int, default=5)
    return ap.parse_args()


def main():
    a = parse_args()
    gkeys = parse_ranges(a.gather_keys)
    sort_keys = parse_ranges(a.sort_keys) if a.sort_keys else gkeys
    check_ranges = parse_ranges(a.check_fields)

    r = SegyReader(a.sgy, endian=a.endian)
    print(f"文件: {a.sgy}")
    i = r.info()
    print(f"端序={i['endian']} 格式码={i['format_code']} 总道数={i['n_traces']} ns={i['ns']}")
    print(f"抽道集键: {ranges_label(gkeys)} | 排序键: {ranges_label(sort_keys)}")
    print(f"待检查字段: {ranges_label(check_ranges)}\n")

    gathers = extract_gathers(r, sort_keys, gkeys)
    if not gathers:
        print("没有抽到任何道集")
        return
    print(f"共 {len(gathers)} 个道集，道数 min={min(g.n_traces for g in gathers)} "
          f"/ median={int(np.median([g.n_traces for g in gathers]))} "
          f"/ max={max(g.n_traces for g in gathers)}\n")

    total = len(gathers)
    widths = max(len(ranges_label([cr])) for cr in check_ranges)
    for cr in check_ranges:
        vals = r.header_field(*cr)          # 全文件一次读
        n_uniq = np.array([len(np.unique(vals[g.trace_indices])) for g in gathers])
        n_const = int((n_uniq == 1).sum())
        pct = n_const / total * 100
        bad = n_uniq[n_uniq > 1]
        med_bad = int(np.median(bad)) if len(bad) else 0
        lab = f"{ranges_label([cr]):<{widths}}"
        verdict = "✔ 完全一致" if n_const == total else "✘ 内部跳变"
        print(f"{lab}  {verdict}   每道集唯一值=1 的道集 {n_const}/{total} ({pct:.1f}%)"
              + (f"  不一致道集 median 唯一值={med_bad}" if len(bad) else ""))
        if a.show_bad and n_const < total:
            shown = 0
            for g, u in zip(gathers, n_uniq):
                if u <= 1:
                    continue
                distinct = np.unique(vals[g.trace_indices])
                vals_s = ",".join(str(int(x)) for x in distinct[:6])
                print(f"      例 {g.gather_id}: 有 {u} 个不同值 [{vals_s}{'…' if u>6 else ''}]")
                shown += 1
                if shown >= a.show_bad:
                    break
    r.close()


if __name__ == "__main__":
    main()
