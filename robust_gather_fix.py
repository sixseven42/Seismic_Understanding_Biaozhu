# -*- coding: utf-8 -*-
"""
robust_gather_fix.py — 线号修正（两步法，保证 gather 道号连续）

规则（由需求确定）：
  边界 = 相邻道线号跳变处；沿 13-16 把每炮扫描成“同线号连续段”。
  （可选容差：--tol N 时每炮先取众数线号为中心，|线号-中心|≤N 的统一写成中心值，
  如中心 721、N=3 → 718-724 都算同一条 721 线，再做下面的分段。）
  凡长度 ≤ thr 道的段（离群点 / 零星重复的线号）整段并入相邻更长的段（归入所在块），
  反复并入直到没有 ≤ thr 的小段。最终每个 gather = 道号连续一段 + 单一有效线号 + 长度 > thr。

用法：
    # 干跑（打印修正前后道集统计 + 道号连续占比，不写文件）
    python robust_gather_fix.py --sgy X.sgy --first-key 9-12 --line-bytes 189-192 --thr 30

    # 写修正后的新 sgy
    python robust_gather_fix.py --sgy X.sgy --first-key 9-12 --line-bytes 189-192 --thr 30 \
        --out X.fixed.sgy

参数：
    --sgy PATH        输入 SEG-Y（自动端序，支持 IBM/IEEE/整数格式）
    --first-key K     第一键字节，如 9-12（默认 9-12）
    --line-bytes R    线号字节之一：173-174 / 189-192 / 193-196 / 197-200
    --trace-num R     道号字节（默认 13-16）
    --thr T           段并入阈值（默认 30）：≤T 道的同线号段并入相邻更长段
    --tol N           线号容差（默认 0）：每炮先取众数线号作中心，|线号-中心|≤N 的
                      都视为同一条线并统一写成中心值（如 721±3 → 718-724 算 721）；
                      0=严格相同（旧行为）。
    --out PATH        可选：修正后另存的 sgy（缺省=仅干跑）
    --endian          auto/big/little
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np

from segy_reader import SegyReader, FILE_HEADER_SIZE, HEADER_SIZE
from gather import parse_ranges, ranges_label

LINE_CANDIDATES = [(173, 174), (189, 192), (193, 196), (197, 200)]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sgy", required=True)
    ap.add_argument("--first-key", default="9-12")
    ap.add_argument("--line-bytes", default="189-192")
    ap.add_argument("--trace-num", default="13-16")
    ap.add_argument("--thr", type=int, default=30,
                    help="段并入阈值：≤thr 道的同线号段视为离群/零星重复，并入相邻更长段")
    ap.add_argument("--tol", type=int, default=0,
                    help="线号容差：每炮取众数线号为中心，|线号-中心|≤tol 的统一写成中心值"
                         "（如 721±3 → 718-724 都算 721；0=严格相同）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--endian", default="auto", choices=["auto", "big", "little"])
    return ap.parse_args()


def split_stats(k1, line):
    """按 (k1, line) 精确分组的组道数数组。"""
    n = len(k1)
    order = np.lexsort((np.zeros(n), line, k1))
    a = k1[order]; b = line[order]
    chg = np.flatnonzero(np.concatenate(([True], (a[1:] != a[:-1]) | (b[1:] != b[:-1]))))
    ends = np.concatenate((chg[1:], [n]))
    return ends - chg


def summarize(tag, sizes):
    s = np.sort(sizes)
    tiny = (sizes <= 20).sum() / len(sizes) * 100
    print(f"{tag}: 道集数 {len(sizes)}  道数[min {s[0]} / median {int(np.median(s))} / max {s[-1]}]  "
          f"≤20道碎片 {tiny:.1f}%")


def contiguity(k1, line, tr):
    """统计按 (k1, line) 分组的 gather 中，道号(13-16)“单段不被拆”的占比。

    单段 = 该 (k1,线号) 组在按道号排序后只出现一段（没有其它线号的组插在中间）。
    数据本身若有缺失道号（13-16 空号）不算破坏单段性。
    """
    n = len(k1)
    total = 0
    multi = 0
    for rec in np.unique(k1):
        m = np.flatnonzero(k1 == rec)
        if len(m) == 0:
            continue
        order = np.argsort(tr[m], kind="stable")
        lbl = line[m][order]
        bd = np.flatnonzero(np.concatenate(([True], lbl[1:] != lbl[:-1])))
        runs = np.diff(np.concatenate((bd, [len(m)])))
        # 每(炮,线号)算一个组；出现在 ≥2 段即为被拆
        _, inv = np.unique(lbl, return_inverse=True)
        cnt_runs = np.bincount(inv[bd])          # 每组的段数（每段起点处记一次）
        gcount = len(np.unique(lbl))
        total += gcount
        multi += int((cnt_runs > 1).sum())
    return total - multi, total


def two_step_correct(k1, tr, line, thr: int, tol: int = 0):
    """线号修正（按”线号跳变分界 + 短段并入相邻”）。返回 newline（每道修正后线号）。

    规则：
      · 边界 = 相邻道线号跳变处（沿 13-16 逐道扫描成同线号连续段）。
      · tol > 0 时先做容差归一带：取每炮出现最多的线号（众数）作中心，凡
        |线号 − 中心| ≤ tol 的道都视为同一条线并统一写成中心值（如中心 721、tol=3，
        718–724 都算作 721）；带外值保持原值。tol=0 时跳过，等价于”严格相同”。
      · 长度 ≤ thr 的段视为离群 / 零星重复，整段并入相邻更长的段（归入所在块）。
      · 留下的每个段都是：道号连续 + 单一有效线号 + 长度 > thr。
      这样同一 gather 的道号(13-16)必然连续，且块内线号统一。
    """
    newline = line.copy()
    for rec in np.unique(k1):
        idx = np.flatnonzero(k1 == rec)
        order = np.argsort(tr[idx], kind="stable")
        lin = line[idx][order].astype(np.int64)

        # 容差归一带（tol>0）：本炮众数线号为中心，|值-中心|≤tol 统一写为中心值
        if tol > 0:
            u, c = np.unique(lin, return_counts=True)
            center = u[c.argmax()]           # np.unique 升序，并列众数取最小，确定性
            near = np.abs(lin - center) <= tol
            lin = np.where(near, center, lin)

        # 同线号连续段
        chg = np.flatnonzero(np.concatenate(([True], lin[1:] != lin[:-1])))
        starts = chg.tolist()
        vals = lin[chg].tolist()
        ends = chg[1:].tolist() + [len(lin)]

        # 反复并入长度 ≤ thr 的最小段到相邻更长段
        changed = True
        while changed:
            changed = False
            lens = [ends[i] - starts[i] for i in range(len(starts))]
            cand = [i for i, ln in enumerate(lens) if ln <= thr and len(starts) > 1]
            if not cand:
                break
            i = min(cand, key=lambda j: (lens[j], starts[j]))
            # 邻居选择：更长者；相等取前一个
            if i == 0:
                j = 1
            elif i == len(starts) - 1:
                j = i - 1
            else:
                j = i - 1 if lens[i - 1] >= lens[i + 1] else i + 1
            # 把小段 i 并入邻居 j（线号取 j 的），合并后保留覆盖两者的连续块
            if i < j:
                # i 在左、j 在右：新块从 i 起点到 j 终点，线号取 vals[j]，占用 i 槽，删除 j
                ends[i] = ends[j]
                vals[i] = vals[j]
                del starts[j]; del ends[j]; del vals[j]
            else:
                # j 在左、i 在右：新块保持 j，终点扩到 i 终点，删除 i
                ends[j] = ends[i]
                del starts[i]; del ends[i]; del vals[i]
            changed = True
        # 写回每个段线号
        for s, e, v in zip(starts, ends, vals):
            newline[idx[order][s:e]] = v
    return newline


def main():
    a = parse_args()
    k1r = parse_ranges(a.first_key)
    lbr = parse_ranges(a.line_bytes)
    trr = parse_ranges(a.trace_num)
    if len(k1r) != 1 or len(lbr) != 1 or len(trr) != 1:
        raise SystemExit("first-key / line-bytes / trace-num 均须为单个字节段")

    r = SegyReader(a.sgy, endian=a.endian)
    i = r.info()
    print(f"文件: {a.sgy}")
    print(f"端序={i['endian']} 格式码={i['format_code']} 总道数={i['n_traces']} ns={i['ns']}")
    k1 = r.header_field(*k1r[0])
    line = r.header_field(*lbr[0])
    tr = r.header_field(*trr[0])

    print(f"\n第一键={ranges_label(k1r)}  线号字节={ranges_label(lbr)}  道号={ranges_label(trr)}"
          f"  thr={a.thr}  tol={a.tol}")
    summarize("修正前", split_stats(k1, line))
    ok0, tot0 = contiguity(k1, line, tr)
    print(f"修正前 gather 道号连续占比: {ok0}/{tot0} = {ok0/tot0*100:.1f}%")

    newline = two_step_correct(k1, tr, line, a.thr, tol=a.tol)
    n_changed = int((newline != line).sum())
    print(f"\n被改写的道数 = {n_changed}（{n_changed/len(k1)*100:.2f}%）")
    summarize("修正后", split_stats(k1, newline))
    ok1, tot1 = contiguity(k1, newline, tr)
    print(f"修正后 gather 道号连续占比: {ok1}/{tot1} = {ok1/tot1*100:.1f}%")

    if not a.out:
        print("\n[干跑] 未写文件；确认效果后加 --out 生成修正后的 sgy")
        return

    print(f"\n复制 {a.sgy} -> {a.out}（文件较大，请耐心）", flush=True)
    shutil.copyfile(a.sgy, a.out)
    a0, b0 = lbr[0]
    w = b0 - a0 + 1
    ee = "big" if i["endian"] == "big" else "little"
    trace_bytes = HEADER_SIZE + i["ns"] * (4 if i["format_code"] in (1, 5) else
                                          {2: 4, 3: 2, 8: 1}.get(i["format_code"], 4))
    mm = np.memmap(a.out, dtype=np.uint8, mode="r+")
    changed = np.flatnonzero(newline != line)
    n = len(changed)
    print(f"改写 {n} 道线号字节 ...", flush=True)
    for idx in range(0, n, 1000):
        chunk = changed[idx:idx + 1000]
        for t in chunk:
            off = FILE_HEADER_SIZE + int(t) * trace_bytes + (a0 - 1)
            mm[off:off + w] = np.frombuffer(int(newline[t]).to_bytes(w, ee, signed=True), np.uint8)
        mm.flush()
    del mm
    print(f"✔ 已生成: {a.out}  大小 {os.path.getsize(a.out)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
