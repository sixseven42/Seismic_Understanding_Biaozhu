# -*- coding: utf-8 -*-
"""检查 15_noisy.sgy 第二个 gather 附近结构：189-192 到底是不是每道集常数线号。"""
import os
import numpy as np
import segy_reader as sr

path = r"E:\AI4geo\Geo_data\异常振幅\15_noisy.sgy"
r = sr.SegyReader(path)
k1 = r.header_field(9, 12)
tr = r.header_field(13, 16)
l189 = r.header_field(189, 192)
l193 = r.header_field(193, 196)
l1720 = r.header_field(17, 20)
l181 = r.header_field(181, 184)
f08 = r.header_field(8, 9)
f28 = r.header_field(28, 29)

# 第二个 k1
uks = np.unique(k1)
print("k1 唯一值:", [int(x) for x in uks])
if len(uks) < 2:
    raise SystemExit("不足 2 个 k1")
rec2 = int(uks[1])
idx = np.flatnonzero(k1 == rec2)
print(f"\n第 2 个 k1 = {rec2}, 该炮 {len(idx)} 道\n")
# 该炮内前 120 道逐道列几个字段
print(f"{'13-16':>6} {'189-192':>8} {'193-196':>8} {'17-20':>7} {'181-184':>8} {'8-9':>5} {'28-29':>5}")
for i in idx[:120]:
    print(f"{int(tr[i]):>6} {int(l189[i]):>8} {int(l193[i]):>8} {int(l1720[i]):>7} "
          f"{int(l181[i]):>8} {int(f08[i]):>5} {int(f28[i]):>5}")

# 189-192 值 568 与 574 在全文件的分布
for v in (568, 574):
    m = l189 == v
    print(f"\n189-192={v}: 全文件 {m.sum()} 道; 按 k1 分布: {dict(zip(*np.unique(k1[m], return_counts=True)))}")
    # 它们落在哪些 193-196 组 / 8-9 组里（看是否分散）
    sub = l193[m]
    print(f"   这些道的 193-196 取值分布(前8): {dict(list(zip(*np.unique(sub, return_counts=True)))[:8])}")

# 用 193-196(本文件里能干净切~55道的字段) 看该炮道集结构：
# 找出该炮前 2 个 193-196 组的 (范围, 大小, 组内 189-192 最小/最大/步长)
cur_rec = idx
kv = l193[cur_rec]
order = np.argsort(tr[cur_rec])
kv = kv[order]; trs = tr[cur_rec][order]; l9 = l189[cur_rec][order]
b = np.flatnonzero(np.concatenate(([True], kv[1:] != kv[:-1])))
ends = np.concatenate((b[1:], [len(order)]))
print(f"\n该炮按 193-196 切成 {len(b)} 段; 前 4 段:")
for j in range(min(4, len(b))):
    seg = slice(b[j], ends[j])
    seg_l9 = l9[seg]
    print(f"  193-196={int(kv[b[j]])} 段道数={ends[j]-b[j]}  189-192: min={int(seg_l9.min())} "
          f"max={int(seg_l9.max())} 不同值数={len(np.unique(seg_l9))} 唯一值={[int(x) for x in np.unique(seg_l9)[:12]]}")

# 该文件已有的修正版?
cp = r"E:\AI4geo\Geo_data\异常振幅\15_noisy_fix_header.sgy"
print("\n修正版存在?", os.path.isfile(cp))
if os.path.isfile(cp):
    r2 = sr.SegyReader(cp)
    l2 = r2.header_field(189, 192)
    ch = np.flatnonzero(l2 != l189)
    print(f"修正版相对原文件改动 {len(ch)} 道; 其中原189-192=574 被改的: {(l189==574).sum() and int(((l189==574)&(l2!=l189)).sum())}")
    ch574 = np.flatnonzero((l189 == 574) & (l2 != l189))
    for t in ch574[:8]:
        print(f"   道#{t} 13-16={int(tr[t])} 原189={int(l189[t])} 新189={int(l2[t])} "
              f"193-196={int(l193[t])} 17-20={int(l1720[t])}")
    r2.close()
r.close()
