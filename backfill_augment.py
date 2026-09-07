# -*- coding: utf-8 -*-
"""
backfill_augment.py — 为已标注记录回补 clip 随机采样增强图

对 labels.jsonl 中每条基础记录：
  1. 删除原单张基础图（images/<gather_id>.png）
  2. 从 npy/<gather_id>.npy 读原始数据，按来源文件对应的 clip 范围随机采 N 个值
  3. 每个 clip 值渲染一张增强图 images/<gather_id>__clip<值>.png
  4. 每张增强图一条独立记录（labels/sentence 与基础记录相同，
     含 augmented_from / clip_percentile 字段）；基础记录 image_path 置 null

clip 范围（用户拍板 2026-08-21）：
  01 开头文件（01SB21 系列）: 95 ~ 99.9
  02/03 开头文件（02WL、03XCN）: 90 ~ 99.9
每条记录独立重新随机采样。

用法：python backfill_augment.py output/labels.jsonl [张数N，默认5]
幂等：重跑会先清理各道集旧增强记录与图片。
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tempfile

import numpy as np

from preprocess import apply_pipeline, sample_clip_values
from imaging import render_data_only

N_AUG_DEFAULT = 5


def clip_range_for(source_file: str) -> tuple[float, float]:
    stem = os.path.basename(source_file)
    if stem.startswith("01"):
        return (95.0, 99.9)
    if stem.startswith(("02", "03")):
        return (90.0, 99.9)
    raise ValueError(f"未配置 clip 范围的文件: {stem}")


def main(jsonl_path: str, n_aug: int):
    out_dir = os.path.dirname(os.path.abspath(jsonl_path))
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    with open(jsonl_path, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    base_recs = [r for r in recs if "augmented_from" not in r]
    print(f"基础记录 {len(base_recs)} 条（总 {len(recs)} 条），每条增强 {n_aug} 张")

    # 幂等：先清掉所有旧增强记录与对应图片
    old_aug_ids = [r["gather_id"] for r in recs if "augmented_from" in r]
    for aid in old_aug_ids:
        p = os.path.join(img_dir, f"{aid}.png")
        if os.path.isfile(p):
            os.remove(p)
    recs = base_recs
    if old_aug_ids:
        print(f"已清理旧增强记录 {len(old_aug_ids)} 条")

    new_recs = []
    ok = fail = 0
    for i, rec in enumerate(base_recs):
        gid = rec["gather_id"]
        try:
            lo, hi = clip_range_for(rec["source_file"])
            raw = np.load(os.path.join(out_dir, rec["npy_path"]))   # 原始数据
            # 删原单张基础图，基础记录 image_path 置 null
            if rec.get("image_path"):
                p = os.path.join(out_dir, rec["image_path"])
                if os.path.isfile(p):
                    os.remove(p)
                rec["image_path"] = None
            new_recs.append(rec)
            clips = sample_clip_values(lo, hi, n_aug)   # 每条独立、无放回采样
            for cp in clips:
                data = apply_pipeline(raw, [{"name": "clip_percentile",
                                             "params": {"percentile": float(cp)}}])
                aug_id = f"{gid}__clip{cp:g}"
                rel_img = os.path.join("images", f"{aug_id}.png")
                render_data_only(data, out_path=os.path.join(out_dir, rel_img))
                new_recs.append({**rec,
                                 "gather_id": aug_id,
                                 "image_path": rel_img,
                                 "augmented_from": gid,
                                 "clip_percentile": float(cp)})
            ok += 1
            if (i + 1) % 20 == 0 or i == len(base_recs) - 1:
                print(f"  进度 {i+1}/{len(base_recs)} ...")
        except Exception as e:
            fail += 1
            new_recs.append(rec)
            print(f"  [失败] {gid}: {e}")

    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for rec in new_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, jsonl_path)
    print(f"完成: 成功 {ok}，失败 {fail} | 总记录 {len(new_recs)} 条（基础 {ok} + 增强 {ok*n_aug}）")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    if not (2 <= len(sys.argv) <= 3):
        print("用法: python backfill_augment.py <labels.jsonl 路径> [张数N，默认5]")
        sys.exit(2)
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else N_AUG_DEFAULT))
