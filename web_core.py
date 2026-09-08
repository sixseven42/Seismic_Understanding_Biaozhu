# -*- coding: utf-8 -*-
"""web_core.py — 标注 Web 界面可单测的纯逻辑（不依赖 gradio、无副作用）。

从原 web_app.py 抽出：选项显示文本、选项反解析、像素框→数据框映射。
web_app.py 复用本模块，删除其本地重复定义。
"""

from __future__ import annotations

import math

# render_data_only 固定输出尺寸（宽×高，像素），与 imaging.py 默认一致。
IMG_W, IMG_H = 512, 1024


def fmt_option(opt) -> str:
    """选项在 Radio 里的显示文本（带数字前缀）。"""
    return f"{opt.hotkey} {opt.label}"


def label_of(display: str) -> str:
    """把 Radio 的显示文本还原为选项 label（去掉数字前缀）。"""
    return display.split(" ", 1)[1] if " " in display else display


def pixel_box_to_data(c0, c1, n_tr: int, ns: int) -> dict:
    """两个角点像素坐标 -> {"xyxy", "traces", "samples"}。

    xyxy 为 512×1024 图像像素（含端点）；traces/samples 为原始数据半开区间 [起, 止)。
    """
    x0, x1 = sorted((float(c0[0]), float(c1[0])))
    y0, y1 = sorted((float(c0[1]), float(c1[1])))
    x0, x1 = max(0.0, x0), min(float(IMG_W - 1), x1)
    y0, y1 = max(0.0, y0), min(float(IMG_H - 1), y1)
    box = {"xyxy": [int(x0), int(y0), int(x1), int(y1)],
           "traces": [math.floor(x0 * n_tr / IMG_W),
                      min(n_tr, math.ceil((x1 + 1) * n_tr / IMG_W))],
           "samples": [math.floor(y0 * ns / IMG_H),
                       min(ns, math.ceil((y1 + 1) * ns / IMG_H))]}
    return box
