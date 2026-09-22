# -*- coding: utf-8 -*-
"""
imaging.py — 道集数据 -> seismic 配色图片
"""

from __future__ import annotations

import os
import tempfile

# matplotlib 默认配置目录可能不可写，导入前指到可写目录
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg_"))

import matplotlib
# 不强制后端：无显示环境自动回退 Agg；GUI 中由 main_window.py 指定 Qt5Agg
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# 注册中文字体（优先项目自带 Noto Sans CJK SC，中英文覆盖完整）
for _fp in (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "NotoSansCJKsc-Regular.otf"),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
):
    if os.path.isfile(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.sans-serif"] = [
            font_manager.FontProperties(fname=_fp).get_name(), "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        break


def draw_gather(ax, data: np.ndarray, dt_ms: float = 1.0,
                cmap: str = "seismic", vlim: float | None = None,
                title: str | None = None, bare: bool = False):
    """在给定的 Axes 上绘制道集（供 Qt 画布复用）。bare=True 时隐藏坐标轴。"""
    n_tr, ns = data.shape
    if vlim is None:
        vlim = float(np.abs(data).max()) or 1.0
    ax.imshow(
        data.T, aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim,
        extent=[0, n_tr, ns * dt_ms, 0], interpolation="bilinear",
    )
    if bare:
        ax.set_axis_off()
    else:
        ax.set_xlabel("道号（道集内）")
        ax.set_ylabel("时间 (ms)")
        if title:
            ax.set_title(title)
    return vlim


def render_data_only(data: np.ndarray, out_path: str,
                     cmap: str = "seismic", vlim: float | None = None,
                     size: tuple[int, int] = (512, 1024)):
    """
    纯数据图：无坐标轴/标题/边框，固定输出 size=(宽, 高) 像素，默认宽512×高1024。
    时间轴从上到下（与剖面习惯一致），道轴从左到右。
    """
    from PIL import Image
    if vlim is None:
        vlim = float(np.abs(data).max()) or 1.0
    norm = np.clip(data.T / vlim, -1, 1) * 0.5 + 0.5     # (ns, n_tr) -> [0,1]
    cm = matplotlib.colormaps[cmap]
    rgb = (cm(norm)[..., :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb)                            # 宽=n_tr, 高=ns
    img = img.resize(size, Image.BILINEAR)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    return out_path


def overlay_boxes(img_path: str, boxes: list[tuple[list[int], str, str]],
                  out_path: str) -> str:
    """
    在已渲染的纯数据图上叠加矩形框（仅供界面显示，不影响导出的训练图）。

    boxes: [(xyxy, color, label), ...]
        xyxy: [x0, y0, x1, y1] 图像像素坐标；color: 如 "#ff0000"；label: 框旁文字（可为空）
    """
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = None
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "fonts", "NotoSansCJKsc-Regular.otf")
    if os.path.isfile(font_path):
        font = ImageFont.truetype(font_path, 18)
    for xyxy, color, label in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
        for off in range(3):   # 3px 边框
            draw.rectangle([x0 - off, y0 - off, x1 + off, y1 + off], outline=color)
        if label and font is not None:
            draw.text((x0 + 4, y0 + 4), label, fill=color, font=font)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    return out_path


def overlay_polygons(img_path: str, polygons: list[tuple[list[list[int]], str, str]],
                     out_path: str) -> str:
    """Overlay closed polygon envelopes on a display image."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = None
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "fonts", "NotoSansCJKsc-Regular.otf")
    if os.path.isfile(font_path):
        font = ImageFont.truetype(font_path, 18)
    for points, color, label in polygons:
        pts = [(int(round(x)), int(round(y))) for x, y in points]
        if len(pts) < 3:
            continue
        closed = pts + [pts[0]]
        for off in range(3):
            shifted = [(x + off, y + off) for x, y in closed]
            draw.line(shifted, fill=color, width=1, joint="curve")
        if label and font is not None:
            draw.text((pts[0][0] + 4, pts[0][1] + 4), label, fill=color, font=font)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    return out_path


def render_gather(
    data: np.ndarray,
    out_path: str | None = None,
    dt_ms: float = 1.0,
    cmap: str = "seismic",
    vlim: float | None = None,
    title: str | None = None,
    dpi: int = 100,
    max_width_in: float = 12.0,
    max_height_in: float = 9.0,
):
    """
    data: (n_traces, ns) float32
    out_path: 给定则保存 PNG；否则返回 fig 供界面嵌入。
    vlim: 颜色幅值范围 ±vlim；None 时取数据绝对最大值。
    """
    n_tr, ns = data.shape
    if vlim is None:
        vlim = float(np.abs(data).max()) or 1.0

    # 图幅按比例自适应并限制上限
    aspect = (n_tr / max(ns, 1))
    w = min(max_width_in, max(4.0, 10.0 * aspect + 2))
    h = min(max_height_in, max(3.0, 8.0))

    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    draw_gather(ax, data, dt_ms=dt_ms, cmap=cmap, vlim=vlim, title=title)
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return out_path
    return fig
