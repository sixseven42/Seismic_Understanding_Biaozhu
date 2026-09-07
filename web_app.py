# -*- coding: utf-8 -*-
"""
web_app.py — 地震道集标注器 Web 版（Gradio）

复用核心模块（segy_reader / gather / preprocess / imaging / labels / storage），
浏览器访问即可标注，无需 VNC/X11。

运行：
    python web_app.py            # 默认 0.0.0.0:7860
    python web_app.py --port 8000
浏览器打开：http://<服务器IP>:7860
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import tempfile

# 依赖装在项目 _vendor 目录（系统 site-packages 只读）。
# 必须放 sys.path 最前：gradio 6 依赖的新版 starlette/fastapi/uvicorn 在 _vendor 里，
# 而 conda 环境里的旧版会与之冲突；_vendor 中的 numpy 与环境同版本（2.2.6），无影响。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor"))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg_"))

import gradio as gr
import numpy as np

from segy_reader import SegyReader, SegyReadError
from gather import extract_gathers, gather_values, parse_key, key_str
from preprocess import apply_pipeline, sample_clip_values
from labels import LabelConfig, LabelConfigError
from storage import LabelStore
from imaging import render_gather, render_data_only, overlay_boxes

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_config.yaml")


# ----------------------------------------------------------------------
class Session:
    def __init__(self):
        self.reader: SegyReader | None = None
        self.file_path = ""
        self.gathers: list = []
        self.gather_key = None
        self.clip_percentile = 99.0
        self.output_dir = ""
        self.store: LabelStore | None = None
        self.label_cfg: LabelConfig | None = None
        self.current = 0
        self.partial: dict[int, dict] = {}
        # 数据增强：clip 范围 [aug_lo, aug_hi] 内随机采样 aug_count 张；0 = 不增强
        self.aug_lo = 80.0
        self.aug_hi = 100.0
        self.aug_count = 0
        # 区域框选：box_mode=正在框选的特征 key；pending_corner=第一个角点像素坐标；
        # box_partial: {道集序号: {feature_key: box|None}}，未保存的框在此暂存
        self.box_mode: str | None = None
        self.pending_corner: tuple | None = None
        self.box_partial: dict[int, dict] = {}

    def pipeline_steps(self):
        return [{"name": "clip_percentile", "params": {"percentile": self.clip_percentile}}]

    def gather_data(self, index: int):
        g = self.gathers[index]
        return apply_pipeline(self.reader.get_traces(g.trace_indices), self.pipeline_steps())


SES = Session()
try:
    CFG = LabelConfig(CONFIG_PATH)
except LabelConfigError as e:
    print(f"[警告] 标签配置加载失败: {e}")
    CFG = None


def _fmt(opt) -> str:
    """选项在 Radio 里的显示文本（带数字前缀）。"""
    return f"{opt.hotkey} {opt.label}"


def _label_of(display: str) -> str:
    return display.split(" ", 1)[1] if " " in display else display


# ----------------------------------------------------------------------
# 区域框选（bbox 特征）
# ----------------------------------------------------------------------
IMG_W, IMG_H = 512, 1024   # render_data_only 固定输出尺寸（宽×高）


def _bbox_feats():
    return [f for f in CFG.features if f.bbox] if CFG else []


def _current_boxes() -> dict:
    """当前道集的框 {feature_key: box|None}：本会话暂存优先，其次已存记录。"""
    boxes = SES.box_partial.get(SES.current)
    if boxes is not None:
        return boxes
    if SES.store and SES.gathers:
        rec = SES.store.get(SES.gathers[SES.current].gather_id)
        if rec:
            return dict(rec.get("regions") or {})
    return {}


def _pixel_box_to_data(c0, c1, n_tr: int, ns: int) -> dict:
    """两个角点像素坐标 -> {"xyxy", "traces", "samples"}。
    xyxy 为 512×1024 图像像素（含端点）；traces/samples 为原始数据半开区间 [起, 止)。"""
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


def _render_current_img() -> str:
    """当前道集显示图：纯数据图 + 已画框叠加（叠加仅供显示，导出图保持干净）。"""
    tmp = os.path.join(tempfile.mkdtemp(prefix="gather_"), "cur.png")
    render_data_only(SES.gather_data(SES.current), out_path=tmp)
    overlays = []
    for feat in _bbox_feats():
        b = _current_boxes().get(feat.key)
        if b:
            overlays.append((b["xyxy"], feat.bbox_color, feat.name))
    if not overlays:
        return tmp
    boxed = os.path.join(os.path.dirname(tmp), "cur_boxed.png")
    return overlay_boxes(tmp, overlays, boxed)


def _box_statuses() -> list[str]:
    outs = []
    boxes = _current_boxes()
    for feat in _bbox_feats():
        if SES.box_mode == feat.key:
            outs.append("👉 框选中：" + ("请点击对角" if SES.pending_corner else "请点击第一个角点"))
            continue
        b = boxes.get(feat.key)
        if b:
            (x0, y0, x1, y1), (t0, t1), (s0, s1) = b["xyxy"], b["traces"], b["samples"]
            outs.append(f"✔ 已画框：像素[{x0},{y0},{x1},{y1}]（左上→右下），道 {t0} 至 {t1}，采样 {s0} 至 {s1}")
        else:
            outs.append("未画框")
    return outs


# ----------------------------------------------------------------------
# 第 1 步：文件与道集
# ----------------------------------------------------------------------
def load_file(path: str, endian_text: str):
    path = (path or "").strip()
    if not path:
        return "⚠️ 请填写文件路径"
    endian = {"自动检测": "auto", "大端": "big", "小端": "little"}[endian_text]
    try:
        if SES.reader is not None:
            SES.reader.close()
        SES.reader = SegyReader(path, endian=endian)
    except SegyReadError as e:
        return f"❌ 读取失败: {e}"
    SES.file_path = path
    i = SES.reader.info()
    return (f"✔ 已加载 | 端序: {'小端' if i['endian']=='little' else '大端'}"
            f" | 道数: {i['n_traces']} | 每道采样点: {i['ns']}"
            f" | 采样间隔: {i['dt_ms']} ms | 格式码: {i['format_code']}"
            f" | 大小: {i['file_size_mb']} MB")


def scan_values(gkey_text: str):
    if SES.reader is None:
        return gr.CheckboxGroup(choices=[], value=[]), "⚠️ 请先加载文件"
    try:
        key = parse_key(gkey_text)
        vals = gather_values(SES.reader, key)
    except (SegyReadError, ValueError) as e:
        return gr.CheckboxGroup(choices=[], value=[]), f"❌ {e}"
    strs = [str(int(v)) for v in vals]
    return gr.CheckboxGroup(choices=strs, value=strs), f"共 {len(strs)} 个键值（默认全选，可取消勾选）"


def build_gathers(sort_text: str, gkey_text: str, selected: list[str], output_dir: str):
    if SES.reader is None:
        return "⚠️ 请先加载文件"
    if not selected:
        return "⚠️ 请先扫描并勾选键值"
    try:
        SES.sort_keys = [parse_key(t) for t in sort_text.split(",") if t.strip()]
        SES.gather_key = parse_key(gkey_text)
    except ValueError:
        return "❌ 道头键格式错误（应如 95-96，多个用英文逗号分隔）"
    try:
        SES.gathers = extract_gathers(
            SES.reader, SES.sort_keys, SES.gather_key,
            values=[int(v) for v in selected],
        )
    except SegyReadError as e:
        return f"❌ 抽道集失败: {e}"
    if not SES.gathers:
        return "⚠️ 没有抽到任何道集"
    # gather_id 前缀 = 来源文件名（不含扩展名），防止多文件标到同一输出目录时互相覆盖
    stem = os.path.splitext(os.path.basename(SES.file_path))[0]
    for g in SES.gathers:
        g.prefix = stem + "__"
    SES.output_dir = (output_dir or "").strip() or os.path.join(os.getcwd(), "output")
    sizes = [g.n_traces for g in SES.gathers]
    return (f"✔ 生成 {len(SES.gathers)} 个道集（键 {key_str(SES.gather_key)}），"
            f"每道集道数 {min(sizes)}~{max(sizes)}。输出目录: {SES.output_dir}")


# ----------------------------------------------------------------------
# 第 2 步：预处理
# ----------------------------------------------------------------------
def preview_first(clip: float):
    if not SES.gathers:
        return gr.skip(), "⚠️ 请先生成道集"
    SES.clip_percentile = float(clip)
    g = SES.gathers[0]
    tmp = os.path.join(tempfile.mkdtemp(prefix="preview_"), "preview.png")
    render_data_only(SES.gather_data(0), out_path=tmp)
    return tmp, f"预览: {g.gather_id}（{clip:g}% clip）"


def start_labeling(clip: float, aug_lo: float, aug_hi: float, aug_count: float):
    if not SES.gathers:
        return "⚠️ 请先生成道集"
    if CFG is None:
        return "❌ 标签配置未加载，请检查 label_config.yaml"
    SES.clip_percentile = float(clip)
    SES.aug_lo, SES.aug_hi = float(aug_lo), float(aug_hi)
    SES.aug_count = max(0, int(aug_count))
    if SES.aug_count > 0 and not (0 < SES.aug_lo <= SES.aug_hi <= 100):
        return "❌ 增强 clip 范围须满足 0 < 下限 ≤ 上限 ≤ 100"
    SES.label_cfg = CFG
    SES.box_mode = None
    SES.pending_corner = None
    SES.box_partial = {}
    os.makedirs(SES.output_dir, exist_ok=True)
    try:
        SES.store = LabelStore(os.path.join(SES.output_dir, "labels.jsonl"))
    except ValueError as e:
        return f"❌ {e}"
    ids = [g.gather_id for g in SES.gathers]
    pos = SES.store.next_unlabeled(ids, 0)
    SES.current = pos if pos is not None else 0
    labeled = len(SES.store.labeled_ids() & set(ids))
    aug_msg = (f" | 增强: 每次保存随机 {SES.aug_count} 张 clip∈[{SES.aug_lo:g},{SES.aug_hi:g}]"
               if SES.aug_count > 0 else " | 不增强")
    return (f"✔ 开始标注 | 已标 {labeled}/{len(SES.gathers)}，"
            f"定位到第 {SES.current+1} 个道集（断点续标）{aug_msg}")


# ----------------------------------------------------------------------
# 第 3 步：标注
# ----------------------------------------------------------------------
def _radio_updates(g):
    """根据已存记录/暂存恢复各特征 Radio 的值。"""
    rec = SES.store.get(g.gather_id) if SES.store else None
    by_key = (rec.get("labels") or {}) if rec else {}
    partial = SES.partial.get(SES.current, {})
    outs = []
    for feat in SES.label_cfg.features:
        want = by_key.get(feat.key) or partial.get(feat.name)
        val = None
        for opt in feat.options:
            if opt.label == want:
                val = _fmt(opt)
        outs.append(gr.Radio(value=val))
    return outs


def _selection_from_radios(values) -> dict:
    sel = {}
    for feat, v in zip(SES.label_cfg.features, values):
        if v:
            sel[feat.name] = _label_of(v)
    return sel


def show_current():
    """返回当前道集的所有显示组件值。"""
    g = SES.gathers[SES.current]
    tmp = _render_current_img()
    ids = [gg.gather_id for gg in SES.gathers]
    labeled = len(SES.store.labeled_ids() & set(ids))
    state = "已标注 ✔" if SES.store.is_labeled(g.gather_id) else "未标注"
    info = (f"**道集 {SES.current+1}/{len(SES.gathers)}** | 键 {g.key} = {g.value}"
            f" | {g.n_traces} 道 | {state} | 总进度 **{labeled}/{len(SES.gathers)}**")
    sel = _selection_from_radios([_radio_value(g, f) for f in SES.label_cfg.features])
    sentence = SES.label_cfg.render_sentence(sel)
    return (tmp, info, sentence, *_radio_updates(g), *_box_statuses())


def _radio_value(g, feat):
    rec = SES.store.get(g.gather_id) if SES.store else None
    by_key = (rec.get("labels") or {}) if rec else {}
    partial = SES.partial.get(SES.current, {})
    want = by_key.get(feat.key) or partial.get(feat.name)
    for opt in feat.options:
        if opt.label == want:
            return _fmt(opt)
    return None


def on_radio_change(*radio_values):
    if SES.label_cfg is None or not SES.gathers:
        return ""
    sel = _selection_from_radios(radio_values)
    SES.partial[SES.current] = sel
    return SES.label_cfg.render_sentence(sel)


def enter_box_mode(feat_key: str):
    """点「▣ 框选」：进入该特征的框选模式，等待图上两次点击。"""
    n = len(_bbox_feats())
    if not SES.gathers or SES.store is None:
        return ("⚠️ 请先开始标注", *([""] * n))
    SES.box_mode = feat_key
    SES.pending_corner = None
    feat = next(f for f in _bbox_feats() if f.key == feat_key)
    return (f"👉 正在框选「{feat.name}」：请在左侧图上点击第一个角点", *_box_statuses())


def clear_box(feat_key: str):
    """点「✕ 清除」：删除该特征的框。"""
    n = len(_bbox_feats())
    if not SES.gathers or SES.store is None:
        return (gr.skip(), "⚠️ 请先开始标注", *([""] * n))
    if SES.box_mode == feat_key:
        SES.box_mode = None
        SES.pending_corner = None
    boxes = dict(_current_boxes())
    boxes[feat_key] = None
    SES.box_partial[SES.current] = boxes
    feat = next(f for f in _bbox_feats() if f.key == feat_key)
    return (_render_current_img(), f"已清除「{feat.name}」的框", *_box_statuses())


def on_img_select(evt: gr.SelectData):
    """图上点击：框选模式下依次记录两个角点。"""
    n = len(_bbox_feats())
    if not SES.gathers or SES.store is None:
        return (gr.skip(), "⚠️ 请先开始标注", *([""] * n))
    if SES.box_mode is None:
        return (gr.skip(), "（先在特征下方点「▣ 框选」，再在图上取点）", *_box_statuses())
    x, y = float(evt.index[0]), float(evt.index[1])
    if SES.pending_corner is None:
        SES.pending_corner = (x, y)
        return (gr.skip(), f"已记录第一个角点 ({int(x)},{int(y)})，请点击对角", *_box_statuses())
    g = SES.gathers[SES.current]
    box = _pixel_box_to_data(SES.pending_corner, (x, y), g.n_traces,
                             SES.reader.info()["ns"])
    feat = next(f for f in _bbox_feats() if f.key == SES.box_mode)
    boxes = dict(_current_boxes())
    boxes[feat.key] = box
    SES.box_partial[SES.current] = boxes
    SES.box_mode = None
    SES.pending_corner = None
    return (_render_current_img(), f"✔「{feat.name}」框选完成", *_box_statuses())


def navigate(delta: int):
    if not SES.gathers or SES.store is None:
        return (gr.skip(), "⚠️ 请先开始标注", "", *([gr.Radio()] * len(CFG.features)),
                *(["…"] * len(_bbox_feats())))
    SES.current = max(0, min(len(SES.gathers) - 1, SES.current + delta))
    return show_current()


def inherit_previous():
    """把最近一张【已标注】道集的类别选项填充到当前道集（仅填充不保存）。
    跳过中间未标注的道集；框是道集专属的，不继承。"""
    if not SES.gathers or SES.store is None:
        return ("⚠️ 请先开始标注", *([gr.Radio()] * len(CFG.features)))
    src_idx, src_rec = None, None
    for i in range(SES.current - 1, -1, -1):
        rec = SES.store.get(SES.gathers[i].gather_id)
        if rec and rec.get("labels"):
            src_idx, src_rec = i, rec
            break
    if src_rec is None:
        # 前面没有任何已标注道集：保持当前已选不变
        cur = _radio_updates(SES.gathers[SES.current])
        cur_sel = _selection_from_radios(
            [_radio_value(SES.gathers[SES.current], f) for f in CFG.features])
        return ("⚠️ 前面没有已标注的道集可继承 | " + CFG.render_sentence(cur_sel), *cur)
    by_key = src_rec["labels"]
    sel = {feat.name: by_key[feat.key] for feat in CFG.features if by_key.get(feat.key)}
    outs = []
    for feat in CFG.features:
        want = sel.get(feat.name)
        outs.append(gr.Radio(value=next((_fmt(o) for o in feat.options if o.label == want), None)))
    SES.partial[SES.current] = sel
    src = SES.gathers[src_idx]
    skipped = SES.current - 1 - src_idx
    hint = f"（跳过中间 {skipped} 张未标注）" if skipped else ""
    return (f"已继承最近已标注道集（{src.gather_id}）的选项{hint}；框不继承，需另行画框："
            + CFG.render_sentence(sel), *outs)


def save_and_next(*radio_values):
    if not SES.gathers or SES.store is None:
        return (gr.skip(), "⚠️ 请先开始标注", "", *([gr.Radio()] * len(CFG.features)),
                *(["…"] * len(_bbox_feats())))
    g = SES.gathers[SES.current]
    sel = _selection_from_radios(radio_values)
    errs = SES.label_cfg.validate_selection(sel)
    if errs:
        cur = show_current()
        return (cur[0], "⚠️ 尚有未选特征：" + "、".join(errs), cur[2], *cur[3:])
    # 区域框校验：bbox 特征为「存在/已压制但有残留」时必须已画框；「不存在」则框置空
    boxes = SES.box_partial.get(SES.current)
    rec_regions = (SES.store.get(g.gather_id) or {}).get("regions") or {}
    regions, box_errs = {}, []
    for feat in SES.label_cfg.features:
        if not feat.bbox:
            continue
        lab = sel.get(feat.name)
        if lab == "不存在":
            regions[feat.key] = None
            continue
        box = None
        if boxes is not None and feat.key in boxes:
            box = boxes[feat.key]
        elif feat.key in rec_regions:
            box = rec_regions[feat.key]
        if box is None:
            box_errs.append(f"「{feat.name}」为「{lab}」但未画框")
        regions[feat.key] = box
    if box_errs:
        cur = show_current()
        return (cur[0], "⚠️ " + "；".join(box_errs), cur[2], *cur[3:])
    sentence = SES.label_cfg.render_sentence(sel)
    # 原始数据（无预处理）：存 npy + 作为增强数据源
    raw = SES.reader.get_traces(g.trace_indices).astype(np.float32)
    rel_npy = os.path.join("npy", f"{g.gather_id}.npy")
    abs_npy = os.path.join(SES.output_dir, rel_npy)
    os.makedirs(os.path.dirname(abs_npy), exist_ok=True)
    np.save(abs_npy, raw)

    base_rec = {
        "gather_id": g.gather_id,
        "gather_key": g.key,
        "gather_value": g.value,
        "n_traces": g.n_traces,
        "labels": SES.label_cfg.to_record_labels(sel),
        "sentence": sentence,
        "regions": regions,
        "npy_path": rel_npy,
        "sort_keys": ", ".join(key_str(k) for k in SES.sort_keys),
        "extract_key": key_str(SES.gather_key),
        "source_file": SES.file_path,
    }

    n_aug = SES.aug_count
    if n_aug > 0:
        # 清理该道集旧的增强记录与图片，再重新随机采样
        SES.store.remove_augmented(g.gather_id)
        img_dir = os.path.join(SES.output_dir, "images")
        for old in glob.glob(os.path.join(img_dir, f"{g.gather_id}__clip*.png")):
            os.remove(old)
        clips = sample_clip_values(SES.aug_lo, SES.aug_hi, n_aug)
        for cp in clips:
            data = apply_pipeline(raw, [{"name": "clip_percentile",
                                         "params": {"percentile": float(cp)}}])
            aug_id = f"{g.gather_id}__clip{cp:g}"
            rel_img = os.path.join("images", f"{aug_id}.png")
            render_data_only(data, out_path=os.path.join(SES.output_dir, rel_img))
            SES.store.upsert({**base_rec,
                              "gather_id": aug_id,
                              "image_path": rel_img,
                              "augmented_from": g.gather_id,
                              "clip_percentile": float(cp)})
        base_rec["image_path"] = None   # 只存增强图，基础图不单独保存
    else:
        rel_img = os.path.join("images", f"{g.gather_id}.png")
        render_data_only(SES.gather_data(SES.current),
                         out_path=os.path.join(SES.output_dir, rel_img))
        base_rec["image_path"] = rel_img
    SES.store.upsert(base_rec)
    SES.partial.pop(SES.current, None)
    SES.box_partial.pop(SES.current, None)
    ids = [gg.gather_id for gg in SES.gathers]
    nxt = SES.store.next_unlabeled(ids, SES.current + 1)
    if nxt is None:
        nxt = SES.store.next_unlabeled(ids, 0)
    if nxt is None:
        cur = show_current()
        return (cur[0], "🎉 全部道集均已标注！" + cur[1], cur[2], *cur[3:])
    SES.current = nxt
    return show_current()


# ----------------------------------------------------------------------
def build_app() -> gr.Blocks:
    feat_names = [f.name for f in CFG.features] if CFG else []
    with gr.Blocks(title="地震道集标注器") as demo:
        gr.Markdown("# 地震道集标注器（Web 版）")

        with gr.Accordion("第 1 步：文件与道集设置", open=True):
            with gr.Row():
                file_path = gr.Textbox(label="sgy/segy 文件路径", scale=3,
                                       placeholder="/path/to/xxx.sgy")
                endian = gr.Dropdown(["自动检测", "大端", "小端"], value="自动检测",
                                     label="字节序", scale=1)
                btn_load = gr.Button("读取文件信息", scale=1)
            file_info = gr.Markdown("尚未加载文件")
            with gr.Row():
                sort_keys = gr.Textbox(label="排序键（逗号分隔，1-based）", value="95-96, 13-16")
                gkey = gr.Textbox(label="抽道集键", value="95-96")
                btn_scan = gr.Button("扫描键值")
            values_box = gr.CheckboxGroup(choices=[], value=[], label="键值（勾选要抽取的）")
            scan_info = gr.Markdown("")
            with gr.Row():
                output_dir = gr.Textbox(label="输出目录",
                                        value=os.path.join(os.getcwd(), "output"))
                btn_build = gr.Button("生成道集", variant="primary")
            build_info = gr.Markdown("")

        with gr.Accordion("第 2 步：预处理设置", open=True):
            with gr.Row():
                clip = gr.Number(label="clip 分位数 (%)", value=99.0, minimum=50, maximum=100)
                btn_preview = gr.Button("预览第一个道集")
                btn_start = gr.Button("开始标注", variant="primary")
            with gr.Row():
                aug_lo = gr.Number(label="增强 clip 下限 (%)", value=80.0, minimum=1, maximum=100)
                aug_hi = gr.Number(label="增强 clip 上限 (%)", value=100.0, minimum=1, maximum=100)
                aug_count = gr.Number(label="增强张数（0=不增强）", value=0, minimum=0,
                                      maximum=50, step=1, precision=0)
            prep_info = gr.Markdown("")
            preview_img = gr.Image(label="预处理预览", type="filepath", height=400)

        with gr.Accordion("第 3 步：标注", open=True):
            anno_info = gr.Markdown("（点开始标注后显示）")
            feats = CFG.features if CFG else []
            half = (len(feats) + 1) // 2   # 特征分两列，压缩整体高度
            radios = []
            box_statuses = []
            box_buttons = []   # (feature_key, 框选按钮, 清除按钮)
            with gr.Row():
                # 左：当前道集图（框选模式下点击取点）
                with gr.Column(scale=3):
                    cur_img = gr.Image(label="当前道集（点「▣ 框选」后在图上点两个对角画框）",
                                       type="filepath", height=640)
                # 右：句子预览 + 两列特征选项
                with gr.Column(scale=2):
                    sentence = gr.Textbox(label="句子预览", interactive=False, lines=3)
                    with gr.Row():
                        for col_feats in (feats[:half], feats[half:]):
                            with gr.Column():
                                for feat in col_feats:
                                    i = feats.index(feat)
                                    r = gr.Radio(choices=[_fmt(o) for o in feat.options],
                                                 label=f"{i+1}. {feat.name}", value=None)
                                    radios.append(r)
                                    if feat.bbox:
                                        with gr.Row():
                                            btn_box = gr.Button(f"▣ 框选{feat.name}", size="sm")
                                            btn_clear = gr.Button("✕ 清除", size="sm")
                                        box_statuses.append(gr.Markdown("未画框"))
                                        box_buttons.append((feat.key, btn_box, btn_clear))
            # 底部：操作按钮
            with gr.Row():
                btn_prev = gr.Button("← 上一张")
                btn_skip = gr.Button("跳过")
                btn_next = gr.Button("下一张 →")
                btn_inherit = gr.Button("⧉ 继承最近已标注")
                btn_save = gr.Button("保存并下一张", variant="primary")

        # ---------------- 事件 ----------------
        btn_load.click(load_file, [file_path, endian], file_info)
        btn_scan.click(scan_values, gkey, [values_box, scan_info])
        btn_build.click(build_gathers, [sort_keys, gkey, values_box, output_dir], build_info)
        btn_preview.click(preview_first, clip, [preview_img, prep_info])
        btn_start.click(start_labeling, [clip, aug_lo, aug_hi, aug_count], prep_info)

        anno_outputs = [cur_img, anno_info, sentence, *radios, *box_statuses]
        for r in radios:
            r.change(on_radio_change, radios, sentence)
        for key, btn_box, btn_clear in box_buttons:
            btn_box.click(lambda k=key: enter_box_mode(k), None, [anno_info, *box_statuses])
            btn_clear.click(lambda k=key: clear_box(k), None,
                            [cur_img, anno_info, *box_statuses])
        cur_img.select(on_img_select, None, [cur_img, anno_info, *box_statuses])
        btn_prev.click(lambda: navigate(-1), None, anno_outputs)
        btn_next.click(lambda: navigate(1), None, anno_outputs)
        btn_skip.click(lambda: navigate(1), None, anno_outputs)
        btn_save.click(save_and_next, radios, anno_outputs)
        btn_inherit.click(inherit_previous, None, [sentence, *radios])
        # 开始标注后立即显示当前道集
        btn_start.click(lambda: show_current() if (SES.gathers and SES.store) else
                        (gr.skip(), "", "", *([gr.Radio()] * len(feat_names)),
                         *(["…"] * len(box_statuses))),
                        None, anno_outputs)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    demo = build_app()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
