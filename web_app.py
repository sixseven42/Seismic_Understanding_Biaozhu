# -*- coding: utf-8 -*-
"""
web_app.py — 地震道集标注器（Gradio 多用户中央服务版）

复用核心模块（segy_reader / gather / preprocess / imaging / labels / storage），
并接入 jobmanager（作业池 + 租约）、users（账号/角色）、cloudsync（COS 回传）。
浏览器访问即可标注，无需 VNC/X11。

运行：
    python web_app.py            # 默认 0.0.0.0:7860
    python web_app.py --port 8000
浏览器打开：http://<服务器IP>:7860
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from web_core import (ABSENT_LABEL, active_filter, box_note, box_target, fmt_option,
                      jobs_progress_html, label_of, parse_filter, pixel_polygon_to_data,
                      prev_gid, staff_counts_html)
from labels import LabelConfig
from users import Accounts
from jobmanager import JobManager, JobError, N_AUG
from cloudsync import Uploader, load_config
from gather import parse_ranges, gather_values, ranges_label
from segy_reader import SegyReader, SegyReadError
from imaging import overlay_polygons

# matplotlib 默认配置目录可能不可写，导入前指到可写目录
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg_"))

# 面波区「⚙ 滤波」挂在这个特征下（需求指定）；改 label_config.yaml 里的 key 时同步改这里
FILTER_FEAT_KEY = "surface_wave"
DISPLAY_CLIP_MIN = 90.0
DISPLAY_CLIP_MAX = 99.9
DISPLAY_CLIP_STEP = 0.1

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_config.yaml")
USERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.yaml")
COS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cos_config.yaml")
JOBS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
os.makedirs(JOBS_ROOT, exist_ok=True)

# 启动引导：账号文件不存在时写入默认 admin（boss/boss123）
Accounts.ensure_default(USERS_PATH)

CFG = LabelConfig(CONFIG_PATH)
ACC = Accounts(USERS_PATH)
UP = Uploader(load_config(COS_PATH))


def make_jm(root: str) -> JobManager:
    """建 JobManager 并挂上云回传钩子。

    增强图的 PNG 现在是**后台线程**画的，所以上传必须挂在「图画完」的回调上，
    不能在保存请求里就地 enqueue（那时图还不存在）。回调里惰性取模块级 UP，
    测试替换 web_app.UP 后依然生效。

    测试若要替换 web_app.JM，请用本函数而不是直接 JobManager(...)：否则钩子不在，
    上传那一路就等于没被覆盖到。
    """
    jm = JobManager(root, CFG)
    jm.set_aug_hook(lambda job, rels: UP.enqueue(job.job_id, job.output_dir, rels))
    return jm


JM = make_jm(JOBS_ROOT)
UP.start()
JM.load_all()
# 补画上次进程被杀时没来得及画的增强图。放后台线程，绝不拖慢启动。
threading.Thread(target=JM.sweep_missing_aug, name="aug-sweep", daemon=True).start()

# 每用户工作态：{username: {"job_ids","job_id","gid","partial","boxes","filter","filter_params"}}
#   job_ids: 选中的作业（可多个）—— 抽道集时把它们的可领道集摊平后等概率随机抽
#   job_id:  当前道集所属作业（保存/进度/滤波参数都按它走，随每张道集变）
#   partial: {特征名: 选项label}（未保存的 Radio 选择）；boxes: {特征key: box|None}（未保存的框）
#   filter: {gid, f1..f4}（仅对当前道集生效，换道集即失效）
#   filter_params: {job_id: {f1..f4}}（按作业记忆的滤波参数，同作业内统一）
WORK: dict[str, dict] = {}
WORK_LOCK = threading.RLock()


def wstate(user: str) -> dict:
    with WORK_LOCK:
        return WORK.setdefault(user, {})


# ---- 「↩ 返回上一张」的浏览历史 ----
# st["hist"] = 当前道集**之前**访问过的道集（旧→新），不含当前道集。
# 用浏览历史而不是「我标注的」列表顺序：那张表按首次标注时间排序（storage.LabelStore.upsert
# 对已存在的 gid 不挪位），修正保存不会让记录回到末尾，靠它「返回上一张」会翻到别处去。
def _push_hist(st: dict, gid: str | None):
    """把 gid 压进浏览历史；空值忽略，连续重复的不重复压。"""
    if not gid:
        return
    hist = [g for g in (st.get("hist") or []) if g]
    if hist and hist[-1] == gid:
        return
    hist.append(gid)
    st["hist"] = hist


def _has_unsaved_input(st: dict) -> bool:
    """当前道集有没有未保存的作答（选项或框）。

    用于「返回上一张」前判断能不能直接把手上这张归还回池：刚领来一张就点回去的情况
    一个字都没填，不该拦；真答过了才值得问一句。服务端状态忠实反映用户输入 ——
    改选项走 on_radio_change 写 st['partial']，画框走 /api/anno_box 写 st['boxes']。
    """
    return (any((st.get("partial") or {}).values())
            or any((st.get("boxes") or {}).values()))


def _visit(st: dict, job_id: str, gid: str):
    """把显示切到 (job_id, gid)，并把**原来那张**压进浏览历史。"""
    old = st.get("gid")
    if old and old != gid:
        _push_hist(st, old)
    st["job_id"] = job_id
    st["gid"] = gid
    st["residual_layer"] = 0


def _user(request) -> str:
    return (request.username or "") if request is not None else ""


def _is_admin(user: str) -> bool:
    return bool(user) and ACC.role(user) == "admin"


def bbox_feats():
    return [f for f in CFG.features if f.bbox]


def active_bbox_feats(selection: dict):
    return [f for f in bbox_feats() if f.active_for(selection or {})]


def _job_choices(user: str) -> list[str]:
    """标注区作业下拉候选：admin 看全部；标注者看 open + 自己标过的。"""
    if _is_admin(user):
        return [j["job_id"] for j in JM.list_jobs()]
    return [j["job_id"] for j in JM.list_jobs()
            if j["state"] == "open" or JM.mine(j["job_id"], user)]


def _norm_job_ids(job_ids) -> list[str]:
    """把下拉的值（多选是 list；单选/空是 str/None）统一成去重的作业 id 列表。"""
    if not job_ids:
        return []
    if isinstance(job_ids, str):
        job_ids = [job_ids]
    return [j for j in dict.fromkeys(str(x) for x in job_ids if x)]


def _mine_choices(user: str, job_ids) -> list[str]:
    """「我标注的」= 选中作业的并集（多作业下按各作业内顺序拼接）。"""
    out = []
    for jid in _norm_job_ids(job_ids):
        try:
            out.extend(r["gather_id"] for r in JM.mine(jid, user))
        except JobError:
            continue
    return out


def _job_of_gid(user: str, gid: str, job_ids) -> str | None:
    """该 gid 属于选中作业里的哪一个（重开「我标注的」时定位）。

    注：同一个 sgy 文件建的两个作业会产生相同 gid，此时取第一个匹配到的作业。
    """
    for jid in _norm_job_ids(job_ids):
        try:
            if JM.record(jid, gid) is not None:
                return jid
        except JobError:
            continue
    return None


# ----------------------------------------------------------------------
# 标注区纯逻辑（per-user 状态）
# ----------------------------------------------------------------------
FILTER_KEYS = ("f1", "f2", "f3", "f4")


def _job_filter_params(st: dict) -> dict | None:
    """该用户当前作业已存的滤波参数（无则 None）。"""
    return (st.get("filter_params") or {}).get(st.get("job_id"))


def _display_clip(job, st: dict) -> float:
    """当前作业共享的显示 clip；未调过时取建作业预览值。"""
    saved = getattr(job, "display_clip", job.clip)
    try:
        value = float(saved)
    except (TypeError, ValueError):
        value = float(job.clip)
    return round(min(DISPLAY_CLIP_MAX, max(DISPLAY_CLIP_MIN, value)), 1)


def _filter_fields(user: str):
    """滤波参数输入框的 4 个输出值：回填该作业已存的统一参数；无则不动（gr.skip）。

    回填是「一个作业内参数统一」的可见保证：换道集/换作业时字段显示的就是本作业
    在用的那一套，不用重填。未保存过参数时保持用户当前输入不动。
    """
    p = _job_filter_params(wstate(user))
    if not p:
        return (gr.skip(),) * len(FILTER_KEYS)
    return tuple(p[k] for k in FILTER_KEYS)


def _radio_value(job, gid: str, feat, st: dict):
    rec = job.store.get(gid)
    by_key = (rec.get("labels") or {}) if rec else {}
    partial = st.get("partial") or {}
    want = partial.get(feat.name) if feat.name in partial else by_key.get(feat.key)
    if feat.name == "集合类型":
        # The collection branch is a property of the job, not an annotation
        # choice.  Always override stale records/session state so the normal
        # UI cannot display or submit the opposite branch.
        want = "残差" if job.job_type == "residual" else "炮集"
    if feat.input == "checkbox":
        values = want if isinstance(want, list) else ([want] if want else [])
        return [fmt_option(opt) for opt in feat.options if opt.label in values]
    for opt in feat.options:
        if opt.label == want:
            return fmt_option(opt)
    return None


def _radio_updates(job, gid: str, st: dict) -> list:
    return [gr.update(value=_radio_value(job, gid, f, st)) for f in CFG.features]


def _selection_from_values(values) -> dict:
    """Convert Gradio display values to configured labels and enforce exclusive choices."""
    sel = {}
    for feat, value in zip(CFG.features, values):
        if not value:
            continue
        if feat.input == "checkbox":
            shown = value if isinstance(value, list) else [value]
            labels = [label_of(v) for v in shown]
            exclusive = [o.label for o in feat.options if o.exclusive]
            picked_exclusive = [x for x in labels if x in exclusive]
            sel[feat.name] = picked_exclusive[-1:] if picked_exclusive else labels
        else:
            sel[feat.name] = label_of(value)
    # Inactive branch values may remain in hidden browser controls; never persist them.
    active_names = {f.name for f in CFG.active_features(sel)}
    return {name: value for name, value in sel.items() if name in active_names}


def _selection(job, gid: str, st: dict) -> dict:
    sel = {}
    for f in CFG.features:
        v = _radio_value(job, gid, f, st)
        if v:
            sel[f.name] = [label_of(x) for x in v] if isinstance(v, list) else label_of(v)
    if "集合类型" not in sel:
        sel["集合类型"] = "残差" if job.job_type == "residual" else "炮集"
    else:
        sel["集合类型"] = "残差" if job.job_type == "residual" else "炮集"
    return sel


def _current_boxes(job, gid: str, st: dict) -> dict:
    """当前道集的框 {feature_key: box|None}：本会话暂存优先，其次已存记录。"""
    boxes = st.get("boxes")
    if boxes is not None:
        return dict(boxes)
    rec = job.store.get(gid)
    if rec:
        return dict(rec.get("regions") or {})
    return {}


def _box_list(value) -> list[dict]:
    """把单个区域或异常振幅的多区域统一展开，兼容旧记录格式。"""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return [value] if isinstance(value, dict) else []


def render_display(request: gr.Request) -> str:
    """当前道集显示图：显示图（.cache，干净）+ 已画框叠加临时副本。

    只显示/领取时绝不写导出 images/：导出图仅在「保存标注」时由 JobManager.save
    生成，否则会出现标一张却多一张下一道集孤立图、图片与 labels.jsonl 对不上的问题。
    """
    user = _user(request)
    st = wstate(user)
    job = JM.get(st["job_id"])
    gid = st["gid"]
    # 面波区「滤波」：状态与 gid 绑定，换道集自动回落到干净图（web_core.active_filter）
    layer = max(0, min(job.layer_count - 1, int(st.get("residual_layer", 0) or 0)))
    st["residual_layer"] = layer
    base = job.display_image(gid, bandpass=active_filter(st, gid), layer=layer,
                             clip_percentile=_display_clip(job, st))
    overlays = []
    boxes = _current_boxes(job, gid, st)
    sel = _selection(job, gid, st)
    for feat in active_bbox_feats(sel):
        for b in _box_list(boxes.get(feat.key)):
            points = b.get("points")
            if not points and b.get("xyxy"):
                x0, y0, x1, y1 = b["xyxy"]
                points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            if points:
                overlays.append((points, feat.bbox_color, feat.name))
    if not overlays:
        return base
    tmp = os.path.join(tempfile.mkdtemp(prefix="gather_"), "cur_boxed.png")
    return overlay_polygons(base, overlays, tmp)


def box_statuses_of(request: gr.Request) -> list[str]:
    user = _user(request)
    st = wstate(user)
    job = JM.get(st["job_id"])
    gid = st["gid"]
    boxes = _current_boxes(job, gid, st)
    sel = _selection(job, gid, st)
    active = active_bbox_feats(sel)
    target = box_target(active, sel, boxes)
    outs = []
    for feat in bbox_feats():
        if feat not in active:
            outs.append(f"{feat.name}：当前集合类型无需包络")
            continue
        raw = boxes.get(feat.key)
        if sel.get(feat.name) == ABSENT_LABEL:
            # 选了「不存在」→ 这一步直接跳过（前端同样会跳过，这里给一句明确说明）
            detail = f"已选「{ABSENT_LABEL}」，无需画框（本步自动跳过）"
        elif raw:
            regions = _box_list(raw)
            details = []
            for b in regions:
                (x0, y0, x1, y1), (t0, t1), (s0, s1) = b["xyxy"], b["traces"], b["samples"]
                n_points = len(b.get("points") or [])
                details.append(f"[{x0},{y0},{x1},{y1}] 道 {t0} 至 {t1}，采样 {s0} 至 {s1}")
            detail = f"✔ 已画 {len(regions)} 个包络：" + "；".join(details)
        else:
            detail = "未画包络"
        if feat.key == target and sel.get(feat.name) != ABSENT_LABEL:
            detail += "｜👉 当前 Ctrl+左键目标"
        outs.append(f"{feat.name}：{detail}")
    return outs


def _my_total_text(user: str) -> str:
    """标注区常驻的「我的累计标注」文案（跨全部作业，不区分作业）。"""
    return f"**我的累计标注：{JM.user_count(user)} 张**（跨全部作业）"


def _staff_rows():
    """管理员用：[(用户名, 角色, 标注张数)] —— 列出 users.yaml 里的全部账号（含 0 张的）。"""
    counts = JM.user_counts()
    return [(name, "管理员" if ACC.role(name) == "admin" else "标注者",
             counts.get(name, 0))
            for name in ACC.names()]


def _progress_text(job, st: dict) -> str:
    """进度文案：本作业 + 整个作业池汇总（多选多个作业时才有池汇总）。"""
    n_lab, n_tot = JM.progress(job.job_id)
    txt = f"本作业 {n_lab}/{n_tot}"
    job_ids = _norm_job_ids(st.get("job_ids"))
    if len(job_ids) > 1:
        p_lab, p_tot, n_job = JM.pool_progress(job_ids)
        txt += f" ｜ 作业池 {p_lab}/{p_tot}（共 {n_job} 个作业）"
    return txt


def show_current(request: gr.Request):
    """返回当前道集的所有显示组件值（img, info, sentence, *radios, *box_statuses）。"""
    user = _user(request)
    st = wstate(user)
    job = JM.get(st["job_id"])
    gid = st["gid"]
    img = render_display(request)
    meta = job.gather_meta(gid)
    rec = job.store.get(gid)
    state = "已标注 ✔" if rec else "未标注"
    layer = max(0, min(job.layer_count - 1, int(st.get("residual_layer", 0) or 0)))
    st["residual_layer"] = layer
    layer_text = f" | 显示：{job.layer_names[layer]} ({layer + 1}/{job.layer_count})"
    info = (f"**道集 {gid}** | 键 {meta['key']} = {meta['value_text']}"
            f" | {meta['n_traces']} 道 | 作业类型：{('残差' if job.job_type == 'residual' else '炮集')}"
            f"{layer_text} | {state} | {_progress_text(job, st)}")
    sel = _selection(job, gid, st)
    sentence = CFG.render_sentence(sel)
    return (img, info, sentence, *_radio_updates(job, gid, st), *box_statuses_of(request),
            *_filter_fields(user), _display_clip(job, st), _my_total_text(user))


def _anno_idle(msg: str = ""):
    """无当前道集时的标注区空态输出。

    **图片要清掉**（而不是 gr.skip 留着旧图）：池子标完 / 未领取时若把上一张留在屏幕上，
    标注者会以为"这张已完成的任务又发给我了"（实测反馈过的误解）。
    radio 与参数字段保持不动（与旧图一起清掉反而会让用户以为选错了张）。
    """
    return (gr.update(value=None), msg, "", *([gr.update(value=None)] * len(CFG.features)),
            *(["…"] * len(bbox_feats())), *((gr.skip(),) * len(FILTER_KEYS)),
            gr.skip(), gr.skip())


def refresh_anno(request: gr.Request):
    user = _user(request)
    st = wstate(user)
    if not st.get("job_id") or not st.get("gid"):
        return _anno_idle("请选择作业（选中即自动领取第一张）")
    return show_current(request)


def cycle_residual_layer(request: gr.Request, direction: int = 1):
    """Move through residual preview layers in either direction."""
    user = _user(request)
    st = wstate(user)
    if not st.get("job_id") or not st.get("gid"):
        return _anno_idle("⚠️ 请先领取道集")
    job = JM.get(st["job_id"])
    if job.job_type != "residual":
        return show_current(request)
    JM.renew(job.job_id, user)
    step = 1 if int(direction) >= 0 else -1
    st["residual_layer"] = (int(st.get("residual_layer", 0) or 0) + step) % job.layer_count
    return show_current(request)


def cycle_residual_previous(request: gr.Request):
    return cycle_residual_layer(request, -1)


def cycle_residual_next(request: gr.Request):
    return cycle_residual_layer(request, 1)


def update_display_clip(request: gr.Request, value):
    """调整当前作业的显示 clip，并立即重绘当前道集。"""
    user = _user(request)
    st = wstate(user)
    if not st.get("job_id") or not st.get("gid"):
        return gr.skip()
    job = JM.get(st["job_id"])
    try:
        clip = round(float(value), 1)
    except (TypeError, ValueError):
        clip = _display_clip(job, st)
    clip = min(DISPLAY_CLIP_MAX, max(DISPLAY_CLIP_MIN, clip))
    # 显示参数属于工区/作业：下一位标注者领取该作业内道集时也沿用它。
    job.display_clip = clip
    JM.renew(job.job_id, user)
    return render_display(request)


# ----------------------------------------------------------------------
# 建作业（admin）
# ----------------------------------------------------------------------
def load_file(request: gr.Request, path: str, endian_text: str):
    if not _is_admin(_user(request)):
        return "❌ 仅管理员可读取文件信息"
    path = (path or "").strip()
    if not path:
        return "⚠️ 请填写文件路径"
    endian = {"自动检测": "auto", "大端": "big", "小端": "little"}[endian_text]
    try:
        reader = SegyReader(path, endian=endian)
    except SegyReadError as e:
        return f"❌ 读取失败: {e}"
    i = reader.info()
    reader.close()
    return (f"✔ 已加载 | 端序: {'小端' if i['endian'] == 'little' else '大端'}"
            f" | 道数: {i['n_traces']} | 每道采样点: {i['ns']}"
            f" | 采样间隔: {i['dt_ms']} ms | 格式码: {i['format_code']}"
            f" | 大小: {i['file_size_mb']} MB")


def load_job_source(request: gr.Request, job_type, path, before_path, endian_text):
    """Read the shot source or the residual job's first (before) source."""
    source = before_path if str(job_type) == "残差" else path
    return load_file(request, source, endian_text)


def scan_values(request: gr.Request, path: str, endian_text: str, gkey_text: str):
    if not _is_admin(_user(request)):
        return gr.CheckboxGroup(choices=[], value=[]), "❌ 仅管理员可扫描键值"
    path = (path or "").strip()
    if not path:
        return gr.CheckboxGroup(choices=[], value=[]), "⚠️ 请先填写文件路径"
    endian = {"自动检测": "auto", "大端": "big", "小端": "little"}[endian_text]
    try:
        gkeys = parse_ranges(gkey_text)
        if not gkeys:
            return gr.CheckboxGroup(choices=[], value=[]), "❌ 请填写抽道集键"
        reader = SegyReader(path, endian=endian)
        vals = gather_values(reader, gkeys)
        reader.close()
    except (SegyReadError, ValueError) as e:
        return gr.CheckboxGroup(choices=[], value=[]), f"❌ {e}"
    multi = len(gkeys) > 1
    if multi:
        strs = [",".join(str(int(x)) for x in t) for t in vals]
    else:
        strs = [str(int(v)) for v in vals]
    hint = f"共 {len(strs)} 个键值（默认全选，可取消勾选）"
    if multi:
        hint += f"｜元组按抽道集键顺序 {ranges_label(gkeys)} 对应"
    return gr.CheckboxGroup(choices=strs, value=strs), hint


def scan_job_values(request: gr.Request, job_type, path, before_path,
                    endian_text, gkey_text):
    source = before_path if str(job_type) == "残差" else path
    return scan_values(request, source, endian_text, gkey_text)


def _parse_job_values(sort_text: str, gkey_text: str, selected: list[str]):
    """返回 (sort_keys, extract_keys, values)，解析失败抛 ValueError。"""
    sort_keys = parse_ranges(sort_text)
    gkeys = parse_ranges(gkey_text)
    if not gkeys:
        raise ValueError("请填写抽道集键")
    multi = len(gkeys) > 1
    if multi:
        try:
            values = [tuple(int(x) for x in s.split(",")) for s in selected]
        except ValueError:
            raise ValueError("键值格式错误（多字段键值应如 “1373,42”）")
    else:
        try:
            values = [int(v) for v in selected]
        except ValueError:
            raise ValueError("键值格式错误")
    return sort_keys, gkeys, values


def create_job_ui(request: gr.Request, path, endian_text, sort_text, gkey_text,
                  selected, clip, aug_lo, aug_hi, title, min_traces, decimate_n,
                  job_type="炮集", before_path="", after_path="", residual_path=""):
    """创建作业：抽一次道集（按道数下限滤去过少者、按抽稀间隔等间隔删）、写 job.json、注册。

    clip: 预览 clip（仅界面显示/画框参照）；aug_lo/aug_hi: 保存时随机采样的 clip 范围。
    min_traces/decimate_n: 先按道数下限过滤、再按抽稀间隔抽稀（见 JobManager.create_job）。
    """
    user = _user(request)
    if not _is_admin(user):
        return "❌ 仅管理员可创建作业", gr.update(), gr.update()
    job_type = "residual" if str(job_type or "炮集") in ("残差", "residual") else "shot"
    if job_type == "residual":
        source_files = [(before_path or "").strip(), (after_path or "").strip(),
                        (residual_path or "").strip()]
        if not all(source_files):
            return "⚠️ 残差作业需要填写去噪前、去噪后、噪声残差三个 SGY 文件路径", gr.update(), gr.update()
        path = source_files[0]
    else:
        source_files = [(path or "").strip()]
        path = source_files[0]
    if not path:
        return "⚠️ 请填写文件路径", gr.update(), gr.update()
    endian = {"自动检测": "auto", "大端": "big", "小端": "little"}[endian_text]
    if not selected:
        return "⚠️ 请先扫描并勾选键值", gr.update(), gr.update()
    try:
        sort_keys, gkeys, values = _parse_job_values(sort_text, gkey_text, selected)
    except ValueError as e:
        return f"❌ {e}", gr.update(), gr.update()
    title = (title or "").strip() or os.path.splitext(os.path.basename(path))[0]
    try:
        min_traces = int(min_traces) if min_traces else 0
    except (TypeError, ValueError):
        min_traces = 0
    try:
        decimate_n = int(decimate_n) if decimate_n else 1
    except (TypeError, ValueError):
        decimate_n = 1
    if decimate_n < 0:
        decimate_n = 1
    try:
        clip = float(clip if clip is not None else 99.0)
        aug_lo = float(aug_lo if aug_lo is not None else 90.0)
        aug_hi = float(aug_hi if aug_hi is not None else 99.9)
    except (TypeError, ValueError):
        return "❌ clip 参数须为数字", gr.update(), gr.update()
    if not (0 < aug_lo <= aug_hi <= 100):
        return "❌ 增强 clip 上下限须满足 0 < 下限 ≤ 上限 ≤ 100", gr.update(), gr.update()
    # 0.1 步长至少要有 N_AUG 个可选值，否则保存时采不出 N_AUG 个不同 clip
    if int(round(aug_hi * 10)) - int(round(aug_lo * 10)) + 1 < N_AUG:
        return (f"❌ 增强 clip 范围过窄：上下限差至少需 0.{N_AUG - 1} "
                f"（0.1 步长才能采出 {N_AUG} 个不同 clip 值）", gr.update(), gr.update())
    try:
        job = JM.create_job(path, sort_keys, gkeys, values,
                            clip, title, user, endian=endian, min_traces=min_traces,
                            aug_lo=aug_lo, aug_hi=aug_hi, decimate_n=decimate_n,
                            job_type=job_type, source_files=source_files)
    except (JobError, SegyReadError) as e:
        return f"❌ {e}", gr.update(), gr.update()
    n = len(job.gather_ids())
    # 两个筛选叠加后无法把「少了多少」归给某一个，故只报实际生效的设置
    applied = []
    if min_traces > 0:
        applied.append(f"道数下限 {min_traces}")
    if decimate_n > 1:
        applied.append(f"抽稀间隔 {decimate_n}（每 {decimate_n} 个留 1 个）")
    note = f"（先滤后抽：{'、'.join(applied)}）" if applied else ""
    job_choices = _job_choices(user)
    mgr_choices = [j["job_id"] for j in JM.list_jobs()]
    mode_text = "残差（去噪前/去噪后/噪声残差）" if job_type == "residual" else "炮集"
    return (f"✔ 已创建{mode_text}作业 {job.job_id}（键 {ranges_label(gkeys)}）｜道集数 {n}{note}"
            f"｜保存时每张增强 {N_AUG} 个随机 clip",
            gr.update(choices=job_choices, value=[job.job_id]),
            gr.update(choices=mgr_choices, value=job.job_id))


# ----------------------------------------------------------------------
# 标注流程 handler
# ----------------------------------------------------------------------
def select_jobs(request: gr.Request, job_ids):
    """选中作业（可多选）即**自动从作业池随机领一张**（不必再点「领取下一张」）。

    - 选择没变、且手上还压着一张没保存的：原样不动，避免白白丢掉未保存的选择/框。
    - 选择变了：清掉上一张的暂存态再领（未保存的内容会丢，这是换选择的既有语义）。
    - 领不到（已关闭/已标完/全被占）时直接把原因显示出来，比让用户去点按钮再看到更清楚。
    """
    user = _user(request)
    st = wstate(user)
    job_ids = _norm_job_ids(job_ids)
    mine = gr.update(choices=_mine_choices(user, job_ids), value=None)
    if not job_ids:
        return (*_anno_idle("⚠️ 请选择作业（可多选）"), mine)
    if st.get("job_ids") == job_ids and st.get("gid"):
        return (*refresh_anno(request), mine)          # 选择没变 → 保持现状
    st["job_ids"] = job_ids
    # hist 一并清掉：换作业选择是**显式换了上下文**，「返回上一张」不该退回上一套选择的道集
    for k in ("gid", "job_id", "partial", "boxes", "filter", "hist"):
        st.pop(k, None)
    anno, _note = _claim_from_pool(request)
    return (*anno, mine)


def _claim_from_pool(request: gr.Request):
    """从当前选中的作业池随机领一张并显示。

    返回 (anno_outputs 形状的输出, 一句状态说明)：说明用于拼在保存/跳过的提示里，
    免得把整行道集信息塞进那些消息。
    """
    user = _user(request)
    st = wstate(user)
    out = JM.claim_pool(st.get("job_ids"), user)
    if out.get("gid"):
        _visit(st, out["job_id"], out["gid"])   # job_id = 当前道集所属作业（保存/进度都按它走）
        st["partial"], st["boxes"] = {}, {}
        st["residual_layer"] = 0
        n_job = len(_norm_job_ids(st.get("job_ids")))
        note = (f"已自动从作业池领取下一张「{out['gid']}」" if n_job > 1
                else f"已自动领取下一张「{out['gid']}」")
        return show_current(request), note
    n_lab, n_tot, n_job = JM.pool_progress(st.get("job_ids"))
    if n_tot and n_lab >= n_tot:
        return (_anno_idle(f"🎉 选中的 {n_job} 个作业已全部标注（{n_lab}/{n_tot}）"),
                f"选中的 {n_job} 个作业已全部标注（{n_lab}/{n_tot}）")
    reason = out.get("reason") or "暂无剩余可领取"
    return _anno_idle("⚠️ " + reason), "⚠️ " + reason


def claim_next(request: gr.Request, job_ids):
    """「领取下一张」：从选中的作业池里再随机抽一张。"""
    user = _user(request)
    st = wstate(user)
    job_ids = _norm_job_ids(job_ids)
    if not job_ids:
        return _anno_idle("⚠️ 请先在「作业」里选择至少一个作业")
    st["job_ids"] = job_ids
    # 已持有未保存的任务：不静默换张，明确指引如何进入下一张
    if st.get("gid") and JM.record(st.get("job_id") or "", st["gid"]) is None:
        cur = show_current(request)
        return (cur[0],
                f"⚠️ 正在标注「{st['gid']}」（尚未保存）。要进入下一张，请先点"
                "「保存并释放」完成本张；要放弃本张，请先点「归还此张」。",
                cur[2], *cur[3:])
    anno, _note = _claim_from_pool(request)
    return anno


def on_radio_change(request: gr.Request, *radio_values):
    """改动选项 → 更新句子预览，并刷新框状态（改选「不存在」会让 Ctrl 目标顺延）。"""
    user = _user(request)
    st = wstate(user)
    if not st.get("gid") or not st.get("job_id"):
        return ("", *(["…"] * len(bbox_feats())))
    JM.renew(st["job_id"], user)
    sel = _selection_from_values(radio_values)
    st["partial"] = sel
    return (CFG.render_sentence(sel), *box_statuses_of(request))


def clear_box(request: gr.Request, feat_key: str):
    user = _user(request)
    st = wstate(user)
    if not st.get("gid") or not st.get("job_id"):
        return (gr.skip(), "⚠️ 请先领取道集", *(["…"] * len(bbox_feats())))
    JM.renew(st["job_id"], user)
    job = JM.get(st["job_id"])
    boxes = _current_boxes(job, st["gid"], st)
    boxes[feat_key] = None
    st["boxes"] = boxes
    feat = next(f for f in bbox_feats() if f.key == feat_key)
    return (render_display(request), f"已清除「{feat.name}」的框", *box_statuses_of(request))


# ----------------------------------------------------------------------
# 面波区「滤波」：只改「显示图」，导出图仍由原始数据渲染
# ----------------------------------------------------------------------
def _filter_outputs(request: gr.Request, msg: str, fresh_img: bool):
    """滤波开关的统一输出，**与 anno_outputs 严格同序同长**：
        (img, info, sentence, *radios, *box_statuses, *4 个滤波参数, clip, 累计)

    滤波只改「图 / 提示 / 参数字段」三处，句子、单选、框状态一律 gr.skip() 保持不动
    （滤波不影响选择）；参数字段回填该作业已存的统一参数（_filter_fields）。
    无当前道集时全部 gr.skip()，只回提示 —— 不碰 st['job_id']，避免 KeyError。
    """
    user = _user(request)
    st = wstate(user)
    n_other = 1 + len(CFG.features) + len(bbox_feats())   # sentence + radios + 框状态
    if not st.get("job_id") or not st.get("gid"):
        return (gr.skip(), msg, *((gr.skip(),) * (n_other + len(FILTER_KEYS))),
                gr.skip(), gr.skip())
    img = render_display(request) if fresh_img else gr.skip()
    return (img, msg, *((gr.skip(),) * n_other), *_filter_fields(user),
            gr.skip(), gr.skip())


def toggle_filter(request: gr.Request, f1, f2, f3, f4):
    """「⚙ 滤波」单键开关：点一次应用，再点一次还原（只影响显示）。

    参数按 (标注者, 作业) 记忆在 st['filter_params'][job_id]：
      - 该作业第一次点，用当前四个输入框的值，并记住；
      - 同作业之后每次点，仍用这套参数（字段同时被回填成它），做到「一个作业内统一」。
    应用状态与 gid 绑定 → 换到下一张必须重新点一次，**不会默认应用**。
    """
    user = _user(request)
    st = wstate(user)
    if not st.get("gid") or not st.get("job_id"):
        return _filter_outputs(request, "⚠️ 请先领取道集", fresh_img=False)
    job_id, gid = st["job_id"], st["gid"]
    if active_filter(st, gid) is not None:            # 当前是「滤波中」→ 还原
        JM.renew(job_id, user)
        st["filter"] = None
        return _filter_outputs(request, "已还原为原始数据（未滤波）", fresh_img=True)
    params, err = parse_filter(f1, f2, f3, f4)        # 当前是「原始」→ 应用
    if err:
        return _filter_outputs(request, f"⚠️ {err}", fresh_img=False)
    job = JM.get(job_id)
    if job.dt_ms <= 0:
        return _filter_outputs(
            request, "⚠️ 该作业缺采样间隔（dt_ms），无法滤波；请用新版重建作业",
            fresh_img=False)
    JM.renew(job_id, user)
    st.setdefault("filter_params", {})[job_id] = params
    st["filter"] = {"gid": gid, **params}
    return _filter_outputs(
        request,
        f"✔ 已应用滤波 f1={params['f1']:g} / f2={params['f2']:g} / "
        f"f3={params['f3']:g} / f4={params['f4']:g} Hz（本作业后续沿用这套参数；"
        f"只影响显示，导出图仍为原始数据）",
        fresh_img=True)


def apply_drag_polygon(user: str, key: str, points) -> tuple[bool, str]:
    """Store a Ctrl-click polygon envelope in natural 512x1024 image pixels."""
    if key not in [f.key for f in bbox_feats()]:
        return False, f"非框选特征: {key}"
    with WORK_LOCK:
        st = WORK.setdefault(user, {})
        if not st.get("job_id") or not st.get("gid"):
            return False, "没有正在标注的道集"
        job = JM.get(st["job_id"])
        g = job.gather(st["gid"])
    box = pixel_polygon_to_data(points, g.n_traces, job.ns)
    with WORK_LOCK:
        st["boxes"] = dict(st.get("boxes") or {})
        st["boxes"][key] = box
    return True, key


def apply_drag_regions(user: str, key: str, regions) -> tuple[bool, str]:
    """Store multiple polygon/rectangle regions for one feature."""
    if key not in [f.key for f in bbox_feats()]:
        return False, f"非框选特征: {key}"
    if not isinstance(regions, list) or not regions:
        return False, "至少需要一个包络区域"
    with WORK_LOCK:
        st = WORK.setdefault(user, {})
        if not st.get("job_id") or not st.get("gid"):
            return False, "没有正在标注的道集"
        job = JM.get(st["job_id"])
        g = job.gather(st["gid"])
    boxes = [pixel_polygon_to_data(points, g.n_traces, job.ns) for points in regions]
    with WORK_LOCK:
        st["boxes"] = dict(st.get("boxes") or {})
        st["boxes"][key] = boxes
    return True, key


def apply_drag_box(user: str, key: str, x0, y0, x1, y1) -> tuple[bool, str]:
    """Backward-compatible rectangle endpoint used by older clients/tests."""
    return apply_drag_polygon(user, key,
                              [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def clear_drag_box(user: str, key: str) -> tuple[bool, str]:
    """清掉该用户当前道集里某特征的框（右键清框用）。

    与「✕ 清除此框」按钮**同一服务端语义**（把 st['boxes'][key] 置 None、保存时按 None 处理），
    但走自定义 HTTP 接口而不是程序化点那个按钮：右键的行为不该依赖 Gradio 按钮在 DOM 里
    的层级/结构（elem_id 落在 <button> 还是外层 div 上，各版本不一样）。
    """
    if key not in [f.key for f in bbox_feats()]:
        return False, f"非框选特征: {key}"
    with WORK_LOCK:
        st = WORK.setdefault(user, {})
        if not st.get("job_id") or not st.get("gid"):
            return False, "没有正在标注的道集"
        st["boxes"] = dict(st.get("boxes") or {})
        st["boxes"][key] = None
    return True, key


def _boxes_payload(user: str) -> dict:
    """当前道集各 bbox 特征的包络 + Ctrl 多点标注的当前目标特征。

    返回 {"boxes": {key: {x0,y0,x1,y1}}, "target": key|None}：
      - boxes：自然像素 points（与导出图 512×1024 同系），并带 xyxy 范围供兼容显示；
        优先读本会话 st['boxes']，重开已保存记录时回退到 labels 记录 regions。
      - target：由 web_core.box_target 按「第一个需框未框」规则算出（Python 侧单一实现，
        可单测；前端不必去猜 DOM 里的选项状态），前端据此决定 Ctrl 左键画给谁。
    """
    st = wstate(user)
    boxes_out, target, note = {}, None, ""
    if st.get("job_id") and st.get("gid"):
        try:
            job = JM.get(st["job_id"])
        except JobError:
            return {"boxes": boxes_out, "target": None, "note": ""}
        boxes = _current_boxes(job, st["gid"], st)
        for f in bbox_feats():
            raw = boxes.get(f.key)
            regions = []
            for b in _box_list(raw):
                if not b.get("xyxy"):
                    continue
                points = b.get("points")
                if not points:
                    x0, y0, x1, y1 = b["xyxy"]
                    points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                regions.append({"points": [[int(x), int(y)] for x, y in points]})
            if regions:
                boxes_out[f.key] = regions if len(regions) > 1 else regions[0]
        sel = _selection(job, st["gid"], st)
        active = active_bbox_feats(sel)
        target = box_target(active, sel, boxes)
        note = box_note(active, sel, target)
    # note 只在「目标为空」时有值：给前端一句准确说明（如「所有拉框项均不存在，无需画框」）；
    # 没有当前道集时为空串 → 前端**不画任何提示**（否则旧图上会浮出误导性的"已完成"）
    return {"boxes": boxes_out, "target": target, "note": note}


def save_anno(request: gr.Request, *radio_values):
    user = _user(request)
    st = wstate(user)
    if not st.get("gid") or not st.get("job_id"):
        return _anno_idle("⚠️ 没有正在标注的道集")
    job_id = st["job_id"]
    gid = st["gid"]
    sel = _selection_from_values(radio_values)
    st["partial"] = sel
    # 保存前这张是否已有记录 → 区分「首次标注」与「修正已有结果」。两者收尾不同：
    # 首次标注保存后自动领下一张；修正保存原地停住（见下方 was_labeled 分支）。
    try:
        was_labeled = JM.record(job_id, gid) is not None
    except JobError:
        was_labeled = False
    # 空闲超 TTL 导致租约被清：renew 只续「仍有效」的租约，返回 None。
    # 区分两种无租约场景：
    #   (a) 该道集尚未标注（首次领取后闲挂超时被清）→ 尝试重新领取同 gid；
    #   (b) 该道集已有记录（重开自己标过的记录，本就无租约）→ 直接走 owner 归属保存。
    if JM.renew(job_id, user) is None and JM.record(job_id, gid) is None:
        # 这里**故意只在本作业里**重领（不要改成 claim_pool）：目的是把要保存的那张
        # 重新纳回自己的租约，换到别的作业毫无意义；claim 没发回同一个 gid 就说明它已被占。
        c = JM.claim(job_id, user)
        if c.get("gid") != gid:
            cur = show_current(request)
            return (cur[0], "⚠️ 该道集已被他人领取或标注（空闲超时已释放），请重新领取",
                    cur[2], *cur[3:])
        st["gid"] = gid   # claim 已把同 gid 重新发回（保持 state 一致）
    # 框来源：st["boxes"] 有暂存则用之，否则回退到已存记录 regions（重开编辑场景）
    ok, rec, aug_recs, err = JM.save(job_id, user, gid,
                                     sel, _current_boxes(JM.get(job_id), gid, st),
                                     is_admin=_is_admin(user))
    if not ok:
        cur = show_current(request)
        return (cur[0], "⚠️ " + err, cur[2], *cur[3:])
    # 云上传由图片后台渲染完成后的钩子统一触发（JM.set_aug_hook）：图片画完后再上传；
    # labels.jsonl 与 NPY 已同步落盘。
    n_labeled, total = JM.progress(job_id)
    st.pop("partial", None)
    st.pop("boxes", None)
    st.pop("filter", None)
    # 保存并释放 → 从**作业池**自动领下一张（多作业下就是在池子里继续随机抽）。
    # 修正已有结果也走这里：按钮就叫「保存并释放」，只覆盖、不前进会让人以为它卡住了
    # （实测反馈过）。「覆盖」这件事由下面的提示语讲清楚，不靠"停住"来表达。
    _push_hist(st, gid)   # 刚处理完这张进历史：保存后「↩ 返回上一张」要能回到它
    st.pop("gid", None)
    cur, note = _claim_from_pool(request)
    done = "已覆盖" if was_labeled else "已保存"
    if JM.get(job_id).job_type == "residual":
        asset_note = f" × {len(aug_recs)} 张增强图（3 个视图各 5 张，3 个 NPY 已保存；图片后台生成）"
    else:
        asset_note = f" × {len(aug_recs)} 张增强图（后台生成中）"
    return (cur[0],
            f"✔ {done}「{rec['gather_id']}」"
            + ("的原结果" if was_labeled else "")
            + asset_note + "，可继续下一张 | "
            f"本作业 {n_labeled}/{total} | {note}",
            cur[2], *cur[3:])


def release_current(request: gr.Request):
    user = _user(request)
    st = wstate(user)
    if not st.get("job_id"):
        return _anno_idle("⚠️ 请先选择作业")
    JM.release(st["job_id"], user)
    st.pop("gid", None)
    st.pop("partial", None)
    st.pop("boxes", None)
    return (gr.update(value=None), "已归还当前道集（回池，可被他人领取）", "",
            *([gr.update(value=None)] * len(CFG.features)), *(["…"] * len(bbox_feats())),
            *((gr.skip(),) * len(FILTER_KEYS)), gr.skip(), _my_total_text(user))


def skip_current(request: gr.Request):
    """「跳过此张」：当前道集放回池（不写记录、不计完成），再从作业池随机领下一张。

    注意是「先只放回、再从**整个池子**抽」—— 若沿用单作业的 skip（放回后只在原作业里抽），
    多选作业时就抽不到别的作业了。
    """
    user = _user(request)
    st = wstate(user)
    if not st.get("job_id") or not st.get("gid"):
        return _anno_idle("⚠️ 没有正在标注的道集")
    skipped = JM.skip_release(st["job_id"], user)
    for k in ("gid", "job_id", "partial", "boxes", "filter"):
        st.pop(k, None)
    cur, note = _claim_from_pool(request)
    return (cur[0],
            f"⏭ 已跳过「{skipped or '当前道集'}」回池（未保存）｜{note}",
            cur[2], *cur[3:])


def refresh_jobs_progress():
    """管理区定时刷新：返回(进度表HTML, 删除按钮, 删除提示)。

    顺带清理“删除确认”超时态：约 3 秒未再点则自动复原按钮/提示。
    """
    html = jobs_progress_html(JM.list_jobs())
    staff = staff_counts_html(_staff_rows())          # 各用户标注总量表（仅管理员可见）
    if _PENDING_DELETE["job_id"] and time.time() - _PENDING_DELETE["at"] > 3.0:
        _cancel_delete()
        return (html, gr.update(value="🗑 删除作业"), gr.update(value=""), staff)
    return (html, gr.update(), gr.update(), staff)


def reopen_mine(request: gr.Request, job_ids, gid: str):
    user = _user(request)
    st = wstate(user)
    job_ids = _norm_job_ids(job_ids)
    if not job_ids or not gid:
        return _anno_idle("⚠️ 请选择要重开的记录")
    # 该 gid 属于选中作业里的哪一个（同文件建的两个作业会有相同 gid → 取第一个）
    job_id = _job_of_gid(user, gid, job_ids)
    if job_id is None:
        # 没标过就不是「我标注的」；admin 允许直接按当前选中的第一个作业打开
        if not _is_admin(user):
            return _anno_idle("⚠️ 只能重开自己标注过的记录")
        job_id = job_ids[0]
    # 服务端校验：标注者只能重开自己标过的记录；admin 可重开任意记录
    if not _is_admin(user):
        try:
            mine_ids = {r["gather_id"] for r in JM.mine(job_id, user)}
        except JobError:
            mine_ids = set()
        if gid not in mine_ids:
            return _anno_idle("⚠️ 只能重开自己标注过的记录")
    st["job_ids"] = job_ids
    _visit(st, job_id, gid)   # 原来那张进浏览历史：重开别的记录后还能「↩ 返回上一张」
    st.pop("partial", None)   # 回退到已存记录标签
    st.pop("boxes", None)     # 回退到已存记录框
    JM.renew(job_id, user)
    return show_current(request)


def back_previous(request: gr.Request):
    """「↩ 返回上一张」：退回上一个访问过的道集，改完点保存即覆盖原结果。

    历史取 st['hist']（浏览栈），不是「我标注的」列表顺序 —— 那张表按**首次标注时间**
    排序，修正保存不会把记录挪到末尾，靠它会翻到别处（详见 web_core.prev_gid）。
    到最早一张仍点 → 明确提示，不静默原地不动。
    """
    user = _user(request)
    st = wstate(user)
    if not st.get("job_id"):
        return _anno_idle("⚠️ 请先选择作业")
    # 手上这张还没保存。最常见的场景恰恰是：刚保存完一张、程序自动领到下一张就想回去改
    # —— 这时它一个字都没填，直接**归还回池**再往回走，否则按钮在最常用的场景里等于摆设。
    # 已经作答过的（选项或框）才拦下来问一句，避免把真实工作静默丢掉。
    released = None
    if st.get("gid") and JM.record(st.get("job_id") or "", st["gid"]) is None:
        if _has_unsaved_input(st):
            cur = show_current(request)
            return (cur[0],
                    f"⚠️ 正在标注「{st['gid']}」且已作答（尚未保存）。要返回上一张，请先点"
                    "「保存并释放」完成本张；要放弃本张，请先点「归还此张」。",
                    cur[2], *cur[3:])
        released = st["gid"]
        JM.release(st["job_id"], user)
        st.pop("gid", None)
    target, hist = prev_gid(st.get("hist"), st.get("gid"))
    if not target:
        cur = show_current(request)
        return (cur[0], "↩ 已是最早一张，没有更早的记录可回了", cur[2], *cur[3:])
    job_id = _job_of_gid(user, target, _norm_job_ids(st.get("job_ids")))
    if job_id is None:
        # 历史里的道集不属于当前选中的作业（换过作业选择，或已被 admin 删除）
        cur = show_current(request)
        return (cur[0],
                f"⚠️ 上一张「{target}」不在当前作业选择里，请用「我标注的」下拉打开",
                cur[2], *cur[3:])
    st["hist"] = hist          # 目标已出栈；当前这张**不**压回（只回退、不前进）
    st.pop("partial", None)    # 回退到已存记录标签
    st.pop("boxes", None)      # 回退到已存记录框
    st.pop("filter", None)     # 滤波只对原道集生效
    st["job_id"] = job_id
    st["gid"] = target
    JM.renew(job_id, user)
    cur = show_current(request)
    note = f"（「{released}」未作任何作答，已归还回池）" if released else ""
    return (cur[0],
            f"↩ 已返回「{target}」（已标注）{note}。改完点「保存并释放」即覆盖原结果。",
            cur[2], *cur[3:])


def inherit_previous(request: gr.Request):
    """把本用户最近一条【已标注】记录的类别选项填充到当前道集（仅填充不保存，框不继承）。

    输出里必须带上 *_box_statuses：本函数用 gr.Radio(value=…) **程序化**改选项，不产生
    DOM change 事件，前端拿不到任何"该刷新了"的信号；框状态标记正是那个信号（写进
    #anno_featcol 会触发前端的观察者去回读 /api/boxes）。少了它，继承完画布上的 Ctrl
    目标还是继承前的 —— 表现成"两项都继承了「不存在」，却仍能拉出一个框"。
    """
    user = _user(request)
    st = wstate(user)
    if not st.get("gid") or not st.get("job_id"):
        return ("⚠️ 请先领取道集", *([gr.update(value=None)] * len(CFG.features)),
                *(["…"] * len(bbox_feats())))
    src = None
    for rec in reversed(JM.mine(st["job_id"], user)):
        if rec.get("gather_id") != st["gid"] and rec.get("labels"):
            src = rec
            break
    if src is None:
        cur_sel = _selection(JM.get(st["job_id"]), st["gid"], st)
        return ("⚠️ 没有可继承的已标注记录 | " + CFG.render_sentence(cur_sel),
                *([gr.update(value=None)] * len(CFG.features)), *box_statuses_of(request))
    by_key = src["labels"]
    sel = {f.name: by_key[f.key] for f in CFG.features if by_key.get(f.key)}
    outs = []
    for f in CFG.features:
        want = sel.get(f.name)
        if f.input == "checkbox":
            vals = want if isinstance(want, list) else ([want] if want else [])
            outs.append(gr.update(value=[fmt_option(o) for o in f.options if o.label in vals]))
        else:
            outs.append(gr.update(value=next((fmt_option(o) for o in f.options if o.label == want), None)))
    st["partial"] = sel
    return ("已继承最近已标注记录（" + src["gather_id"] + "）的选项；框不继承，需另行画框："
            + CFG.render_sentence(sel), *outs, *box_statuses_of(request))


def refresh_page(request: gr.Request):
    """刷新作业多选 + 我标注的 + 当前显示。"""
    user = _user(request)
    st = wstate(user)
    job_choices = _job_choices(user)
    job_ids = [j for j in _norm_job_ids(st.get("job_ids")) if j in job_choices]
    st["job_ids"] = job_ids                     # 选中里已被删除/关闭的 → 收敛掉
    # 当前道集所属作业已不在选择里（被取消勾选/被删）→ 清掉当前道集
    if st.get("gid") and st.get("job_id") not in job_ids:
        for k in ("gid", "job_id", "partial", "boxes", "filter"):
            st.pop(k, None)
    dd = gr.update(choices=job_choices, value=job_ids)
    mine = gr.update(choices=_mine_choices(user, job_ids), value=None)
    if st.get("gid"):
        anno = show_current(request)
    else:
        anno = _anno_idle("请选择作业（可多选；选中即自动从作业池随机领取一张）")
    return (dd, mine, *anno)


# ----------------------------------------------------------------------
# 管理（admin）
# ----------------------------------------------------------------------
def resync_job(request: gr.Request, job_id: str):
    user = _user(request)
    if not _is_admin(user):
        return "❌ 仅管理员可重传"
    if not job_id:
        return "⚠️ 请选择作业"
    try:
        job = JM.get(job_id)
    except JobError as e:
        return f"❌ {e}"
    UP.resync_all(job_id, job.output_dir)
    return f"✔ 已入队重传 {job_id}（images/*.png + labels.jsonl）"


def _mgmt_choices() -> list[str]:
    return [j["job_id"] for j in JM.list_jobs()]

def _deleted_choices() -> list[str]:
    return [j["job_id"] for j in JM.deleted_jobs()]

def _job_title(job_id: str) -> str:
    try:
        return JM.get(job_id).title or job_id
    except JobError:
        return job_id


# 「删除作业」两次确认的临时态（内存，进程内有效）
_PENDING_DELETE: dict = {"job_id": None, "at": 0.0}


def _cancel_delete():
    _PENDING_DELETE["job_id"] = None
    _PENDING_DELETE["at"] = 0.0


def _del_pack(btn_label: str, info: str, user: str, mgr_value):
    """删除按钮输出组（顺序与 del_outputs 一致）。"""
    return (gr.update(value=btn_label),
            info,
            gr.update(choices=_mgmt_choices(), value=mgr_value),
            gr.update(choices=_deleted_choices(), value=None),
            jobs_progress_html(JM.list_jobs()),
            gr.update(choices=_job_choices(user), value=[]),
            staff_counts_html(_staff_rows()))


def delete_job_click(request: gr.Request, job_id: str):
    """删除作业：首次点击进入确认态，3 秒内再点一次才真正软删除。"""
    user = _user(request)
    if not _is_admin(user):
        return _del_pack("🗑 删除作业", "❌ 仅管理员可删除", user, None)
    job_id = (job_id or "").strip()
    if not job_id:
        return _del_pack("🗑 删除作业", "⚠️ 请先选择要删除的作业", user, None)
    title = _job_title(job_id)
    armed = (_PENDING_DELETE["job_id"] == job_id
             and time.time() - _PENDING_DELETE["at"] <= 3.0)
    if not armed:
        _cancel_delete()
        _PENDING_DELETE["job_id"] = job_id
        _PENDING_DELETE["at"] = time.time()
        return _del_pack("⚠️ 再次点击确认删除",
                         f"⚠️ 再点一次确认删除「{title}」（{job_id}）；"
                         f"约 3 秒未确认或换选其它作业会自动取消。本地结果不会删除。",
                         user, job_id)
    try:
        JM.delete_job(job_id)
    except JobError as e:
        _cancel_delete()
        return _del_pack("🗑 删除作业", f"❌ {e}", user, None)
    _cancel_delete()
    return _del_pack("🗑 删除作业",
                     f"✔ 已删除「{title}」→ 移入“已删除作业”（本地文件保留，可恢复）",
                     user, None)


def restore_job_click(request: gr.Request, job_id: str):
    """把已软删除的作业恢复为进行中。"""
    user = _user(request)
    if not _is_admin(user):
        return ("❌ 仅管理员可恢复",
                gr.update(choices=_mgmt_choices(), value=None),
                gr.update(choices=_deleted_choices(), value=None),
                jobs_progress_html(JM.list_jobs()),
                gr.update(choices=_job_choices(user), value=[]),
                staff_counts_html(_staff_rows()))
    job_id = (job_id or "").strip()
    if not job_id:
        return ("⚠️ 请先选择要恢复的作业",
                gr.update(choices=_mgmt_choices(), value=None),
                gr.update(choices=_deleted_choices(), value=None),
                jobs_progress_html(JM.list_jobs()),
                gr.update(choices=_job_choices(user), value=[]),
                staff_counts_html(_staff_rows()))
    title = _job_title(job_id)
    try:
        JM.restore_job(job_id)
    except JobError as e:
        return (f"❌ {e}",
                gr.update(choices=_mgmt_choices(), value=None),
                gr.update(choices=_deleted_choices(), value=None),
                jobs_progress_html(JM.list_jobs()),
                gr.update(choices=_job_choices(user), value=[]),
                staff_counts_html(_staff_rows()))
    _cancel_delete()
    return (f"✔ 已恢复「{title}」为进行中，回到作业列表",
            gr.update(choices=_mgmt_choices(), value=None),
            gr.update(choices=_deleted_choices(), value=None),
            jobs_progress_html(JM.list_jobs()),
            gr.update(choices=_job_choices(user), value=[]),
            staff_counts_html(_staff_rows()))


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def _who_md(user: str) -> str:
    """账号栏文案：当前用户 + 角色。"""
    role = ACC.role(user) or "-"
    return f"👤 **当前用户**：`{user}`（{role}）"


def logout_click(request: gr.Request):
    """登出第一步：归还当前租约并清该用户工作态；随后给出 /logout 链接换账号。

    Gradio 6 原生的 GET /logout 会删除本会话 token(cookie) 并跳回登录框，
    届时用另一账号登录即完成切换；all_session=false 只登出当前浏览器，
    不把同一账号在其它标签页/机器上的会话踢掉。
    """
    user = _user(request)
    st = wstate(user)
    notes = []
    if st.get("job_id"):
        try:
            if JM.release(st["job_id"], user):
                notes.append("已归还当前道集（回池，可被他人领取）")
        except JobError:
            notes.append("归还当前道集失败，详见服务端日志")
        st.pop("gid", None)
        st.pop("partial", None)
        st.pop("boxes", None)
    if not notes:
        notes.append("当前无在标注的道集")
    return gr.update(value="  ".join(notes) + "　**[退出登录并切换账号](/logout?all_session=false)**")


def init_page(request: gr.Request):
    """页面加载：按角色决定可见区、填充作业下拉，并显示当前账号。"""
    user = _user(request)
    admin = _is_admin(user)
    mgr_choices = [j["job_id"] for j in JM.list_jobs()] if admin else []
    del_choices = [j["job_id"] for j in JM.deleted_jobs()] if admin else []
    return (gr.update(choices=_job_choices(user), value=[]),
            gr.update(visible=admin),      # 建作业区
            gr.update(visible=admin),      # 管理区
            gr.update(choices=mgr_choices, value=None),
            gr.update(value=_who_md(user)),   # 账号栏
            gr.update(choices=del_choices, value=None),   # 已删除作业（可恢复）
            # 标注量表按最新数据重渲：组件初值是**服务启动那一刻**的快照，
            # 不刷新的话新开的页面会看到过期计数（等第一次 5 秒 tick 才修正）
            jobs_progress_html(JM.list_jobs()),
            _my_total_text(user))


def _box_clear(key: str):
    def handler(request: gr.Request):
        return clear_box(request, key)
    return handler


def _box_clear(key: str):
    def handler(request: gr.Request):
        return clear_box(request, key)
    return handler


ANNO_CSS = """
/* 标注区：左右两栏等高。图片区高度撑满标注行（不再被 Gradio Image 外壳的内联
   固定高 640px 截断），512×1024 竖长图始终完整可见，无需点“全局/适应”按钮。
   用 max-width/max-height 等比约束 + flex 居中：<img> 元素框恰等于所绘图区域，
   图上点框坐标与像素映射不受影响。 */
#anno_body { align-items: stretch; }
#preview-clip-slider { flex: 0 1 300px !important; max-width: 300px; }
#anno_imgcol, #anno_featcol { min-width: 0; }
#anno_imgcol { display: flex; flex-direction: column; }
#anno_imgcol > .block {
    height: 100% !important;       /* 压过组件内联 height:640px */
    min-height: 0;
    overflow: hidden;
}
#anno_imgcol .image-container { height: 100% !important; }
#anno_imgcol .image-frame {
    height: 100%;
    display: flex; align-items: center; justify-content: center;
}
#anno_imgcol .image-frame img {
    width: auto; height: auto;
    max-width: 100%; max-height: 100%;
    object-fit: contain; display: block;
}

/* 管理区作业进度一览表 */
.jobs-progress { width: 100%; border-collapse: collapse; font-size: 13px; margin: 6px 0; }
.jobs-progress th, .jobs-progress td {
    border-bottom: 1px solid var(--border-color-primary);
    padding: 4px 8px; text-align: left; vertical-align: middle;
}
.jobs-progress th { font-weight: 600; }
.jobs-progress .jp-id { display: block; font-size: 11px; opacity: .6; }
.jobs-progress .jp-state, .jobs-progress .jp-count, .jobs-progress .jp-pct {
    white-space: nowrap;
}
.jp-bar { width: 30%; }
.jp-bg { background: var(--border-color-primary); border-radius: 8px;
         height: 10px; min-width: 120px; }
.jp-bg > div { height: 10px; border-radius: 8px; }
.jp-run { background: #2563eb; }
.jp-done { background: #16a34a; }
.jp-close { background: #9ca3af; }
.jp-empty { color: var(--body-text-color-subdued); }

/* 各用户标注总量表（管理区，仅管理员可见） */
.staff-counts { width: 100%; border-collapse: collapse; font-size: 13px; margin: 6px 0; }
.staff-counts th, .staff-counts td {
    border-bottom: 1px solid var(--border-color-primary);
    padding: 4px 8px; text-align: left; vertical-align: middle;
}
.staff-counts th { font-weight: 600; }
.staff-counts .sc-count { text-align: right; font-variant-numeric: tabular-nums; }
.staff-counts .sc-role { color: var(--body-text-color-subdued); }
.staff-counts .sc-total td { font-weight: 600; border-top: 1px solid var(--border-color-primary); }
"""

# 多点包络：bbox 特征配置（key/名称/颜色/条件），供前端注入 JS 使用
def _when_meta(feat):
    w = feat.when or {}
    parent = next((x.key for x in CFG.features if x.name == w.get("feature")), "")
    return [parent, w.get("equals", ""), w.get("not_equals", "")]


_DRAG_BBOX = [[f.key, f.name, f.bbox_color, f.region_shape, *_when_meta(f),
               f.key == "abnormal_amplitude"]
              for f in CFG.features if f.bbox]


def _absent_label(feat) -> str:
    """该特征「不存在」选项的显示 label；没有这个选项就返回 ""（永远不会被跳过）。"""
    for o in feat.options:
        if o.label == ABSENT_LABEL:
            return o.label
    return ""


# 全部特征（键+名+「不存在」label，顺序=题号顺序）：前端据此按 #q-<key> / #bx-<key> 定位
# 每个题项、并判断某个拉框项是否可以跳过（不依赖 Gradio 生成的 DOM 细节）
_ANNO_FEATS = [[f.key, f.name, _absent_label(f), f.input, *_when_meta(f),
                [o.label for o in f.options if o.exclusive]] for f in CFG.features]

ANNO_JS = """
<style>
#anno_imgcol .image-container { position: relative; }
#anno_imgcol .image-container img {
    cursor: crosshair;
    -webkit-user-drag: none;    /* 禁掉原生图片拖拽，框的拖动/缩放才拿得到手势 */
    user-select: none;
}
#anno_drag_canvas {
    position: absolute; pointer-events: none; z-index: 50;
    image-rendering: auto; border: 0; touch-action: none;
}
/* 当前题高亮（数字键答的就是它；↑/↓ 前后移动） */
.anno-current {
    outline: 2px solid #2563eb !important;
    outline-offset: 2px;
    background: rgba(37, 99, 235, 0.10);
    border-radius: 6px;
    transition: background .12s ease;
}
.anno-current * { font-weight: 600; }
/* 选「不存在」后该拉框项自动跳过：置灰提示这一步不用做 */
.anno-skip { opacity: .42; }
</style>
<script>
(function(){
  const BBOX = __DRAG_BBOX__;
  const FEATS = __ANNO_FEATS__;     // [[key,name],...]，顺序 = 界面题号顺序（1..N）
  const KEY2COLOR = {}, KEY2NAME = {}, KEY2SHAPE = {}, KEY2MULTI = {};
  BBOX.forEach(function(b){ KEY2COLOR[b[0]] = b[2] || '#ff0000'; KEY2NAME[b[0]] = b[1] || b[0]; KEY2SHAPE[b[0]] = b[3] || 'rectangle'; KEY2MULTI[b[0]] = b[7] === true || b[0] === 'abnormal_amplitude'; });
  let boxes = {};          // key -> box 或 box[]，坐标 = 图片自然像素
  let targetKey = BBOX.length ? BBOX[0][0] : null;  // Ctrl+两点当前画给谁（服务端给的 target）
  let targetNote = '';     // 目标为空时服务端给的准确说明；空串 = 什么都不画
  let pending = null;      // Ctrl 连续点击中的包络：{key:…, points:[[nx,ny],...]}
  let multiDirty = false; // 异常振幅的多个矩形尚未按空格确认
  let hoverNat = null;     // 最近一次鼠标位置（自然像素），用于橡皮筋预览
  let grab = null;         // 当前手势 {type:move|resize, key, ...}
  let canvas = null, lastSrc = '';
  let cursor = 0;          // 当前题序号（0-based，覆盖 1..N 选择题 + 拉框项）
  let lastSig = '';        // 上次应用的画布几何签名；未变则 place() 直接返回
  let rafId = 0;           // requestAnimationFrame 合并用
  const MIN = 4;           // 最小边长（自然像素）

  function getImg(){
    var im = document.querySelector('#anno_imgcol .image-container img');
    return (im && im.complete && im.naturalWidth) ? im : null;
  }
  // 取当前登录用户：优先读账号栏渲染后的 <code>（markdown 反引号渲染后不再是文本反引号）
  function currentUser(){
    var md = document.getElementById('who_md'); if (!md) return null;
    var code = md.querySelector('code');
    if (code){ var u = (code.textContent || '').trim(); if (u) return u; }
    var t = md.textContent || '';
    var i = t.indexOf('当前用户');            // 兜底：渲染后文本形如 "当前用户：boss（admin）"
    if (i >= 0){
      var seg = t.slice(i + '当前用户'.length);
      var j = seg.search(/[：:]/); if (j >= 0) seg = seg.slice(j + 1);
      seg = seg.trim();
      var cut = seg.search(/[（( ]/);
      var name = (cut >= 0 ? seg.slice(0, cut) : seg).trim();
      if (name) return name;
    }
    return null;
  }
  function ensureCanvas(){
    var w = document.querySelector('#anno_imgcol .image-container'); if (!w) return null;
    var cv = document.getElementById('anno_drag_canvas');
    if (!cv){ cv = document.createElement('canvas'); cv.id = 'anno_drag_canvas'; w.appendChild(cv); }
    cv.style.pointerEvents = 'none';          // 指针一律透传给下层 <img>
    return cv;
  }
  // 指针事件绑在 <img> 上（它始终在接收事件，不受画布时序/重建影响）
  function bindImg(im){
    if (!im || im.__annoBound) return;
    im.__annoBound = true;
    im.addEventListener('pointerdown', onDown);
    im.addEventListener('pointermove', onMove);
    im.addEventListener('pointerup', onUp);
    im.addEventListener('pointercancel', onCancel);
    im.addEventListener('pointerleave', onLeave);
    im.addEventListener('contextmenu', onCtxMenu);   // 右键清框（并吃掉浏览器菜单）
  }
  function clamp(v, a, b2){ return Math.max(a, Math.min(b2, v)); }
  function imgRect(){
    var im = (canvas && canvas._img) ? canvas._img : getImg();
    return im ? im.getBoundingClientRect() : null;
  }
  // 局部(0..图片显示宽) -> 自然像素；绘制/命中/指针统一按同一矩形换算
  function localToNat(lx, ly){
    var im = (canvas && canvas._img) ? canvas._img : getImg();
    var r = imgRect();
    if (!im || !r || !r.width || !r.height) return null;
    return [Math.max(0, Math.min(im.naturalWidth - 1, lx / r.width  * im.naturalWidth)),
            Math.max(0, Math.min(im.naturalHeight - 1, ly / r.height * im.naturalHeight))];
  }
  function screenRect(b){                       // 自然像素 -> 局部(0..显示宽高)
    var im = (canvas && canvas._img) ? canvas._img : getImg();
    var r = imgRect(); if (!im || !r) return {x:0,y:0,w:0,h:0};
    var bb = b;
    if (b.points && b.points.length){
      var xs = b.points.map(function(p){return p[0];}), ys = b.points.map(function(p){return p[1];});
      bb = {x0:Math.min.apply(null,xs), y0:Math.min.apply(null,ys),
            x1:Math.max.apply(null,xs), y1:Math.max.apply(null,ys)};
    }
    return {x: bb.x0/im.naturalWidth*r.width,  y: bb.y0/im.naturalHeight*r.height,
            w: (bb.x1-bb.x0)/im.naturalWidth*r.width, h: (bb.y1-bb.y0)/im.naturalHeight*r.height};
  }
  function copyBox(b){ return {x0:b.x0, y0:b.y0, x1:b.x1, y1:b.y1}; }
  function boxItems(key){
    var value = boxes[key];
    return Array.isArray(value) ? value : (value ? [value] : []);
  }
  function natCorner(b, which){
    switch(which){
      case 'nw': return [b.x0, b.y0];
      case 'ne': return [b.x1, b.y0];
      case 'sw': return [b.x0, b.y1];
      default:   return [b.x1, b.y1];
    }
  }
  function redraw(){
    if (!canvas) return;
    var c = canvas.getContext('2d');
    c.clearRect(0, 0, canvas.width, canvas.height);
    if (!canvas._img) return;
    var hs = Math.max(7, Math.min(14, Math.round(canvas.width * 0.02) || 9));
    for (var key in boxes){
      var list = boxItems(key), col = KEY2COLOR[key] || '#1f6feb';
      for (var bi = 0; bi < list.length; bi++){
      var b = list[bi]; if (!b) continue;
      var s = screenRect(b);
      c.strokeStyle = col;
      c.lineWidth = (grab && grab.key === key) ? 4 : 3;
      var isRect = KEY2SHAPE[key] === 'rectangle';
      var pts = (!isRect && b.points && b.points.length) ? b.points.map(function(p){
        return [p[0]/canvas._img.naturalWidth*canvas.width,
                p[1]/canvas._img.naturalHeight*canvas.height];
      }) : [[s.x,s.y],[s.x+s.w,s.y],[s.x+s.w,s.y+s.h],[s.x,s.y+s.h]];
      c.beginPath(); c.moveTo(pts[0][0], pts[0][1]);
      for (var pi=1; pi<pts.length; pi++) c.lineTo(pts[pi][0], pts[pi][1]);
      c.lineTo(pts[0][0], pts[0][1]); c.stroke();
      for (var i = 0; i < pts.length; i++){
        var px = pts[i][0], py = pts[i][1];
        c.fillStyle = col; c.fillRect(px - hs/2, py - hs/2, hs, hs);
        c.fillStyle = '#ffffff'; c.fillRect(px - hs/4, py - hs/4, hs/2, hs/2);
      }
    }
    }
    drawBoxHint(c);
  }
  // 画布左上角的目标提示 + Ctrl 多点包络的实时预览
  function drawBoxHint(c){
    if (canvas.width > 60){
      var name = targetKey ? (KEY2NAME[targetKey] || targetKey) : null;
      // 目标为空时**不要**自作主张说"已完成"：那只可能是"两项都选了不存在"或"没有当前道集"。
      // 文案由服务端给（note），没有就不画 —— 免得在旧图上浮出一句误导性提示。
      var text = pending ? (pending.points ? ('继续 Ctrl+左键添加顶点，松开 Ctrl 完成（' + pending.points.length + ' 点）') : (KEY2MULTI[pending.key] ? 'Ctrl+左键点第二角完成一个矩形，继续添加；按空格完成' : 'Ctrl+左键点第二角后完成矩形'))
                         : (name ? (KEY2SHAPE[targetKey] === 'rectangle'
                                    ? (KEY2MULTI[targetKey] ? ('按住 Ctrl 点击两个角点添加矩形，按空格完成：' + name)
                                                               : ('按住 Ctrl 点击两个角点框选：' + name))
                                    : ('按住 Ctrl 点击多个点形成包络：' + name))
                                 : (targetNote || ''));
      if (text){
        var col = pending ? '#f59e0b' : '#2563eb';
        c.font = '13px sans-serif';
        var w = c.measureText(text).width;
        c.fillStyle = 'rgba(255,255,255,0.82)';
        c.fillRect(6, 6, w + 12, 22);
        c.fillStyle = col;
        c.fillText(text, 12, 21);
        c.strokeStyle = col; c.lineWidth = 1;
        c.strokeRect(6.5, 6.5, w + 11, 21);
      }
    }
    if (pending && pending.points && pending.points.length){
      var p = screenRect({x0:pending.points[0][0], y0:pending.points[0][1],
                          x1:pending.points[0][0], y1:pending.points[0][1]});
      var col2 = KEY2COLOR[pending.key] || '#1f6feb';
      c.strokeStyle = col2; c.lineWidth = 2;
      c.beginPath();
      pending.points.forEach(function(q, i){ var z=screenRect({x0:q[0],y0:q[1],x1:q[0],y1:q[1]});
        if (i===0) c.moveTo(z.x,z.y); else c.lineTo(z.x,z.y); });
      if (hoverNat){ var hz=screenRect({x0:hoverNat[0],y0:hoverNat[1],x1:hoverNat[0],y1:hoverNat[1]}); c.lineTo(hz.x,hz.y); }
      c.setLineDash([6,4]); c.stroke(); c.setLineDash([]);
    }
  }
  // 兼容异常振幅矩形模式的两个角点换算
  function rectFrom(a, b){
    var im = getImg(); if (!im) return {x0:0, y0:0, x1:0, y1:0};
    var W = im.naturalWidth - 1, H = im.naturalHeight - 1;
    return {x0: Math.round(clamp(Math.min(a[0], b[0]), 0, W)),
            y0: Math.round(clamp(Math.min(a[1], b[1]), 0, H)),
            x1: Math.round(clamp(Math.max(a[0], b[0]), 0, W)),
            y1: Math.round(clamp(Math.max(a[1], b[1]), 0, H))};
  }
  function polygonFrom(points){
    var im = getImg(); if (!im) return [];
    var W = im.naturalWidth - 1, H = im.naturalHeight - 1;
    return (points || []).map(function(p){ return [Math.round(clamp(p[0],0,W)), Math.round(clamp(p[1],0,H))]; });
  }
  // 同一帧内多次请求只做一次：避免滚动时在一帧里反复重排
  function raf(fn){
    if (rafId) return;
    rafId = requestAnimationFrame(function(){ rafId = 0; fn(); });
  }
  function rectSig(a, b){
    return [a.left - b.left, a.top - b.top, a.width, a.height]
             .map(function(v){ return Math.round(v); }).join(',');
  }
  function place(force){
    var w = document.querySelector('#anno_imgcol .image-container');
    var img = getImg();
    canvas = ensureCanvas();
    if (!canvas) return;
    if (!w || !img){ canvas.style.display = 'none'; return; }
    bindImg(img);
    if (img.currentSrc && img.currentSrc !== lastSrc){   // 换了道集/重开 → 清本地并回填服务器
      lastSrc = img.currentSrc;
      boxes = {}; grab = null; pending = null; hoverNat = null; multiDirty = false;
      cursor = firstUnansweredIndex(items());   // 光标回到第一道未答题
      lastSig = '';                             // 换了图 → 强制重排一次
      fetchBoxes();
      paintCursor(false);
    }
    var cr = w.getBoundingClientRect(), ir = img.getBoundingClientRect();
    var sig = rectSig(ir, cr) + '|' + (canvas._img === img ? 1 : 0);
    // 几何没变就**什么都不做**。canvas 与 img 同在一个 position:relative 容器里，
    // 页面滚动时两者一起移动、相对位置不变 —— 原来每次 scroll 都重排 + 重设
    // canvas.width（会重建画布缓冲、清空内容）+ 全量重绘，正是滚动卡顿的主因。
    if (!force && sig === lastSig) return;
    lastSig = sig;
    var w2 = Math.round(ir.width), h2 = Math.round(ir.height);
    canvas.style.display = '';
    canvas.style.left = (ir.left - cr.left) + 'px';
    canvas.style.top  = (ir.top  - cr.top)  + 'px';
    canvas.style.width  = w2 + 'px';
    canvas.style.height = h2 + 'px';
    if (canvas.width !== w2) canvas.width = w2;      // 赋同值也会清空画布，必须判一下
    if (canvas.height !== h2) canvas.height = h2;
    canvas._img = img; canvas._rect = ir;
    redraw();
  }
  function fetchBoxes(){
    var u = currentUser(); if (!u) return;
    fetch('/api/boxes?user=' + encodeURIComponent(u), {headers:{'Accept':'application/json'}})
      .then(function(r){ return r.ok ? r.json() : {boxes:{}, target:null, note:''}; })
      .then(function(d){
        if (!d) return;
        // 正在拖动/缩放时**不替换 boxes**：那会把手上这个框打回服务端的旧值。
        // 目标和提示仍照常更新（它们不碰几何，不影响手势）。
        if (!grab){
          var localMulti = (multiDirty && targetKey && KEY2MULTI[targetKey]) ? boxes[targetKey] : null;
          boxes = d.boxes || {};
          if (localMulti && Array.isArray(localMulti)) boxes[targetKey] = localMulti;
        }
        // 目标特征由服务端按「第一个需框未框」算（web_core.box_target），前端不猜 DOM；
        // note 是「目标为空」时服务端给的说明（如"所有拉框项均不存在"），空串则不画提示
        targetKey = d.target !== undefined ? d.target : targetKey;
        targetNote = d.note || '';
        if (pending && pending.key !== targetKey) pending = null;
        if (!grab){
          redraw();
          paintCursor(false);
        }
      })
      .catch(function(){});
  }
  // 去抖地回读一次 Ctrl 目标。触发点必须是**服务端 outputs 落到 DOM 之后**。
  //
  // 这个顺序曾经是错的：原来只在 document 的 change 监听里 fetchBoxes()，而那个监听
  // 比 Gradio 的 on_radio_change（写 st['partial'] 的那个）**先跑**，读到的还是改之前
  // 的选项；等服务端跑完，on_radio_change 的 outputs 里没有图片，place() 不触发，
  // 于是**再没人回读** —— target/note 永久慢一拍。表现就是：
  //   ① 两个都选「不存在」了，画布仍提示「框选：近炮点强能量噪声」→ 还能拉出一个框；
  //   ② 把面波改回「存在」，画布反而提示「所有拉框项均选「不存在」，本张无需画框」→ 框不出来。
  // 「⧉ 继承最近已标注」更彻底：它用 gr.Radio(value=…) 程序化改选项，压根不产生 DOM
  // change 事件，所以继承完画布目标从没刷新过。
  // 现在改为挂在 featcol 的 MutationObserver 上（服务端 outputs 就是写进这个列的），
  // 它触发时 st['partial'] 一定已经更新，读到的就是新状态。
  let refreshTimer = 0;
  function scheduleTargetRefresh(){
    if (refreshTimer) return;                       // 同一批变更只问一次
    refreshTimer = setTimeout(function(){ refreshTimer = 0; fetchBoxes(); }, 60);
  }
  function localPt(e){
    var r = imgRect();
    if (!r) return null;
    return {x: e.clientX - r.left, y: e.clientY - r.top};
  }
  // ---- 命中检测（局部坐标 0..显示宽高）----
  function hitTest(x, y){
    if (!canvas || !canvas._img) return null;
    var tol = Math.max(9, Math.round(canvas.width * 0.02));
    for (var key in boxes){
      var list = boxItems(key);
      for (var bi = 0; bi < list.length; bi++){
      var b = list[bi]; if (!b) continue;
      var s = screenRect(b);
      if (KEY2SHAPE[key] !== 'rectangle' && b.points && b.points.length){
        if (x >= s.x && x <= s.x+s.w && y >= s.y && y <= s.y+s.h) return {type:'polygon', key:key, index:bi};
        continue;
      }
      var corners = {nw:[s.x,s.y], ne:[s.x+s.w,s.y], sw:[s.x,s.y+s.h], se:[s.x+s.w,s.y+s.h]};
      for (var cName in corners){
        var p = corners[cName];
        if (Math.abs(p[0]-x) <= tol && Math.abs(p[1]-y) <= tol) return {type:'resize', key:key, corner:cName, index:bi};
      }
      if (x >= s.x && x <= s.x+s.w && y >= s.y && y <= s.y+s.h) return {type:'move', key:key, index:bi};
      }
    }
    return null;
  }
  // ---- 交互（全部以图片局部坐标运算，与绘制同系）----
  // 框够大吗：太小的（Ctrl 手抖/点两下没动）不算框，挡在 MIN 这一关
  function bigEnough(b){
    if (!b) return false;
    if (b.points) return b.points.length >= 3;
    return (b.x1 - b.x0) >= MIN && (b.y1 - b.y0) >= MIN;
  }
  function commitPolygon(key, points){
    var ps = polygonFrom(points);
    if (ps.length < 3) return;
    var xs=ps.map(function(p){return p[0];}), ys=ps.map(function(p){return p[1];});
    var b = {points: ps, x0:Math.min.apply(null,xs), y0:Math.min.apply(null,ys),
             x1:Math.max.apply(null,xs), y1:Math.max.apply(null,ys)};
    if (KEY2MULTI[key]){
      if (!Array.isArray(boxes[key])) boxes[key] = [];
      boxes[key].push(b);
      multiDirty = true;
      return;
    }
    boxes[key] = b;
    // 先在本地推进高亮，避免用户完成矩形后仍看到旧的包络提示。
    advanceAfterBox(key);
    // postBox 在服务端写入成功后才回读 /api/boxes，避免读到旧 target。
    postBox(key, b);
  }
  function finishMultiRect(){
    if (!targetKey || !KEY2MULTI[targetKey] || pending) return false;
    var list = boxItems(targetKey);
    if (!list.length) return false;
    var key = targetKey;
    postBox(key, list);
    advanceAfterBox(key, true);
    redraw();
    return true;
  }
  function onDown(e){
    if (!canvas || !canvas._img) return;
    // 只处理主键：右键走 contextmenu（见 onCtxMenu），中键不参与手势。
    // 不在这儿拦住的话，右键落在已有框上会被 hitTest 命中、白白启动一次"拖动"并捕获指针。
    if (e.button) return;
    var lp = localPt(e); if (!lp) return;
    var nat = localToNat(lp.x, lp.y); if (!nat) return;
    hoverNat = nat;                                 // 松开 Ctrl 成框时要用「最近一次鼠标位置」
    // 一律吃掉默认行为：否则在 <img> 上按住左键会启动浏览器**原生图片拖拽**，
    // 手势被浏览器接管 → 框既拖不动也缩不了（拖动/缩放曾经"失灵"就是这个原因）。
    e.preventDefault(); e.stopPropagation();
    // ---- Ctrl（或 Mac Cmd）+左键：异常振幅两点矩形，其他特征多点包络 ----
    if (e.ctrlKey || e.metaKey){
      if (!targetKey) return;                       // 无 bbox 特征可框（或都「不存在」）
      if (pending && pending.key === targetKey && KEY2SHAPE[targetKey] === 'rectangle'){
        var rb = rectFrom(pending.p, nat);
        pending = null;
        if (bigEnough(rb)) commitPolygon(targetKey, [[rb.x0,rb.y0],[rb.x1,rb.y0],[rb.x1,rb.y1],[rb.x0,rb.y1]]);
      } else if (pending && pending.key === targetKey){
        pending.points.push(nat);
      } else {
        pending = KEY2SHAPE[targetKey] === 'rectangle'
          ? {key: targetKey, p: nat}
          : {key: targetKey, points: [nat]};
      }
      redraw();
      return;
    }
    // ---- 普通左键：拖动 / 拖角缩放微调已有框 ----
    var hit = hitTest(lp.x, lp.y);
    if (!hit) return;                               // 空白区点击不做任何事
    if (hit.type === 'polygon') return;             // 多边形清除后重画，不做矩形式拖拽
    var key = hit.key;
    grab = hit;
    var hitBox = boxItems(key)[hit.index || 0];
    grab.snapshot = hitBox ? copyBox(hitBox) : null;
    if (hit.type === 'resize'){
      var opp = {nw:'se', se:'nw', ne:'sw', sw:'ne'}[hit.corner];
      grab.anchorNat = natCorner(hitBox, opp);       // 对角锚点（自然像素，固定）
    } else {
      grab.nat0 = nat;                              // 整体移动起点
    }
    if (e.target.setPointerCapture){ try { e.target.setPointerCapture(e.pointerId); } catch(_){} }
    redraw();
  }
  function onMove(e){
    if (!canvas || !canvas._img) return;
    var lp = localPt(e); if (!lp) return;
    hoverNat = localToNat(lp.x, lp.y);              // 记下位置：Ctrl 第一角后的橡皮筋预览用
    if (!grab){                                     // 悬停：仅更新光标提示
      if (e.ctrlKey || e.metaKey || pending){       // Ctrl 态/待定第一角 → 十字，准备定点
        if (e.target.style) e.target.style.cursor = 'crosshair';
        if (pending) redraw();
        return;
      }
      var h = hitTest(lp.x, lp.y);
      var cur = h ? (h.type === 'polygon' ? 'crosshair' : h.type === 'move' ? 'move'
                     : (h.corner === 'nw' || h.corner === 'se') ? 'nwse-resize' : 'nesw-resize')
                  : 'crosshair';
      if (e.target.style) e.target.style.cursor = cur;
      return;
    }
    e.preventDefault();
    var nat = hoverNat; if (!nat) return;
    var g = grab, key = g.key, im = canvas._img;
    var list = boxItems(key), b = list[g.index || 0] || {x0:0, y0:0, x1:0, y1:0};
    var W = im.naturalWidth - 1, H = im.naturalHeight - 1;
    if (g.type === 'move'){
      var dx = nat[0] - g.nat0[0], dy = nat[1] - g.nat0[1];
      var w0 = g.snapshot.x1 - g.snapshot.x0, h0 = g.snapshot.y1 - g.snapshot.y0;
      b.x0 = Math.round(clamp(g.snapshot.x0 + dx, 0, W - w0));
      b.y0 = Math.round(clamp(g.snapshot.y0 + dy, 0, H - h0));
      b.x1 = b.x0 + w0; b.y1 = b.y0 + h0;
    } else if (g.type === 'resize'){
      var A = g.anchorNat;
      var nx = clamp(nat[0], 0, W), ny = clamp(nat[1], 0, H);
      if (g.corner === 'nw'){ b.x0 = Math.round(clamp(nx, 0, A[0] - MIN)); b.y0 = Math.round(clamp(ny, 0, A[1] - MIN)); }
      else if (g.corner === 'se'){ b.x1 = Math.round(clamp(nx, A[0] + MIN, W)); b.y1 = Math.round(clamp(ny, A[1] + MIN, H)); }
      else if (g.corner === 'ne'){ b.x1 = Math.round(clamp(nx, A[0] + MIN, W)); b.y0 = Math.round(clamp(ny, 0, A[1] - MIN)); }
      else if (g.corner === 'sw'){ b.x0 = Math.round(clamp(nx, 0, A[0] - MIN)); b.y1 = Math.round(clamp(ny, A[1] + MIN, H)); }
    }
    if (KEY2MULTI[key]) boxes[key][g.index || 0] = b;
    else boxes[key] = b;
    redraw();
  }
  // 右键清框的服务端提交：直连自定义接口，不程序化点 Gradio 按钮（右键不该依赖它的 DOM 层级）。
  // 先取消还没到点的去抖回读：否则它可能在服务端清掉**之前**发出、把刚清掉的框又拉回来。
  function postClear(key){
    var u = currentUser(); if (!u) return;
    if (refreshTimer){ clearTimeout(refreshTimer); refreshTimer = 0; }
    fetch('/api/anno_box_clear', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({user:u, key:key})})
      .then(function(){ fetchBoxes(); })            // 以服务端为准收尾（目标随之顺延回来）
      .catch(function(){});
  }
  function postBox(key, b){
    var u = currentUser(); if (!u) return Promise.resolve();
    var payload = {user:u, key:key};
    if (KEY2MULTI[key] && Array.isArray(b)){
      payload.regions = b.map(function(item){ return item.points; });
    } else {
      payload.points = Array.isArray(b) ? b : b.points;
    }
    return fetch('/api/anno_box', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)})
      .then(function(){ if (KEY2MULTI[key]) multiDirty = false; fetchBoxes(); })
      .catch(function(){});
  }
  function endGesture(e, commit){
    if (!grab) return;
    var g = grab; grab = null;
    if (e.target && e.target.setPointerCapture){ try { e.target.releasePointerCapture(e.pointerId); } catch(_){} }
    var b = boxItems(g.key)[g.index || 0];
    var ok = bigEnough(b);
    if (commit && ok){
      b.x0 = Math.round(b.x0); b.y0 = Math.round(b.y0);
      b.x1 = Math.round(b.x1); b.y1 = Math.round(b.y1);
      if (KEY2SHAPE[g.key] === 'rectangle'){
        b.points = [[b.x0,b.y0],[b.x1,b.y0],[b.x1,b.y1],[b.x0,b.y1]];
      }
      if (KEY2MULTI[g.key]) boxes[g.key][g.index || 0] = b;
      else boxes[g.key] = b;
      postBox(g.key, KEY2MULTI[g.key] ? boxes[g.key] : b); // 拖动/缩放微调后同样提交
    } else {
      if (g.snapshot){
        if (KEY2MULTI[g.key]) boxes[g.key][g.index || 0] = g.snapshot;
        else boxes[g.key] = g.snapshot;
      } else if (KEY2MULTI[g.key]) boxes[g.key].splice(g.index || 0, 1);
      else delete boxes[g.key];
    }
    if (e.target && e.target.style) e.target.style.cursor = '';
    redraw();
  }
  function onUp(e){ endGesture(e, true); }
  function onCancel(e){ endGesture(e, false); }
  function onLeave(e){
    hoverNat = null;
    if (e.target && e.target.style) e.target.style.cursor = '';
    if (pending) redraw();
  }
  // 松开 Ctrl（Mac 上的 Cmd）→ 以**松开瞬间的鼠标位置**作第二角直接成框。
  // 「Ctrl 再点第二角」原样保留，两条路共用 commitBox，行为完全一致。
  function onKeyUp(e){
    if (e.key !== 'Control' && e.key !== 'Meta' && e.key !== 'OS') return;
    if (!pending || !targetKey || pending.key !== targetKey) return;
    // 指针已离开图片（onLeave 清了 hoverNat）→ 不知道第二角在哪，留着第一角别瞎猜
    var nat = hoverNat;
    if (!nat) return;
    // pending.p is accepted only for pre-polygon clients/tests; the current UI always uses points.
    if (!pending.points && pending.p){
      var rb = rectFrom(pending.p, nat);
      if (!bigEnough(rb)) return;
      pending = null;
      commitPolygon(targetKey, [[rb.x0,rb.y0],[rb.x1,rb.y0],[rb.x1,rb.y1],[rb.x0,rb.y1]]);
      redraw();
      return;
    }
      if (KEY2SHAPE[targetKey] === 'rectangle') return;
      var points = pending.points.slice();
    if (nat && (points.length === 0 || points[points.length-1][0] !== nat[0] || points[points.length-1][1] !== nat[1])) points.push(nat);
    if (points.length < 3) return;
    pending = null;
    commitPolygon(targetKey, points);
    redraw();
  }
  // 右键：清除此框。先取消还没闭合的半成品；没有半成品时，若鼠标压在某个已落定的框上
  // 就删掉那个框（与「✕ 清除此框」同一语义）。放在 contextmenu 上而不是 pointerdown：
  // 一是能顺手 preventDefault 掉浏览器菜单，二是不去打扰 pointer 手势状态机。
  function onCtxMenu(e){
    if (!canvas || !canvas._img) return;
    e.preventDefault(); e.stopPropagation();
    if (pending){ pending = null; redraw(); return; }
    var lp = localPt(e); if (!lp) return;
    var hit = hitTest(lp.x, lp.y);
    if (!hit) return;                               // 空白处右键：什么都不做，别误删
    clearKey(hit.key);                              // 本地立刻消失，点一下即有反馈
    postClear(hit.key);                             // 服务端同步清掉，清完回读目标
  }
  // ---- 按钮联动：✕ 清除该特征的框 ----
  function clearKey(key){
    pending = null;
    if (KEY2MULTI[key]) multiDirty = false;
    delete boxes[key];                  // 先本地清掉，点一下即消失
    if (canvas) { redraw(); place(); }
    // 注意：这里**不能**立刻 fetchBoxes()。服务端要等本次 Gradio 事件跑完才清掉框，
    // 立刻回填会把旧框又拉回来，表现成「第一下没清除、点第二次才清」。
    // 服务端返回的新图会换 src → place() 的换图分支自然会重新拉一次（那时已经是清的）。
  }
  // 键盘总入口：数字键答题并前移、↑/↓ 跨题移动、Enter 保存并下一张
  function onKeydown(e){
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    // Residual jobs cycle the three aligned sources with D or Right Arrow.
    // The mode is server-authored in the gather info, so shot jobs keep the
    // existing Right Arrow question navigation behavior.
    if (isResidualMode() && (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft'
        || e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight')){
      if (isTyping(e.target)) return;
      var next = e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight';
      var btnId = next ? '#btn-residual-next' : '#btn-residual-prev';
      var layerBtn = document.querySelector(btnId + ' button, ' + btnId);
      if (layerBtn){
        e.preventDefault(); e.stopPropagation(); layerBtn.click();
      }
      return;
    }
    if ((e.key === ' ' || e.code === 'Space') && !isTyping(e.target)){
      if (finishMultiRect()) { e.preventDefault(); e.stopPropagation(); }
      return;
    }
    // Enter 单独判：只读字段（如「句子预览」）不算「正在输入」，按 Enter 仍应保存；
    // 可写字段（建作业表单）里照常换行/输入，不抢 Enter。
    if (e.key === 'Enter'){
      var t = e.target || {};
      if (isTyping(t) && !t.readOnly) return;
      // 一律 preventDefault：否则焦点停在上一个点过的按钮上时，浏览器会原生再点它
      // （例如「领取下一张」→ 提示「尚未保存」而根本没保存），Enter 就"没反应"了。
      e.preventDefault(); e.stopPropagation();
      clickSave();
      return;
    }
    if (isTyping(e.target)) return;                 // 建作业表单/句子框里照常输入
    // ↑/↓ 必须吃掉默认行为：焦点在 radio 上时浏览器默认是「同组内换选项」，
    // 那正是「方向键在同一题里跳选项」的来源。捕获阶段先拦，再移动题目光标。
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown' || e.key === 'ArrowLeft'
        || e.key === 'ArrowRight'){
      e.preventDefault(); e.stopPropagation();
      if (e.key === 'ArrowUp') moveCursor(-1);
      else if (e.key === 'ArrowDown') moveCursor(1);
      return;
    }
    var d = null;
    if (e.key >= '1' && e.key <= '9') d = parseInt(e.key, 10);
    else if (e.code && /^Numpad[1-9]$/.test(e.code)) d = parseInt(e.code.slice(6), 10);
    if (d === null) return;
    if (handleDigit(d)) e.preventDefault();
    else paintCursor(false);
  }
  function isResidualMode(){
    var infoEl = document.getElementById('anno_info');
    return checkedLabel('gather_type') === '残差' ||
      !!(infoEl && (infoEl.textContent || '').indexOf('作业类型：残差') >= 0);
  }
  function syncResidualButton(){
    ['btn-residual-prev', 'btn-residual-next'].forEach(function(id){
      var button = document.getElementById(id);
      if (button && button.style) button.style.display = isResidualMode() ? '' : 'none';
    });
  }
  // ---- 键盘：数字键答当前题并自动前移（焦点在文本框里时不拦截）----
  // 每题单独包在一个 id=q-<key> 的容器里（见 build_app），据此**按 id 确定性分组**。
  // 不能用 radio 的 name 属性分组：Gradio 6 的 radio 输入可能没有 name，
  // 那时按 name + 下标兜底会把**每个选项**当成一题，方向键就变成在同一题的选项间乱跳。
  function radiosOf(key){
    var el = document.getElementById('q-' + key);
    return el ? Array.prototype.slice.call(el.querySelectorAll('input[type="radio"], input[type="checkbox"]')) : [];
  }
  function optionText(inp){
    var lb = inp.closest ? inp.closest('label') : null;
    return lb ? (lb.textContent || '').trim() : '';
  }
  function isTyping(el){
    if (!el) return false;
    if (el.isContentEditable) return true;
    var tag = (el.tagName || '').toLowerCase();
    if (tag === 'textarea' || tag === 'select') return true;
    if (tag === 'input'){
      var ty = (el.type || '').toLowerCase();
      return !(ty === 'radio' || ty === 'checkbox' || ty === 'button' || ty === 'submit');
    }
    return false;
  }
  function groupAnswered(g){
    for (var j = 0; j < g.length; j++) if (g[j].checked) return true;
    return false;
  }
  // 选项在界面上带快捷键前缀（如 "2 不存在"），比较时要去掉，才能与配置里的 label 对齐。
  // 前缀要求「数字 + 空白」，所以 "50Hz工业噪声" 这类以数字开头的选项名不会被误切。
  function stripNum(t){ return (t || '').replace(/^\\s*\\d+\\s+/, ''); }
  function checkedLabel(key){
    var g = radiosOf(key);
    for (var i = 0; i < g.length; i++) if (g[i].checked) return stripNum(optionText(g[i]));
    return '';
  }
  function featureVisible(f){
    var parent = f[4], eq = f[5], neq = f[6];
    if (!parent) return true;
    var cur = checkedLabel(parent);
    if (eq) return cur === eq;
    if (neq) return cur !== neq;
    return true;
  }
  function syncConditionalVisibility(){
    FEATS.forEach(function(f){
      var el = document.getElementById('q-' + f[0]);
      if (el && el.style) el.style.display = featureVisible(f) ? '' : 'none';
    });
    BBOX.forEach(function(b){
      var el = document.getElementById('bx-' + b[0]);
      if (el && el.style) el.style.display = featureVisible([b[0], b[1], '', 'radio', b[4], b[5], b[6]]) ? '' : 'none';
    });
  }
  // 该 bbox 特征此刻是否需要画框：选了「不存在」就不需要（与后端 box_target 同一规则）
  function needsBox(feat){
    var absent = feat[2];
    if (!absent) return true;                 // 该特征没有「不存在」选项 → 总要框
    var cur = checkedLabel(feat[0]);
    if (!cur) return true;                    // 还没答 → 先不跳过
    return cur !== absent;
  }
  function featOf(key){
    for (var i = 0; i < FEATS.length; i++) if (FEATS[i][0] === key) return FEATS[i];
    return null;
  }
  // 可跳过的题项：没框需求的那个拉框项
  function skippable(it){
    if (!it || it.kind !== 'box') return false;
    var f = featOf(it.key);
    return f ? !needsBox(f) : false;
  }
  // 按 dir 找下一个「可停留」的题项；前方全是可跳过的就原地不动
  function nextIndex(its, from, dir){
    var i = from + dir;
    while (i >= 0 && i < its.length && skippable(its[i])) i += dir;
    if (i < 0 || i >= its.length) return from;
    return i;
  }
  // 完成一个拉框后，高亮自动落到下一个可标注项（例如异常振幅 → 面波）。
  // 该推进在本地提交时立即发生，服务端回读只负责最终校正，避免保存请求的时序造成旧目标闪回。
  function advanceAfterBox(key, updateTarget){
    var its = items();
    for (var i = 0; i < its.length; i++){
      if (its[i].kind === 'box' && its[i].key === key){
        cursor = nextIndex(its, i, 1);
        if (updateTarget){
          targetKey = (its[cursor] && its[cursor].kind === 'box') ? its[cursor].key : null;
          targetNote = '';
        }
        paintCursor(true);
        return;
      }
    }
  }
  // 题序 = 各选择题（配置顺序）→ 各拉框项，与界面编号 1..N 严格一致；
  // 每项自带高亮元素 el（选择题 = 该题的 #q-<key> 容器，拉框项 = #bx-<key>）。
  function items(){
    var out = [];
    FEATS.forEach(function(f){
      var el = document.getElementById('q-' + f[0]);
      var g = radiosOf(f[0]);
      if (el && g.length && featureVisible(f)) out.push({kind:'radio', key:f[0], el:el, group:g});
    });
    BBOX.forEach(function(b){
      var el = document.getElementById('bx-' + b[0]);
      if (el && featureVisible([b[0], b[1], '', 'radio', b[4], b[5], b[6]])) out.push({kind:'box', key:b[0], el:el});
    });
    return out;
  }
  function itemEl(it){ return it.el; }
  function firstUnansweredIndex(its){
    for (var i = 0; i < its.length; i++){
      if (its[i].kind === 'radio' && !groupAnswered(its[i].group)) return i;
    }
    return Math.max(0, its.length - 1);
  }
  function clampCursor(its){
    if (!its.length){ cursor = 0; return 0; }
    if (cursor < 0) cursor = 0;
    if (cursor > its.length - 1) cursor = its.length - 1;
    return cursor;
  }
  // 高亮当前题（"2. 异常振幅"整块发光）+ 置灰可跳过的拉框项。
  // scrollIntoView 只在**用户主动移动光标**时做：否则每次重绘都滚一次，
  // 会跟用户自己的滚动"抢方向盘"（滚动时被拽回当前题，看起来就是卡顿）。
  function paintCursor(scrollIntoView){
    var its = items(); clampCursor(its);
    for (var i = 0; i < its.length; i++){
      var el = itemEl(its[i]);
      if (!el || !el.classList) continue;
      if (i === cursor) el.classList.add('anno-current');
      else el.classList.remove('anno-current');
      if (skippable(its[i])) el.classList.add('anno-skip');
      else el.classList.remove('anno-skip');
    }
    if (!scrollIntoView) return;
    var cur = its[cursor] ? itemEl(its[cursor]) : null;
    if (cur && cur.scrollIntoView) cur.scrollIntoView({block:'nearest'});
  }
  function optionByDigit(g, d){
    for (var i = 0; i < g.length; i++){
      var m = optionText(g[i]).match(/^(\\d+)/);
      if (m && parseInt(m[1], 10) === d){ g[i].click(); return true; }
    }
    return false;
  }
  // 数字键：答当前题 → 自动跳到下一题（拉框项不吃数字键；不需要框的拉框项直接跳过）
  function handleDigit(d){
    var its = items(); clampCursor(its);
    var it = its[cursor];
    if (!it || it.kind !== 'radio') return false;
    if (!optionByDigit(it.group, d)) return false;
    // 先落选项再算下一题：选「不存在」后，本题对应的拉框项这一步就会被跳过
    cursor = nextIndex(its, cursor, 1);
    paintCursor(true);
    return true;
  }
  function moveCursor(delta){
    var its = items();
    cursor = nextIndex(its, clampCursor(its), delta > 0 ? 1 : -1);
    paintCursor(true);
    return cursor;
  }
  // Enter = 点「保存并释放」。elem_id 可能落在包一层的 div 上，也可能直接落在 <button> 上；
  // 都找不到就按按钮文字兜底（保存并释放 这个文案很独特），保证 Enter 一定点得到。
  function saveBtn(){
    var el = document.getElementById('btn-save');
    if (el){
      if ((el.tagName || '').toUpperCase() === 'BUTTON') return el;
      var inner = el.querySelector ? el.querySelector('button') : null;
      if (inner) return inner;
    }
    var all = document.querySelectorAll ? document.querySelectorAll('button') : [];
    for (var i = 0; i < all.length; i++){
      var txt = (all[i].textContent || '');
      if (txt.indexOf('保存并释放') >= 0) return all[i];
    }
    return null;
  }
  function clickSave(){
    var btn = saveBtn();
    if (!btn || !btn.click) return false;
    btn.click();
    return true;
  }
  function setup(){
    document.addEventListener('click', function(e){
      var t = e.target && e.target.closest ? e.target.closest('[id^="cb-"]') : null;
      if (t){ var k = (t.id || '').replace(/^cb-/, ''); if (KEY2COLOR[k] !== undefined) clearKey(k); }
    });
    // 双保险：兜住原生图片拖拽（某些浏览器不看 -webkit-user-drag），否则拖动框会变成拖图片
    document.addEventListener('dragstart', function(e){
      if (e.target && e.target.closest && e.target.closest('#anno_imgcol')) e.preventDefault();
    }, true);
    document.addEventListener('keydown', onKeydown, true);
    // 松开 Ctrl 也能定第二角（见 onKeyUp）。用捕获阶段，免得被别的组件先 stopPropagation。
    document.addEventListener('keyup', onKeyUp, true);
    // 鼠标点选某一题 → 光标跟过去（方便接着用键盘往下答）
    document.addEventListener('change', function(e){
      var t = e.target;
      if (!t || (t.type !== 'radio' && t.type !== 'checkbox') || !t.closest || !t.closest('#anno_featcol')) return;
      if (t.type === 'checkbox'){
        var q = t.closest('[id^="q-"]'), key = q ? q.id.replace(/^q-/, '') : '';
        var feat = featOf(key), exclusive = (feat && feat[7]) || [];
        var label = stripNum(optionText(t));
        radiosOf(key).forEach(function(other){
          if (other === t || !other.checked) return;
          var otherLabel = stripNum(optionText(other));
          if ((t.checked && exclusive.indexOf(label) >= 0) ||
              (t.checked && exclusive.indexOf(otherLabel) >= 0)) other.click();
        });
      }
      syncConditionalVisibility();
      var its = items();
      for (var i = 0; i < its.length; i++){
        if (its[i].kind === 'radio' && its[i].group.indexOf(t) >= 0){ cursor = i; break; }
      }
      paintCursor(false);
      // 目标刷新的触发点**不在这里**：本监听比 Gradio 的 on_radio_change 先跑，此刻
      // 服务端 st['partial'] 还是改之前的值，问到的必然是旧 target（详见 scheduleTargetRefresh）。
      // 交给 featcol 观察者在服务端 outputs 落地后再问。
    }, true);
    window.addEventListener('resize', function(){ if (!grab) raf(function(){ place(); }); });
    // 滚动：只做「几何变了才重排」的一次比对，平静滚动零开销（见 place 的签名判断）。
    // 之前是同步 place()，每次 scroll 事件都重建画布 + 全量重绘，这是卡顿的主因之一。
    window.addEventListener('scroll', function(){ if (!grab) raf(function(){ place(); }); }, true);
    var root = document.getElementById('anno_imgcol') || document.body;
    if (window.MutationObserver){
      // 图片区：只关心 img 被换掉（childList）或 src 变了；不再监听 class ——
      // Gradio 会频繁切换 class（pending/generating/selected），之前每次都触发重排。
      new MutationObserver(function(){ if (!grab) raf(function(){ place(); }); })
        .observe(root, {childList:true, subtree:true, attributes:true, attributeFilter:['src']});
      // 题目区：Gradio 重渲染会丢掉 .anno-current 高亮，只在这种结构变化时补画一次；
      // 不再挂在 place() 里（否则滚动/定时也会连带跑一遍 DOM 查询与 class 切换）。
      // 同一个观察者顺带做 Ctrl 目标的回读 —— 选项类 outputs（句子预览 / 框状态）就写在这列里，
      // 它一触发就说明服务端 handler 已经跑完，这时读 /api/boxes 才拿得到新选项算出的 target。
      var fcol = document.getElementById('anno_featcol');
      if (fcol){
        new MutationObserver(function(){
          raf(function(){ syncConditionalVisibility(); syncResidualButton(); paintCursor(false); });
          scheduleTargetRefresh();
        }).observe(fcol, {childList:true, subtree:true});
      }
      var info = document.getElementById('anno_info');
      if (info){
        new MutationObserver(syncResidualButton).observe(info, {childList:true, subtree:true});
      }
    }
    window.addEventListener('load', function(ev){
      if (ev.target && ev.target.tagName === 'IMG' && ev.target.closest && ev.target.closest('#anno_imgcol')){
        raf(function(){ place(); });
      }
    }, true);
    // 兜底：图片加载/重渲染时序不稳时，周期校正画布位置与 img 事件绑定。
    // place() 已带几何签名判断，没变化时只是两次 rect 读取，不会重排/重绘。
    setInterval(function(){ if (!grab){ var im = getImg(); if (im) raf(function(){ place(); }); } }, 900);
    syncConditionalVisibility();
    syncResidualButton();
    place();
  }
  // 测试钩子：在 Node 里以最小 DOM 桩加载本脚本时暴露纯逻辑，便于无浏览器验证
  // （见 tests/test_anno_js.py）；正常页面不设 window.__ANNO_TEST__ 即为无副作用。
  if (window.__ANNO_TEST__){
    window.__annoTest = {items: items, handleDigit: handleDigit, moveCursor: moveCursor,
                         paintCursor: paintCursor, firstUnansweredIndex: firstUnansweredIndex,
                         rectFrom: rectFrom, isTyping: isTyping, optionText: optionText,
                         onKeydown: onKeydown, clickSave: clickSave, saveBtn: saveBtn,
                         needsBox: needsBox, skippable: skippable, nextIndex: nextIndex,
                         onKeyUp: onKeyUp, onCtxMenu: onCtxMenu, bigEnough: bigEnough,
                         scheduleTargetRefresh: scheduleTargetRefresh,
                         cursorIndex: function(){ return cursor; },
                         setCursor: function(i){ cursor = i; },
                         // 状态读写：Ctrl 两点/右键清框的状态机没法在 pytest 里真跑鼠标，
                         // 靠这几个存取器把状态摆到位再直接调处理函数（同 setCursor 的思路）。
                         pending: function(){ return pending; },
                         setPending: function(p){ pending = p; },
                         hoverNat: function(){ return hoverNat; },
                         setHover: function(n){ hoverNat = n; },
                         targetKey: function(){ return targetKey; },
                         setTarget: function(k){ targetKey = k; },
                         setCanvas: function(c){ canvas = c; },
                         boxes: function(){ return boxes; },
                         setBoxes: function(b){ boxes = b; },
                         setGrab: function(g){ grab = g; },
                         setup: setup};
  }
  var tries = 0;
  var iv = setInterval(function(){
    tries++;
    if (document.getElementById('anno_imgcol')){ clearInterval(iv); setup(); }
    else if (tries > 80) clearInterval(iv);
  }, 400);
})();
</script>
""".replace("__DRAG_BBOX__", json.dumps(_DRAG_BBOX, ensure_ascii=False)) \
   .replace("__ANNO_FEATS__", json.dumps(_ANNO_FEATS, ensure_ascii=False))


def build_app() -> gr.Blocks:
    feats = CFG.features

    with gr.Blocks(title="地震道集标注器（多用户）") as demo:
        gr.Markdown("# 地震道集标注器（多用户中央服务）")

        # ---------- 账号栏（登出 / 切换账号） ----------
        with gr.Row():
            who = gr.Markdown("正在加载账号…", scale=4, elem_id="who_md")
            btn_logout = gr.Button("登出 / 切换账号", scale=1)

        # ---------- ① 建作业（仅管理员可见，服务端校验为准） ----------
        with gr.Accordion("① 建作业（仅管理员）", open=False, visible=False) as admin_build:
            job_type = gr.Dropdown(["炮集", "残差"], value="炮集", label="作业类型（管理员创建时确定）")
            with gr.Row():
                file_path = gr.Textbox(label="sgy/segy 文件路径", scale=3,
                                       placeholder="/path/to/xxx.sgy")
                endian = gr.Dropdown(["自动检测", "大端", "小端"], value="自动检测",
                                     label="字节序", scale=1)
                btn_load = gr.Button("读取文件信息", scale=1)
            file_info = gr.Markdown("尚未加载文件")
            with gr.Column(visible=False) as residual_files:
                residual_before = gr.Textbox(label="去噪前 SGY 文件路径",
                                             placeholder="/path/to/before.sgy")
                residual_after = gr.Textbox(label="去噪后 SGY 文件路径",
                                            placeholder="/path/to/after.sgy")
                residual_noise = gr.Textbox(label="噪声残差 SGY 文件路径",
                                            placeholder="/path/to/residual.sgy")
                gr.Markdown("残差作业会校验三份文件的道数、采样布局和全部道头一致，再按同一 line 对齐显示。")
            with gr.Row():
                sort_keys = gr.Textbox(label="排序键（逗号分隔，1-based）", value="9-12,189-192")
                gkey = gr.Textbox(label="抽道集键", value="9-12,189-192")
                btn_scan = gr.Button("扫描键值")
            values_box = gr.CheckboxGroup(choices=[], value=[], label="键值（勾选要抽取的）")
            scan_info = gr.Markdown("")
            with gr.Row():
                clip = gr.Number(label="预览 clip 分位数 (%)（仅界面显示/画框参照）",
                                 value=99.0, minimum=50, maximum=100)
                aug_lo = gr.Number(label="增强 clip 下限 (%)", value=90.0, minimum=1,
                                   maximum=100)
                aug_hi = gr.Number(label="增强 clip 上限 (%)", value=99.9, minimum=1,
                                   maximum=100)
                title = gr.Textbox(label="作业标题（可空，默认=文件名）")
            with gr.Row():
                min_traces = gr.Number(label="道数下限（道数 < 该值的道集不参与标注，0=不限）",
                                       value=20, minimum=0, precision=0)
                decimate_n = gr.Number(
                    label="抽稀间隔 N（每 N 个道集保留 1 个，按抽取顺序等间隔删；0/1=不抽稀）",
                    value=1, minimum=0, precision=0)
                build_hint = gr.Markdown(f"保存标注时每张道集将随机采 **{N_AUG} 个不同 clip 值**各渲一张导出图")
            btn_create = gr.Button("创建作业", variant="primary")
            build_info = gr.Markdown("")

        # ---------- ② 作业与标注（全体可见） ----------
        with gr.Accordion("② 作业与标注", open=True):
            with gr.Row():
                job_dd = gr.Dropdown(choices=[], multiselect=True, scale=3,
                                     label="作业（可多选：从选中的作业池里随机抽道集）")
                btn_claim = gr.Button("领取下一张", variant="primary", scale=1)
                btn_skip = gr.Button("⏭ 跳过此张", scale=1)
                btn_release = gr.Button("归还此张", scale=1)
                btn_refresh = gr.Button("刷新", scale=1)
            with gr.Row():
                mine_dd = gr.Dropdown(choices=[], label="我标注的（选择以重开编辑）", scale=3)
                btn_inherit = gr.Button("⧉ 继承最近已标注", scale=1)
            anno_info = gr.Markdown("请选择作业（选中即自动领取第一张）", elem_id="anno_info")
            # 本用户的标注总量（跨全部作业，不区分作业）——常驻显示，随每次标注刷新
            my_total_md = gr.Markdown(_my_total_text(_user(None) or ""))
            with gr.Row(elem_id="display-controls"):
                display_clip = gr.Slider(
                    minimum=DISPLAY_CLIP_MIN, maximum=DISPLAY_CLIP_MAX,
                    value=99.0, step=DISPLAY_CLIP_STEP,
                    label="显示 clip 分位数", scale=0, min_width=280,
                    elem_id="preview-clip-slider")
                btn_residual_prev = gr.Button("上一个视图（A / ←）", elem_id="btn-residual-prev", scale=0)
                btn_residual_next = gr.Button("下一个视图（D / →）", elem_id="btn-residual-next", scale=0)
            radios = []
            box_statuses = []
            box_buttons = []   # (feature_key, 清除按钮)
            filter_ui = None   # 面波区「⚙ 滤波」面板（见 FILTER_FEAT_KEY）
            # 题序固定：1..N = 各选择题（键盘数字键依次作答），N+1.. = 各拉框项（鼠标 Ctrl）
            with gr.Row(elem_id="anno_body"):
                with gr.Column(scale=3, elem_id="anno_imgcol", min_width=0):
                    cur_img = gr.Image(
                        label="当前道集（按住 Ctrl 点击多个点，松开 Ctrl 完成不规则包络）",
                        type="filepath", height=640)
                with gr.Column(scale=2, elem_id="anno_featcol", min_width=0):
                    sentence = gr.Textbox(label="句子预览", interactive=False, lines=3)
                    # —— 选择题：单列自上而下 = 题号顺序，当前题会被 JS 高亮 ——
                    # 每题包一层 id=q-<key>：前端按 id 精确分组（不靠 radio 的 name 属性）
                    question_no = {
                        "gather_type": 1, "abnormal_amplitude": 2,
                        "noise_type": 2, "aliasing_noise": 3,
                        "denoise_quality": 3, "surface_wave": 4,
                        "near_shot_noise": 5,
                    }
                    for i, feat in enumerate(feats):
                        with gr.Column(elem_id=f"q-{feat.key}"):
                            choices = [fmt_option(o) for o in feat.options]
                            label = f"{question_no.get(feat.key, i + 1)}. {feat.name}"
                            if feat.input == "checkbox":
                                radios.append(gr.CheckboxGroup(choices=choices, label=label,
                                                             value=[], interactive=(feat.name != "集合类型")))
                            else:
                                radios.append(gr.Radio(choices=choices, label=label, value=None,
                                                        interactive=(feat.name != "集合类型")))
                    # —— 拉框项统一放在所有选择题之后 ——
                    for bi, feat in enumerate(bbox_feats()):
                        with gr.Column(elem_id=f"bx-{feat.key}"):
                            gr.Markdown(f"**{feat.name} 包络**"
                                        f"（按住 Ctrl 依次点击多个点，松开 Ctrl 完成）")
                            with gr.Row():
                                btn_clear = gr.Button("✕ 清除此包络", size="sm",
                                                      elem_id=f"cb-{feat.key}")
                            box_statuses.append(gr.Markdown("未画包络"))
                            box_buttons.append((feat.key, btn_clear))
                            if feat.key == FILTER_FEAT_KEY:
                                # 四角频率 + 单键开关：点一次应用、再点一次还原。
                                # 参数按作业记忆，同作业内统一（见 toggle_filter）。
                                with gr.Column(elem_id="filter_panel"):
                                    with gr.Row():
                                        ff1 = gr.Number(label="f1 低截 (Hz)", value=8.0,
                                                        minimum=0, scale=1)
                                        ff2 = gr.Number(label="f2 低通 (Hz)", value=9.0,
                                                        minimum=0, scale=1)
                                        ff3 = gr.Number(label="f3 高通 (Hz)", value=120.0,
                                                        minimum=0, scale=1)
                                        ff4 = gr.Number(label="f4 高截 (Hz)", value=121.0,
                                                        minimum=0, scale=1)
                                    btn_filter = gr.Button(
                                        "⚙ 滤波（点一次应用，再点一次还原）", size="sm",
                                        elem_id=f"fb-{feat.key}")
                                filter_ui = {"key": feat.key,
                                             "toggle": btn_filter,
                                             "nums": (ff1, ff2, ff3, ff4)}
            with gr.Row():
                btn_save = gr.Button("保存并释放（按 Enter 下一张）", variant="primary",
                                     elem_id="btn-save", scale=3)
                # 返回上一张：连续往回翻修正已标注的记录，改完保存即覆盖原结果
                btn_back = gr.Button("↩ 返回上一张（修正已标注的）", scale=2)

        # ---------- ③ 标注量 / 作业进度（全体可见：管理员与标注者都能看） ----------
        with gr.Accordion("③ 标注量 / 作业进度（每 5 秒自动刷新）", open=True):
            mgr_progress = gr.HTML(jobs_progress_html(JM.list_jobs()))
            gr.Markdown("表内「已完成 / 总数」为该作业的道集标注量；进度条按比例填充。")
            # 定时器放在**全体可见**的这一节里：若放在仅管理员可见区，标注者那边可能不触发
            progress_timer = gr.Timer(value=5)

        # ---------- ④ 管理（仅管理员可见） ----------
        with gr.Accordion("④ 管理（仅管理员）", open=False, visible=False) as admin_mgr:
            with gr.Row():
                mgr_job_dd = gr.Dropdown(choices=[], label="作业", scale=3)
                btn_resync = gr.Button("全部重传", scale=1)
                btn_delete = gr.Button("🗑 删除作业", scale=1)
            resync_info = gr.Markdown("")
            del_info = gr.Markdown("")
            with gr.Row():
                del_job_dd = gr.Dropdown(choices=[], label="已删除作业（本地保留，可恢复）",
                                         scale=3)
                btn_restore = gr.Button("♻ 恢复此作业", scale=1)
            gr.Markdown("**各用户标注总量**（跨全部作业，每 5 秒刷新）")
            staff_progress = gr.HTML(staff_counts_html(_staff_rows()))
            gr.Markdown("**账号管理**：编辑项目根目录 `users.yaml`（增删用户/改角色），"
                        "下一次登录即生效，无需重启服务。")

        # ---------------- 事件 ----------------
        demo.load(init_page, None,
                  [job_dd, admin_build, admin_mgr, mgr_job_dd, who, del_job_dd,
                   mgr_progress, my_total_md])
        btn_logout.click(logout_click, None, who)

        def _job_type_visibility(t):
            residual = str(t) == "残差"
            return gr.update(visible=not residual), gr.update(visible=residual)

        job_type.change(_job_type_visibility, job_type, [file_path, residual_files])
        btn_load.click(load_job_source,
                       [job_type, file_path, residual_before, endian], file_info)
        btn_scan.click(scan_job_values,
                       [job_type, file_path, residual_before, endian, gkey],
                       [values_box, scan_info])
        btn_create.click(create_job_ui,
                         [file_path, endian, sort_keys, gkey, values_box, clip,
                          aug_lo, aug_hi, title, min_traces, decimate_n,
                          job_type, residual_before, residual_after, residual_noise],
                         [build_info, job_dd, mgr_job_dd])

        # 尾部 4 个是滤波参数输入框（按作业回填，见 _filter_fields）
        filter_nums = list(filter_ui["nums"]) if filter_ui is not None else []
        anno_outputs = [cur_img, anno_info, sentence, *radios, *box_statuses, *filter_nums,
                        display_clip, my_total_md]
        job_dd.change(select_jobs, job_dd, anno_outputs + [mine_dd])
        mine_dd.change(reopen_mine, [job_dd, mine_dd], anno_outputs)
        btn_claim.click(claim_next, job_dd, anno_outputs)
        btn_skip.click(skip_current, None, anno_outputs)
        btn_release.click(release_current, None, anno_outputs)
        btn_refresh.click(refresh_page, None, [job_dd, mine_dd, *anno_outputs])
        btn_save.click(save_anno, radios, anno_outputs)
        btn_back.click(back_previous, None, anno_outputs)
        btn_residual_prev.click(cycle_residual_previous, None, anno_outputs)
        btn_residual_next.click(cycle_residual_next, None, anno_outputs)
        display_clip.change(update_display_clip, display_clip, cur_img,
                            show_progress="hidden")
        # outputs 带上框状态：继承是程序化改选项，框状态标记是前端唯一能拿到的"该刷新了"信号
        # （详见 inherit_previous 的 docstring）
        btn_inherit.click(inherit_previous, None, [sentence, *radios, *box_statuses])

        for r in radios:
            # 连同框状态一起刷新：Ctrl 目标取决于选项（选「不存在」就不需要框）
            r.change(on_radio_change, radios, [sentence, *box_statuses])
        for key, btn_clear in box_buttons:
            # 清除框只动图/提示/框状态（不碰 radio 与参数字段）
            btn_clear.click(_box_clear(key), None, [cur_img, anno_info, *box_statuses])

        # 面波区「⚙ 滤波」单键开关：输出与 anno_outputs 同序（含尾部 4 个参数字段）
        if filter_ui is not None:
            filter_ui["toggle"].click(toggle_filter, filter_nums, anno_outputs)

        btn_resync.click(resync_job, mgr_job_dd, resync_info)
        btn_delete.click(delete_job_click, mgr_job_dd,
                         [btn_delete, del_info, mgr_job_dd, del_job_dd,
                          mgr_progress, job_dd, staff_progress])
        btn_restore.click(restore_job_click, del_job_dd,
                          [del_info, mgr_job_dd, del_job_dd, mgr_progress, job_dd,
                           staff_progress])

        # 标注量/作业进度：每 5 秒自动刷新（③ 区全体可见，因此标注者也能看到最新标注量）；
        # 顺带刷新管理区的删除确认按钮/提示（对标注者是不可见的组件，更新无副作用）
        progress_timer.tick(refresh_jobs_progress,
                            outputs=[mgr_progress, btn_delete, del_info, staff_progress],
                            api_name=False, show_progress="hidden")
    demo.queue(default_concurrency_limit=16)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--share", action="store_true",
                    help="额外开启 Gradio 公网隧道：生成一个 https://*.gradio.live 地址供校园网/外网访问；"
                         "局域网 0.0.0.0:port 不受影响，两者同时可用（需本机能出网到 HuggingFace）")
    args = ap.parse_args()
    demo = build_app()
    if args.share:
        print("注：--share 公网隧道在此版本不启用（拖动框选依赖自定义接口），仍监听局域网端口。")
    import uvicorn

    app = FastAPI()

    @app.post("/api/anno_box")
    async def anno_box(req: Request):
        try:
            p = await req.json()
        except Exception:
            return JSONResponse({"ok": False, "err": "bad json"}, status_code=400)
        user = str(p.get("user") or "").strip()
        if not user or ACC.role(user) is None:
            return JSONResponse({"ok": False, "err": "unknown user"}, status_code=401)
        try:
            regions = p.get("regions")
            if regions is not None:
                ok, msg = apply_drag_regions(user, str(p.get("key") or ""), regions)
                return {"ok": ok, "msg": msg}
            points = p.get("points")
            if not points:
                points = [[float(p.get("x0")), float(p.get("y0"))],
                          [float(p.get("x1")), float(p.get("y0"))],
                          [float(p.get("x1")), float(p.get("y1"))],
                          [float(p.get("x0")), float(p.get("y1"))]]
            ok, msg = apply_drag_polygon(user, str(p.get("key") or ""), points)
        except Exception as e:                       # noqa: BLE001 —— 统一转 400 提示
            return JSONResponse({"ok": False, "err": str(e)}, status_code=400)
        return {"ok": ok, "msg": msg}

    @app.post("/api/anno_box_clear")
    async def anno_box_clear(req: Request):
        """右键清框：直接清服务端该特征的框，不依赖 Gradio 按钮的 DOM。"""
        try:
            p = await req.json()
        except Exception:
            return JSONResponse({"ok": False, "err": "bad json"}, status_code=400)
        user = str(p.get("user") or "").strip()
        if not user or ACC.role(user) is None:
            return JSONResponse({"ok": False, "err": "unknown user"}, status_code=401)
        ok, msg = clear_drag_box(user, str(p.get("key") or ""))
        return {"ok": ok, "msg": msg}

    @app.get("/api/boxes")
    async def anno_boxes(user: str = ""):
        """读取当前道集各 bbox 特征的框（自然像素 xyxy），供前端初始化/重开编辑。"""
        user = (user or "").strip()
        if not user or ACC.role(user) is None:
            return JSONResponse({"ok": False, "err": "unknown user"}, status_code=401)
        return {"ok": True, **_boxes_payload(user)}

    gr.mount_gradio_app(app, demo, path="/", auth=ACC.authenticate, css=ANNO_CSS,
                        head=ANNO_JS,
                        auth_message="多用户地震标注服务：使用 users.yaml 中的账号登录；"
                                     "登录后点右上角「登出 / 切换账号」即可换账号。")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
