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
import os
import tempfile
import threading
import time

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from web_core import fmt_option, jobs_progress_html, label_of, pixel_box_to_data
from labels import LabelConfig
from users import Accounts
from jobmanager import JobManager, JobError, N_AUG
from cloudsync import Uploader, load_config
from gather import parse_ranges, gather_values, ranges_label
from segy_reader import SegyReader, SegyReadError
from imaging import overlay_boxes

# matplotlib 默认配置目录可能不可写，导入前指到可写目录
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplcfg_"))

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_config.yaml")
USERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.yaml")
COS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cos_config.yaml")
JOBS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
os.makedirs(JOBS_ROOT, exist_ok=True)

# 启动引导：账号文件不存在时写入默认 admin（boss/boss123）
Accounts.ensure_default(USERS_PATH)

CFG = LabelConfig(CONFIG_PATH)
ACC = Accounts(USERS_PATH)
JM = JobManager(JOBS_ROOT, CFG)
UP = Uploader(load_config(COS_PATH))
UP.start()
JM.load_all()

# 每用户工作态：{username: {"job_id","gid","partial","boxes","box_mode","pending_corner"}}
#   partial: {特征名: 选项label}（未保存的 Radio 选择）；boxes: {特征key: box|None}（未保存的框）
WORK: dict[str, dict] = {}
WORK_LOCK = threading.RLock()


def wstate(user: str) -> dict:
    with WORK_LOCK:
        return WORK.setdefault(user, {})


def _user(request) -> str:
    return (request.username or "") if request is not None else ""


def _is_admin(user: str) -> bool:
    return bool(user) and ACC.role(user) == "admin"


def bbox_feats():
    return [f for f in CFG.features if f.bbox]


def _job_choices(user: str) -> list[str]:
    """标注区作业下拉候选：admin 看全部；标注者看 open + 自己标过的。"""
    if _is_admin(user):
        return [j["job_id"] for j in JM.list_jobs()]
    return [j["job_id"] for j in JM.list_jobs()
            if j["state"] == "open" or JM.mine(j["job_id"], user)]


def _mine_choices(user: str, job_id: str | None) -> list[str]:
    if not job_id:
        return []
    try:
        return [r["gather_id"] for r in JM.mine(job_id, user)]
    except JobError:
        return []


# ----------------------------------------------------------------------
# 标注区纯逻辑（per-user 状态）
# ----------------------------------------------------------------------
def _radio_value(job, gid: str, feat, st: dict) -> str | None:
    rec = job.store.get(gid)
    by_key = (rec.get("labels") or {}) if rec else {}
    partial = st.get("partial") or {}
    want = by_key.get(feat.key) or partial.get(feat.name)
    for opt in feat.options:
        if opt.label == want:
            return fmt_option(opt)
    return None


def _radio_updates(job, gid: str, st: dict) -> list:
    return [gr.Radio(value=_radio_value(job, gid, f, st)) for f in CFG.features]


def _selection(job, gid: str, st: dict) -> dict:
    sel = {}
    for f in CFG.features:
        v = _radio_value(job, gid, f, st)
        if v:
            sel[f.name] = label_of(v)
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


def render_display(request: gr.Request) -> str:
    """当前道集显示图：显示图（.cache，干净）+ 已画框叠加临时副本。

    只显示/领取时绝不写导出 images/：导出图仅在「保存标注」时由 JobManager.save
    生成，否则会出现标一张却多一张下一道集孤立图、图片与 labels.jsonl 对不上的问题。
    """
    user = _user(request)
    st = wstate(user)
    job = JM.get(st["job_id"])
    gid = st["gid"]
    base = job.display_image(gid)
    overlays = []
    boxes = _current_boxes(job, gid, st)
    for feat in bbox_feats():
        b = boxes.get(feat.key)
        if b:
            overlays.append((b["xyxy"], feat.bbox_color, feat.name))
    if not overlays:
        return base
    tmp = os.path.join(tempfile.mkdtemp(prefix="gather_"), "cur_boxed.png")
    return overlay_boxes(base, overlays, tmp)


def box_statuses_of(request: gr.Request) -> list[str]:
    user = _user(request)
    st = wstate(user)
    job = JM.get(st["job_id"])
    boxes = _current_boxes(job, st["gid"], st)
    outs = []
    for feat in bbox_feats():
        if st.get("box_mode") == feat.key:
            outs.append("👉 框选中：请在图上按住鼠标拖动，松开即确定")
            continue
        b = boxes.get(feat.key)
        if b:
            (x0, y0, x1, y1), (t0, t1), (s0, s1) = b["xyxy"], b["traces"], b["samples"]
            outs.append(f"✔ 已画框：像素[{x0},{y0},{x1},{y1}]（左上→右下），道 {t0} 至 {t1}，采样 {s0} 至 {s1}")
        else:
            outs.append("未画框")
    return outs


def show_current(request: gr.Request):
    """返回当前道集的所有显示组件值（img, info, sentence, *radios, *box_statuses）。"""
    user = _user(request)
    st = wstate(user)
    job = JM.get(st["job_id"])
    gid = st["gid"]
    img = render_display(request)
    n_labeled, total = JM.progress(st["job_id"])
    meta = job.gather_meta(gid)
    rec = job.store.get(gid)
    state = "已标注 ✔" if rec else "未标注"
    info = (f"**道集 {gid}** | 键 {meta['key']} = {meta['value_text']}"
            f" | {meta['n_traces']} 道 | {state} | 作业总进度 **{n_labeled}/{total}**")
    sel = _selection(job, gid, st)
    sentence = CFG.render_sentence(sel)
    return (img, info, sentence, *_radio_updates(job, gid, st), *box_statuses_of(request))


def _anno_idle(msg: str = ""):
    """无当前道集时的标注区空态输出（img 不动、radio 不动、状态占位）。"""
    return (gr.skip(), msg, "", *([gr.Radio()] * len(CFG.features)),
            *(["…"] * len(bbox_feats())))


def refresh_anno(request: gr.Request):
    user = _user(request)
    st = wstate(user)
    if not st.get("job_id") or not st.get("gid"):
        return _anno_idle("请选择作业后点「领取下一张」")
    return show_current(request)


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
                  selected, clip, aug_lo, aug_hi, title, min_traces):
    """创建作业：抽一次道集（可按道数下限滤去过少者）、写 job.json、注册并刷新下拉。

    clip: 预览 clip（仅界面显示/画框参照）；aug_lo/aug_hi: 保存时随机采样的 clip 范围。
    """
    user = _user(request)
    if not _is_admin(user):
        return "❌ 仅管理员可创建作业", gr.update(), gr.update()
    path = (path or "").strip()
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
                            aug_lo=aug_lo, aug_hi=aug_hi)
    except (JobError, SegyReadError) as e:
        return f"❌ {e}", gr.update(), gr.update()
    n = len(job.gather_ids())
    dropped = max(0, len(values) - n) if min_traces > 0 else 0
    note = f"（已滤去 {dropped} 个道数 < {min_traces} 的道集）" if dropped else ""
    job_choices = _job_choices(user)
    mgr_choices = [j["job_id"] for j in JM.list_jobs()]
    return (f"✔ 已创建作业 {job.job_id}（键 {ranges_label(gkeys)}）｜道集数 {n}{note}"
            f"｜保存时每张增强 {N_AUG} 个随机 clip",
            gr.update(choices=job_choices, value=job.job_id),
            gr.update(choices=mgr_choices, value=job.job_id))


# ----------------------------------------------------------------------
# 标注流程 handler
# ----------------------------------------------------------------------
def open_job_for(request: gr.Request, job_id: str):
    user = _user(request)
    st = wstate(user)
    if not job_id:
        return (*_anno_idle("⚠️ 请选择作业"), gr.update(choices=[], value=None))
    st["job_id"] = job_id
    st.pop("gid", None)
    st.pop("partial", None)
    st.pop("boxes", None)
    st["box_mode"] = None
    st["pending_corner"] = None
    return (*refresh_anno(request), gr.update(choices=_mine_choices(user, job_id), value=None))


def claim_next(request: gr.Request, job_id: str):
    user = _user(request)
    st = wstate(user)
    if not job_id:
        return _anno_idle("⚠️ 请先在「作业」下拉选择作业")
    # 已持有本作业的未保存任务：不静默续领同一张，明确指引如何进入下一张
    if st.get("job_id") == job_id and st.get("gid"):
        gid = st["gid"]
        if JM.record(job_id, gid) is None:      # 该张尚未保存
            cur = show_current(request)
            return (cur[0],
                    f"⚠️ 正在标注「{gid}」（尚未保存）。要进入下一张，请先点"
                    "「保存并释放」完成本张；要放弃本张，请先点「归还此张」。",
                    cur[2], *cur[3:])
    st["job_id"] = job_id
    out = JM.claim(job_id, user)
    if out["gid"]:
        st["gid"] = out["gid"]
        st["partial"], st["boxes"] = {}, {}
        st["box_mode"] = None
        st["pending_corner"] = None
        return show_current(request)
    return _anno_idle("⚠️ " + (out.get("reason") or "无任务"))


def on_radio_change(request: gr.Request, *radio_values):
    user = _user(request)
    st = wstate(user)
    if not st.get("gid") or not st.get("job_id"):
        return ""
    JM.renew(st["job_id"], user)
    sel = {}
    for feat, v in zip(CFG.features, radio_values):
        if v:
            sel[feat.name] = label_of(v)
    st["partial"] = sel
    return CFG.render_sentence(sel)


def enter_box_mode(request: gr.Request, feat_key: str):
    user = _user(request)
    st = wstate(user)
    if not st.get("gid") or not st.get("job_id"):
        return ("⚠️ 请先领取道集", *(["…"] * len(bbox_feats())))
    JM.renew(st["job_id"], user)
    st["box_mode"] = feat_key
    st["pending_corner"] = None
    feat = next(f for f in bbox_feats() if f.key == feat_key)
    return (f"👉 正在框选「{feat.name}」：请在图上按住鼠标拖动，松开即确定", *box_statuses_of(request))


def clear_box(request: gr.Request, feat_key: str):
    user = _user(request)
    st = wstate(user)
    if not st.get("gid") or not st.get("job_id"):
        return (gr.skip(), "⚠️ 请先领取道集", *(["…"] * len(bbox_feats())))
    JM.renew(st["job_id"], user)
    if st.get("box_mode") == feat_key:
        st["box_mode"] = None
        st["pending_corner"] = None
    job = JM.get(st["job_id"])
    boxes = _current_boxes(job, st["gid"], st)
    boxes[feat_key] = None
    st["boxes"] = boxes
    feat = next(f for f in bbox_feats() if f.key == feat_key)
    return (render_display(request), f"已清除「{feat.name}」的框", *box_statuses_of(request))


def apply_drag_box(user: str, key: str, x0, y0, x1, y1) -> tuple[bool, str]:
    """把「拖动框选」的最终矩形写入该用户当前道集的 boxes（真实像素坐标 512×1024）。

    由自定义接口 /api/anno_box 调用（前端鼠标松开即提交）；保存/校验继续读 st['boxes']。
    """
    if key not in [f.key for f in bbox_feats()]:
        return False, f"非框选特征: {key}"
    with WORK_LOCK:
        st = WORK.setdefault(user, {})
        if not st.get("job_id") or not st.get("gid"):
            return False, "没有正在标注的道集"
        job = JM.get(st["job_id"])
        g = job.gather(st["gid"])
    box = pixel_box_to_data((float(x0), float(y0)), (float(x1), float(y1)),
                            g.n_traces, job.ns)
    with WORK_LOCK:
        st["boxes"] = dict(st.get("boxes") or {})
        st["boxes"][key] = box
        st["box_mode"] = None
        st["pending_corner"] = None
    return True, key


def _boxes_payload(user: str) -> dict:
    """当前道集各 bbox 特征的框（自然像素 xyxy，与导出图 512×1024 同系）。

    供 GET /api/boxes：前端在显示图/重开时拉取本道集已有框，用于叠加可编辑手柄。
    优先读本会话 st['boxes']；重开已保存记录时回退到 labels 记录 regions。
    """
    st = wstate(user)
    out = {}
    if st.get("job_id") and st.get("gid"):
        try:
            job = JM.get(st["job_id"])
        except JobError:
            return out
        boxes = _current_boxes(job, st["gid"], st)
        for f in bbox_feats():
            b = boxes.get(f.key)
            if b and b.get("xyxy"):
                x0, y0, x1, y1 = b["xyxy"]
                out[f.key] = {"x0": int(x0), "y0": int(y0),
                              "x1": int(x1), "y1": int(y1)}
    return out


def save_anno(request: gr.Request, *radio_values):
    user = _user(request)
    st = wstate(user)
    if not st.get("gid") or not st.get("job_id"):
        return _anno_idle("⚠️ 没有正在标注的道集")
    job_id = st["job_id"]
    gid = st["gid"]
    sel = {}
    for feat, v in zip(CFG.features, radio_values):
        if v:
            sel[feat.name] = label_of(v)
    st["partial"] = sel
    # 空闲超 TTL 导致租约被清：renew 只续「仍有效」的租约，返回 None。
    # 区分两种无租约场景：
    #   (a) 该道集尚未标注（首次领取后闲挂超时被清）→ 尝试重新领取同 gid；
    #   (b) 该道集已有记录（重开自己标过的记录，本就无租约）→ 直接走 owner 归属保存。
    if JM.renew(job_id, user) is None and JM.record(job_id, gid) is None:
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
    # 云上传（异步，失败绝不阻塞/回滚标注）：本次落盘的 N_AUG 张增强图 + labels.jsonl
    rels = [a["image_path"] for a in aug_recs if a.get("image_path")]
    rels.append("labels.jsonl")
    UP.enqueue(job_id, JM.get(job_id).output_dir, rels)
    n_labeled, total = JM.progress(job_id)
    st.pop("gid", None)
    st.pop("partial", None)
    st.pop("boxes", None)
    st["box_mode"] = None
    st["pending_corner"] = None
    # 保存并释放 → 自动领取下一张（保持原「保存并下一张」的连续标注体验）
    nxt = JM.claim(job_id, user)
    if not nxt.get("gid"):
        if n_labeled >= total:
            return (gr.update(value=None),
                    f"🎉 本作业已全部标注（{n_labeled}/{total}）",
                    "", *([gr.Radio(value=None)] * len(CFG.features)),
                    *(["…"] * len(bbox_feats())))
        return _anno_idle("⚠️ " + (nxt.get("reason") or "暂无剩余可领取"))
    st["gid"] = nxt["gid"]
    st["partial"], st["boxes"] = {}, {}
    st["box_mode"] = None
    st["pending_corner"] = None
    cur = show_current(request)
    return (cur[0],
            f"✔ 已保存「{rec['gather_id']}」× {len(aug_recs)} 张增强图 | 作业进度 "
            f"{n_labeled}/{total} | 已自动领取下一张",
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
    st["box_mode"] = None
    st["pending_corner"] = None
    return (gr.update(value=None), "已归还当前道集（回池，可被他人领取）", "",
            *([gr.Radio(value=None)] * len(CFG.features)), *(["…"] * len(bbox_feats())))


def skip_current(request: gr.Request):
    """「跳过此张」：当前道集放回池（不写记录、不计完成），并自动领取下一张。"""
    user = _user(request)
    st = wstate(user)
    if not st.get("job_id") or not st.get("gid"):
        return _anno_idle("⚠️ 没有正在标注的道集")
    job_id = st["job_id"]
    out = JM.skip(job_id, user)
    if out.get("gid"):
        st["gid"] = out["gid"]
        st["partial"], st["boxes"] = {}, {}
        st["box_mode"] = None
        st["pending_corner"] = None
        cur = show_current(request)
        skipped = out.get("skipped")
        return (cur[0],
                f"⏭ 已跳过「{skipped or '当前道集'}」回池（未保存）｜自动领取下一张",
                cur[2], *cur[3:])
    n_labeled, total = JM.progress(job_id)
    if n_labeled >= total:
        return (gr.update(value=None),
                f"🎉 本作业已全部标注（{n_labeled}/{total}）",
                "", *([gr.Radio(value=None)] * len(CFG.features)),
                *(["…"] * len(bbox_feats())))
    return _anno_idle("⚠️ " + (out.get("reason") or "暂无剩余可领取"))


def refresh_jobs_progress():
    """管理区定时刷新：返回(进度表HTML, 删除按钮, 删除提示)。

    顺带清理“删除确认”超时态：约 3 秒未再点则自动复原按钮/提示。
    """
    html = jobs_progress_html(JM.list_jobs())
    if _PENDING_DELETE["job_id"] and time.time() - _PENDING_DELETE["at"] > 3.0:
        _cancel_delete()
        return (html, gr.update(value="🗑 删除作业"), gr.update(value=""))
    return (html, gr.update(), gr.update())


def reopen_mine(request: gr.Request, job_id: str, gid: str):
    user = _user(request)
    st = wstate(user)
    if not job_id or not gid:
        return _anno_idle("⚠️ 请选择要重开的记录")
    # 服务端校验：标注者只能重开自己标过的记录；admin 可重开任意记录
    if not _is_admin(user):
        try:
            mine_ids = {r["gather_id"] for r in JM.mine(job_id, user)}
        except JobError:
            mine_ids = set()
        if gid not in mine_ids:
            return _anno_idle("⚠️ 只能重开自己标注过的记录")
    st["job_id"] = job_id
    st["gid"] = gid
    st.pop("partial", None)   # 回退到已存记录标签
    st.pop("boxes", None)     # 回退到已存记录框
    st["box_mode"] = None
    st["pending_corner"] = None
    JM.renew(job_id, user)
    return show_current(request)


def inherit_previous(request: gr.Request):
    """把本用户最近一条【已标注】记录的类别选项填充到当前道集（仅填充不保存，框不继承）。"""
    user = _user(request)
    st = wstate(user)
    if not st.get("gid") or not st.get("job_id"):
        return ("⚠️ 请先领取道集", *([gr.Radio()] * len(CFG.features)))
    src = None
    for rec in reversed(JM.mine(st["job_id"], user)):
        if rec.get("gather_id") != st["gid"] and rec.get("labels"):
            src = rec
            break
    if src is None:
        cur_sel = _selection(JM.get(st["job_id"]), st["gid"], st)
        return ("⚠️ 没有可继承的已标注记录 | " + CFG.render_sentence(cur_sel),
                *([gr.Radio()] * len(CFG.features)))
    by_key = src["labels"]
    sel = {f.name: by_key[f.key] for f in CFG.features if by_key.get(f.key)}
    outs = []
    for f in CFG.features:
        want = sel.get(f.name)
        outs.append(gr.Radio(value=next((fmt_option(o) for o in f.options if o.label == want), None)))
    st["partial"] = sel
    return ("已继承最近已标注记录（" + src["gather_id"] + "）的选项；框不继承，需另行画框："
            + CFG.render_sentence(sel), *outs)


def refresh_page(request: gr.Request):
    """刷新作业下拉 + 我标注的 + 当前显示。"""
    user = _user(request)
    st = wstate(user)
    job_id = st.get("job_id")
    job_choices = _job_choices(user)
    if job_id not in job_choices:
        st.pop("gid", None)
        st.pop("partial", None)
        st.pop("boxes", None)
        st["box_mode"] = None
        st["pending_corner"] = None
        job_id = None
    mine_choices = _mine_choices(user, job_id) if job_id else []
    dd = gr.update(choices=job_choices, value=job_id)
    mine = gr.update(choices=mine_choices, value=None)
    if st.get("gid"):
        anno = show_current(request)
    else:
        anno = _anno_idle("请选择作业后点「领取下一张」")
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
            gr.update(choices=_job_choices(user), value=None))


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
                gr.update(choices=_job_choices(user), value=None))
    job_id = (job_id or "").strip()
    if not job_id:
        return ("⚠️ 请先选择要恢复的作业",
                gr.update(choices=_mgmt_choices(), value=None),
                gr.update(choices=_deleted_choices(), value=None),
                jobs_progress_html(JM.list_jobs()),
                gr.update(choices=_job_choices(user), value=None))
    title = _job_title(job_id)
    try:
        JM.restore_job(job_id)
    except JobError as e:
        return (f"❌ {e}",
                gr.update(choices=_mgmt_choices(), value=None),
                gr.update(choices=_deleted_choices(), value=None),
                jobs_progress_html(JM.list_jobs()),
                gr.update(choices=_job_choices(user), value=None))
    _cancel_delete()
    return (f"✔ 已恢复「{title}」为进行中，回到作业列表",
            gr.update(choices=_mgmt_choices(), value=None),
            gr.update(choices=_deleted_choices(), value=None),
            jobs_progress_html(JM.list_jobs()),
            gr.update(choices=_job_choices(user), value=None))


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
        st["box_mode"] = None
        st["pending_corner"] = None
    if not notes:
        notes.append("当前无在标注的道集")
    return gr.update(value="  ".join(notes) + "　**[退出登录并切换账号](/logout?all_session=false)**")


def init_page(request: gr.Request):
    """页面加载：按角色决定可见区、填充作业下拉，并显示当前账号。"""
    user = _user(request)
    admin = _is_admin(user)
    mgr_choices = [j["job_id"] for j in JM.list_jobs()] if admin else []
    del_choices = [j["job_id"] for j in JM.deleted_jobs()] if admin else []
    return (gr.update(choices=_job_choices(user), value=None),
            gr.update(visible=admin),      # 建作业区
            gr.update(visible=admin),      # 管理区
            gr.update(choices=mgr_choices, value=None),
            gr.update(value=_who_md(user)),   # 账号栏
            gr.update(choices=del_choices, value=None))   # 已删除作业（可恢复）


def _box_enter(key: str):
    def handler(request: gr.Request):
        return enter_box_mode(request, key)
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
"""

# 拖动框选：bbox 特征配置（key/名称/颜色），供前端注入 JS 使用
_DRAG_BBOX = [[f.key, f.name, f.bbox_color] for f in CFG.features if f.bbox]

ANNO_JS = """
<style>
#anno_imgcol .image-container { position: relative; }
#anno_imgcol .image-container img { cursor: crosshair; }
#anno_drag_canvas {
    position: absolute; pointer-events: none; z-index: 50;
    image-rendering: auto; border: 0; touch-action: none;
}
</style>
<script>
(function(){
  const BBOX = __DRAG_BBOX__;
  const KEY2COLOR = {}; BBOX.forEach(function(b){ KEY2COLOR[b[0]] = b[2] || '#ff0000'; });
  let boxes = {};          // key -> {x0,y0,x1,y1}，坐标 = 图片自然像素（导出图固定 512×1024）
  let drawKey = null;      // 点了「▣框选」等待新画/重画的特征；null = 可拖动/缩放已有框
  let grab = null;         // 当前手势 {type:draw|move|resize, key, ...}
  let canvas = null, lastSrc = '';
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
    return {x: b.x0/im.naturalWidth*r.width,  y: b.y0/im.naturalHeight*r.height,
            w: (b.x1-b.x0)/im.naturalWidth*r.width, h: (b.y1-b.y0)/im.naturalHeight*r.height};
  }
  function copyBox(b){ return {x0:b.x0, y0:b.y0, x1:b.x1, y1:b.y1}; }
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
      var b = boxes[key]; if (!b) continue;
      var s = screenRect(b), col = KEY2COLOR[key] || '#1f6feb';
      c.strokeStyle = col;
      c.lineWidth = (grab && grab.key === key) ? 4 : 3;
      c.strokeRect(s.x, s.y, s.w, s.h);
      var pts = [[s.x,s.y],[s.x+s.w,s.y],[s.x,s.y+s.h],[s.x+s.w,s.y+s.h]];
      for (var i = 0; i < pts.length; i++){
        var px = pts[i][0], py = pts[i][1];
        c.fillStyle = col; c.fillRect(px - hs/2, py - hs/2, hs, hs);
        c.fillStyle = '#ffffff'; c.fillRect(px - hs/4, py - hs/4, hs/2, hs/2);
      }
    }
  }
  function place(){
    var w = document.querySelector('#anno_imgcol .image-container');
    var img = getImg();
    canvas = ensureCanvas();
    if (!canvas) return;
    if (!w || !img){ canvas.style.display = 'none'; return; }
    bindImg(img);
    if (img.currentSrc && img.currentSrc !== lastSrc){   // 换了道集/重开 → 清本地并回填服务器
      lastSrc = img.currentSrc;
      boxes = {}; grab = null; drawKey = null;
      fetchBoxes();
    }
    var cr = w.getBoundingClientRect(), ir = img.getBoundingClientRect();
    canvas.style.display = '';
    canvas.style.left = (ir.left - cr.left) + 'px';
    canvas.style.top  = (ir.top  - cr.top)  + 'px';
    canvas.style.width  = Math.round(ir.width)  + 'px';
    canvas.style.height = Math.round(ir.height) + 'px';
    canvas.width  = Math.round(ir.width);
    canvas.height = Math.round(ir.height);
    canvas._img = img; canvas._rect = ir;
    redraw();
  }
  function fetchBoxes(){
    var u = currentUser(); if (!u) return;
    fetch('/api/boxes?user=' + encodeURIComponent(u), {headers:{'Accept':'application/json'}})
      .then(function(r){ return r.ok ? r.json() : {boxes:{}}; })
      .then(function(d){ if (d && d.boxes){ boxes = d.boxes || {}; if (!grab) redraw(); } })
      .catch(function(){});
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
      var b = boxes[key]; if (!b) continue;
      var s = screenRect(b);
      var corners = {nw:[s.x,s.y], ne:[s.x+s.w,s.y], sw:[s.x,s.y+s.h], se:[s.x+s.w,s.y+s.h]};
      for (var cName in corners){
        var p = corners[cName];
        if (Math.abs(p[0]-x) <= tol && Math.abs(p[1]-y) <= tol) return {type:'resize', key:key, corner:cName};
      }
      if (x >= s.x && x <= s.x+s.w && y >= s.y && y <= s.y+s.h) return {type:'move', key:key};
    }
    return null;
  }
  // ---- 交互（全部以图片局部坐标运算，与绘制同系）----
  function onDown(e){
    if (!canvas || !canvas._img) return;
    var lp = localPt(e); if (!lp) return;
    var nat = localToNat(lp.x, lp.y); if (!nat) return;
    e.preventDefault(); e.stopPropagation();
    if (drawKey){                                   // 拖新框（重画该特征）
      grab = {type:'draw', key:drawKey,
              snapshot: boxes[drawKey] ? copyBox(boxes[drawKey]) : null, down:nat};
      boxes[drawKey] = {x0:nat[0], y0:nat[1], x1:nat[0], y1:nat[1]};
    } else {
      var hit = hitTest(lp.x, lp.y);
      if (!hit) return;                             // 空白区点击不做任何事
      var key = hit.key;
      grab = hit;
      grab.snapshot = boxes[key] ? copyBox(boxes[key]) : null;
      if (hit.type === 'resize'){
        var opp = {nw:'se', se:'nw', ne:'sw', sw:'ne'}[hit.corner];
        grab.anchorNat = natCorner(boxes[key], opp);   // 对角锚点（自然像素，固定）
      } else {
        grab.nat0 = nat;                              // 整体移动起点
      }
    }
    if (e.target.setPointerCapture){ try { e.target.setPointerCapture(e.pointerId); } catch(_){} }
    redraw();
  }
  function onMove(e){
    if (!canvas || !canvas._img) return;
    var lp = localPt(e); if (!lp) return;
    if (!grab){                                     // 悬停：仅更新光标提示
      if (drawKey){ if(e.target.style) e.target.style.cursor = 'crosshair'; return; }
      var h = hitTest(lp.x, lp.y);
      var cur = h ? (h.type === 'move' ? 'move'
                     : (h.corner === 'nw' || h.corner === 'se') ? 'nwse-resize' : 'nesw-resize')
                  : 'crosshair';
      if (e.target.style) e.target.style.cursor = cur;
      return;
    }
    e.preventDefault();
    var nat = localToNat(lp.x, lp.y); if (!nat) return;
    var g = grab, key = g.key, im = canvas._img;
    var b = boxes[key] || {x0:0, y0:0, x1:0, y1:0};
    var W = im.naturalWidth - 1, H = im.naturalHeight - 1;
    if (g.type === 'draw'){
      b.x0 = Math.round(clamp(Math.min(g.down[0], nat[0]), 0, W));
      b.y0 = Math.round(clamp(Math.min(g.down[1], nat[1]), 0, H));
      b.x1 = Math.round(clamp(Math.max(g.down[0], nat[0]), 0, W));
      b.y1 = Math.round(clamp(Math.max(g.down[1], nat[1]), 0, H));
    } else if (g.type === 'move'){
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
    boxes[key] = b;
    redraw();
  }
  function postBox(key, b){
    var u = currentUser(); if (!u) return;
    fetch('/api/anno_box', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({user:u, key:key, x0:b.x0, y0:b.y0, x1:b.x1, y1:b.y1})})
      .catch(function(){});
  }
  function endGesture(e, commit){
    if (!grab) return;
    var g = grab; grab = null;
    if (e.target && e.target.setPointerCapture){ try { e.target.releasePointerCapture(e.pointerId); } catch(_){} }
    var b = boxes[g.key];
    var ok = b && (b.x1 - b.x0) >= MIN && (b.y1 - b.y0) >= MIN;
    if (commit && ok){
      b.x0 = Math.round(b.x0); b.y0 = Math.round(b.y0);
      b.x1 = Math.round(b.x1); b.y1 = Math.round(b.y1);
      boxes[g.key] = b;
      if (g.type === 'draw') drawKey = null;        // 画完即回到编辑态（可拖动/缩放）
      postBox(g.key, b);
    } else {
      if (g.snapshot) boxes[g.key] = g.snapshot;
      else delete boxes[g.key];
      if (g.type !== 'draw') drawKey = null;
    }
    if (e.target && e.target.style) e.target.style.cursor = drawKey ? 'crosshair' : '';
    redraw();
  }
  function onUp(e){ endGesture(e, true); }
  function onCancel(e){ endGesture(e, false); }
  function onLeave(e){ if (e.target && e.target.style) e.target.style.cursor = drawKey ? 'crosshair' : ''; }
  // ---- 按钮联动 ----
  function setDraw(key){ drawKey = key; if (canvas) { redraw(); place(); } }
  function clearKey(key){ drawKey = null; delete boxes[key]; if (canvas) { redraw(); place(); } }
  function setup(){
    document.addEventListener('click', function(e){
      var t = e.target && e.target.closest ? e.target.closest('[id^="bb-"]') : null;
      if (t){ var k = (t.id || '').replace(/^bb-/, ''); if (KEY2COLOR[k] !== undefined) setDraw(k); }
    });
    document.addEventListener('click', function(e){
      var t = e.target && e.target.closest ? e.target.closest('[id^="cb-"]') : null;
      if (t){ var k = (t.id || '').replace(/^cb-/, ''); if (KEY2COLOR[k] !== undefined) clearKey(k); }
    });
    window.addEventListener('resize', function(){ if (!grab) place(); });
    window.addEventListener('scroll', function(){ if (!grab) place(); }, true);
    var root = document.getElementById('anno_imgcol') || document.body;
    if (window.MutationObserver){
      new MutationObserver(function(){ if (!grab) place(); })
        .observe(root, {childList:true, subtree:true, attributes:true, attributeFilter:['src','class']});
    }
    window.addEventListener('load', function(ev){
      if (ev.target && ev.target.tagName === 'IMG' && ev.target.closest && ev.target.closest('#anno_imgcol')) place();
    }, true);
    // 兜底：图片加载/重渲染时序不稳时，周期校正画布位置与 img 事件绑定
    setInterval(function(){ if (!grab){ var im = getImg(); if (im) place(); } }, 900);
    place();
  }
  var tries = 0;
  var iv = setInterval(function(){
    tries++;
    if (document.getElementById('anno_imgcol')){ clearInterval(iv); setup(); }
    else if (tries > 80) clearInterval(iv);
  }, 400);
})();
</script>
""".replace("__DRAG_BBOX__", str(_DRAG_BBOX))


def build_app() -> gr.Blocks:
    feats = CFG.features
    half = (len(feats) + 1) // 2

    with gr.Blocks(title="地震道集标注器（多用户）") as demo:
        gr.Markdown("# 地震道集标注器（多用户中央服务）")

        # ---------- 账号栏（登出 / 切换账号） ----------
        with gr.Row():
            who = gr.Markdown("正在加载账号…", scale=4, elem_id="who_md")
            btn_logout = gr.Button("登出 / 切换账号", scale=1)

        # ---------- ① 建作业（仅管理员可见，服务端校验为准） ----------
        with gr.Accordion("① 建作业（仅管理员）", open=False, visible=False) as admin_build:
            with gr.Row():
                file_path = gr.Textbox(label="sgy/segy 文件路径", scale=3,
                                       placeholder="/path/to/xxx.sgy")
                endian = gr.Dropdown(["自动检测", "大端", "小端"], value="自动检测",
                                     label="字节序", scale=1)
                btn_load = gr.Button("读取文件信息", scale=1)
            file_info = gr.Markdown("尚未加载文件")
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
                build_hint = gr.Markdown(f"保存标注时每张道集将随机采 **{N_AUG} 个不同 clip 值**各渲一张导出图")
            btn_create = gr.Button("创建作业", variant="primary")
            build_info = gr.Markdown("")

        # ---------- ② 作业与标注（全体可见） ----------
        with gr.Accordion("② 作业与标注", open=True):
            with gr.Row():
                job_dd = gr.Dropdown(choices=[], label="作业", scale=3)
                btn_claim = gr.Button("领取下一张", variant="primary", scale=1)
                btn_skip = gr.Button("⏭ 跳过此张", scale=1)
                btn_release = gr.Button("归还此张", scale=1)
                btn_refresh = gr.Button("刷新", scale=1)
            with gr.Row():
                mine_dd = gr.Dropdown(choices=[], label="我标注的（选择以重开编辑）", scale=3)
                btn_inherit = gr.Button("⧉ 继承最近已标注", scale=1)
            anno_info = gr.Markdown("请选择作业后点「领取下一张」")
            radios = []
            box_statuses = []
            box_buttons = []   # (feature_key, 框选按钮, 清除按钮)
            with gr.Row(elem_id="anno_body"):
                with gr.Column(scale=3, elem_id="anno_imgcol", min_width=0):
                    cur_img = gr.Image(label="当前道集（点「▣ 框选」后按住鼠标拖动框选）",
                                       type="filepath", height=640)
                with gr.Column(scale=2, elem_id="anno_featcol", min_width=0):
                    sentence = gr.Textbox(label="句子预览", interactive=False, lines=3)
                    with gr.Row():
                        for col_feats in (feats[:half], feats[half:]):
                            with gr.Column():
                                for feat in col_feats:
                                    i = feats.index(feat)
                                    r = gr.Radio(choices=[fmt_option(o) for o in feat.options],
                                                 label=f"{i + 1}. {feat.name}", value=None)
                                    radios.append(r)
                                    if feat.bbox:
                                        with gr.Row():
                                            btn_box = gr.Button(f"▣ 框选{feat.name}", size="sm",
                                                                 elem_id=f"bb-{feat.key}")
                                            btn_clear = gr.Button("✕ 清除", size="sm",
                                                                  elem_id=f"cb-{feat.key}")
                                        box_statuses.append(gr.Markdown("未画框"))
                                        box_buttons.append((feat.key, btn_box, btn_clear))
            btn_save = gr.Button("保存并释放", variant="primary")

        # ---------- ③ 管理（仅管理员可见） ----------
        with gr.Accordion("③ 管理（仅管理员）", open=False, visible=False) as admin_mgr:
            with gr.Row():
                mgr_job_dd = gr.Dropdown(choices=[], label="作业", scale=3)
                btn_resync = gr.Button("全部重传", scale=1)
                btn_delete = gr.Button("🗑 删除作业", scale=1)
            resync_info = gr.Markdown("")
            del_info = gr.Markdown("")
            gr.Markdown("**作业进度一览**（每 5 秒自动刷新）")
            mgr_progress = gr.HTML(jobs_progress_html(JM.list_jobs()))
            with gr.Row():
                del_job_dd = gr.Dropdown(choices=[], label="已删除作业（本地保留，可恢复）",
                                         scale=3)
                btn_restore = gr.Button("♻ 恢复此作业", scale=1)
            gr.Markdown("**账号管理**：编辑项目根目录 `users.yaml`（增删用户/改角色），"
                        "下一次登录即生效，无需重启服务。")

        # ---------------- 事件 ----------------
        demo.load(init_page, None,
                  [job_dd, admin_build, admin_mgr, mgr_job_dd, who, del_job_dd])
        btn_logout.click(logout_click, None, who)

        btn_load.click(load_file, [file_path, endian], file_info)
        btn_scan.click(scan_values, [file_path, endian, gkey], [values_box, scan_info])
        btn_create.click(create_job_ui,
                         [file_path, endian, sort_keys, gkey, values_box, clip,
                          aug_lo, aug_hi, title, min_traces],
                         [build_info, job_dd, mgr_job_dd])

        anno_outputs = [cur_img, anno_info, sentence, *radios, *box_statuses]
        job_dd.change(open_job_for, job_dd, anno_outputs + [mine_dd])
        mine_dd.change(reopen_mine, [job_dd, mine_dd], anno_outputs)
        btn_claim.click(claim_next, job_dd, anno_outputs)
        btn_skip.click(skip_current, None, anno_outputs)
        btn_release.click(release_current, None, anno_outputs)
        btn_refresh.click(refresh_page, None, [job_dd, mine_dd, *anno_outputs])
        btn_save.click(save_anno, radios, anno_outputs)
        btn_inherit.click(inherit_previous, None, [sentence, *radios])

        for r in radios:
            r.change(on_radio_change, radios, sentence)
        for key, btn_box, btn_clear in box_buttons:
            btn_box.click(_box_enter(key), None, [anno_info, *box_statuses])
            btn_clear.click(_box_clear(key), None, [cur_img, anno_info, *box_statuses])

        btn_resync.click(resync_job, mgr_job_dd, resync_info)
        btn_delete.click(delete_job_click, mgr_job_dd,
                         [btn_delete, del_info, mgr_job_dd, del_job_dd,
                          mgr_progress, job_dd])
        btn_restore.click(restore_job_click, del_job_dd,
                          [del_info, mgr_job_dd, del_job_dd, mgr_progress, job_dd])

        # 管理区作业进度/删除确认：每 5 秒自动刷新（表仅放在 admin 可见的管理区）
        gr.Timer(value=5).tick(refresh_jobs_progress,
                               outputs=[mgr_progress, btn_delete, del_info],
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
            ok, msg = apply_drag_box(user,
                                     str(p.get("key") or ""),
                                     float(p.get("x0")), float(p.get("y0")),
                                     float(p.get("x1")), float(p.get("y1")))
        except Exception as e:                       # noqa: BLE001 —— 统一转 400 提示
            return JSONResponse({"ok": False, "err": str(e)}, status_code=400)
        return {"ok": ok, "msg": msg}

    @app.get("/api/boxes")
    async def anno_boxes(user: str = ""):
        """读取当前道集各 bbox 特征的框（自然像素 xyxy），供前端初始化/重开编辑。"""
        user = (user or "").strip()
        if not user or ACC.role(user) is None:
            return JSONResponse({"ok": False, "err": "unknown user"}, status_code=401)
        return {"ok": True, "boxes": _boxes_payload(user)}

    gr.mount_gradio_app(app, demo, path="/", auth=ACC.authenticate, css=ANNO_CSS,
                        head=ANNO_JS,
                        auth_message="多用户地震标注服务：使用 users.yaml 中的账号登录；"
                                     "登录后点右上角「登出 / 切换账号」即可换账号。")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
