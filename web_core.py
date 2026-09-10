# -*- coding: utf-8 -*-
"""web_core.py — 标注 Web 界面可单测的纯逻辑（不依赖 gradio、无副作用）。

从原 web_app.py 抽出：选项显示文本、选项反解析、像素框→数据框映射。
web_app.py 复用本模块，删除其本地重复定义。
"""

from __future__ import annotations

import html
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


def parse_filter(f1, f2, f3, f4) -> tuple[dict | None, str]:
    """校验面波区「滤波」的四角频率，返回 (params, err)。

    合法：params = {"f1","f2","f3","f4"}（float，Hz）、err = ""；
    非法：params = None、err 为中文提示。界面角频率一律按 Hz 输入，
    实际通带由 jobmanager 用 job.json 的 dt_ms 换算到 FFT 频点。
    """
    try:
        vals = [float(v) for v in (f1, f2, f3, f4)]
    except (TypeError, ValueError):
        return None, "滤波参数须为数字（四个角频率，单位 Hz）"
    if not all(math.isfinite(v) for v in vals):
        return None, "滤波参数须为有限数值"
    a, b, c, d = vals
    if not (0.0 <= a < b < c < d):
        return None, (f"滤波角频率须满足 0 ≤ f1 < f2 < f3 < f4，"
                      f"当前 {a:g} / {b:g} / {c:g} / {d:g} Hz")
    return {"f1": a, "f2": b, "f3": c, "f4": d}, ""


def active_filter(st: dict, gid: str) -> dict | None:
    """本会话对该道集生效的滤波参数；无 / 非本道集 / 结构异常一律返回 None。

    状态与 gid 绑定，换到别的道集即自动失效 —— 无需在领取/跳过/重开等每个
    handler 里手动清除，也就不会出现「忘了还开着滤波」而按滤波图下标注。
    """
    f = st.get("filter")
    if not isinstance(f, dict) or f.get("gid") != gid:
        return None
    params = {k: f.get(k) for k in ("f1", "f2", "f3", "f4")}
    if any(not isinstance(v, (int, float)) for v in params.values()):
        return None
    return params


ABSENT_LABEL = "不存在"


def box_target(feats, selection: dict, boxes: dict) -> str | None:
    """Ctrl+左键「两点定矩形」当前作用的 bbox 特征 key；当前不该画框时返回 None。

    规则（依次）：
      1. 已选且非「不存在」、但还没框的 → 第一个（正常流程：答完题按顺序框）；
      2. 尚未作答、也还没框的 → 第一个（允许先框后选）；
      3. 都框好了 → 最后一个**非「不存在」**的（Ctrl 再点即重画它，便于微调）；
      4. 一个都不需要框（例如两个都选了「不存在」）→ **None**，Ctrl 不再画出多余的框。
    选「不存在」的特征在第 1 步就被跳过，也不会在第 3 步被当成重画目标。
    """
    feats = list(feats or [])
    if not feats:
        return None
    sel = selection or {}
    bx = boxes or {}

    def has(k: str) -> bool:
        return bool(bx.get(k))

    def is_absent(f) -> bool:
        return sel.get(f.name) == ABSENT_LABEL

    for f in feats:
        lab = sel.get(f.name)
        if lab and not is_absent(f) and not has(f.key):
            return f.key
    for f in feats:
        if not sel.get(f.name) and not has(f.key):
            return f.key
    for f in reversed(feats):
        if not is_absent(f):
            return f.key
    return None


_STATE_CN = {"open": "进行中", "closed": "已关闭", "broken": "异常"}


def jobs_progress_html(jobs: list[dict]) -> str:
    """作业进度一览表 HTML 片段（纯字符串，便于单测与定时刷新）。

    jobs: JM.list_jobs() 的结构，每项含 job_id/title/state/labeled/total。
    展示每作业：标题+ID、状态、已完成 x/总数 y、百分比、进度条；total=0 防除零。
    """
    rows = []
    for j in jobs:
        jid = html.escape(str(j.get("job_id") or ""))
        title = html.escape(str(j.get("title") or "") or jid)
        state = str(j.get("state") or "")
        try:
            lab = int(j.get("labeled") or 0)
            tot = int(j.get("total") or 0)
        except (TypeError, ValueError):
            lab = tot = 0
        pct_i = int(round(lab / tot * 100)) if tot else 0
        state_txt = html.escape(_STATE_CN.get(state, state))
        if tot and lab >= tot:
            bar_cls = "jp-done"
        elif state == "closed":
            bar_cls = "jp-close"
        else:
            bar_cls = "jp-run"
        rows.append(
            "<tr>"
            f"<td class='jp-title'>{title}<span class='jp-id'>{jid}</span></td>"
            f"<td class='jp-state'>{state_txt}</td>"
            f"<td class='jp-count'>{lab} / {tot}</td>"
            f"<td class='jp-pct'>{pct_i}%</td>"
            f"<td class='jp-bar'><div class='jp-bg'><div class='{bar_cls}' "
            f"style='width:{min(100, max(0, pct_i))}%'></div></div></td>"
            "</tr>"
        )
    if not rows:
        return "<p class='jp-empty'>暂无作业</p>"
    head = ("<table class='jobs-progress'>"
            "<tr><th>作业</th><th>状态</th><th>已完成</th><th>进度</th><th></th></tr>")
    return head + "".join(rows) + "</table>"
