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
