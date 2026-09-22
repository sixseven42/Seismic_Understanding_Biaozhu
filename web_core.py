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


def pixel_polygon_to_data(points, n_tr: int, ns: int) -> dict:
    """Natural-pixel polygon -> stored polygon plus compatible bounding ranges."""
    clean = []
    for point in points or []:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("包络点格式无效")
        x = max(0.0, min(float(IMG_W - 1), float(point[0])))
        y = max(0.0, min(float(IMG_H - 1), float(point[1])))
        p = [int(round(x)), int(round(y))]
        if not clean or p != clean[-1]:
            clean.append(p)
    if len(clean) < 3 or len({tuple(p) for p in clean}) < 3:
        raise ValueError("不规则包络至少需要 3 个不同的点")
    xs, ys = [p[0] for p in clean], [p[1] for p in clean]
    bounds = pixel_box_to_data((min(xs), min(ys)), (max(xs), max(ys)), n_tr, ns)
    return {"points": clean, **bounds}


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


def staff_counts_html(rows) -> str:
    """各用户标注总量表（仅管理员的管理区显示）。

    rows: [(用户名, 角色中文, 标注张数), ...]，按调用方给的顺序展示。
    只做展示，不做权限判断 —— 该 HTML 只挂在管理员可见的区块里。
    """
    body = []
    total = 0
    for name, role, n in rows:
        try:
            cnt = int(n)
        except (TypeError, ValueError):
            cnt = 0
        total += cnt
        body.append(
            "<tr>"
            f"<td class='sc-user'>{html.escape(str(name))}</td>"
            f"<td class='sc-role'>{html.escape(str(role))}</td>"
            f"<td class='sc-count'>{cnt}</td>"
            "</tr>")
    if not body:
        return "<p class='jp-empty'>暂无账号</p>"
    head = ("<table class='staff-counts'>"
            "<tr><th>用户</th><th>角色</th><th>标注张数</th></tr>")
    tail = (f"<tr class='sc-total'><td>合计</td><td></td>"
            f"<td class='sc-count'>{total}</td></tr>")
    return head + "".join(body) + tail + "</table>"


ABSENT_LABEL = "不存在"


def box_note(feats, selection: dict, target: str | None) -> str:
    """画布提示语的兜底文案 —— 目标为空时给一句**准确**的说明，别乱说"已完成"。

    target 为空只可能是三种情况，含义完全不同：
      1. 本题所有带框特征都选了「不存在」→ 根本不需要画框（不是"已完成"）；
      2. 特征配置里没有带框特征 → 本题不涉及框选；
      3. 没有当前道集（池子标完/未领取）→ 此时前端应**什么都不画**。
    前两种给说明，第三种返回空串（前端据此隐藏提示）。
    """
    if target:
        return ""
    feats = list(feats or [])
    if not feats:
        return "本题不需要画框"
    sel = selection or {}
    if all(sel.get(f.name) == ABSENT_LABEL for f in feats):
        return "所有拉框项均选「不存在」，本张无需画框"
    return ""


def box_target(feats, selection: dict, boxes: dict) -> str | None:
    """Ctrl+左键当前作用的包络特征 key；当前不该标范围时返回 None。

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


def prev_gid(hist, cur: str | None = None) -> tuple[str | None, list[str]]:
    """「↩ 返回上一张」的下一步：返回 (目标道集, 回退后的 hist)。

    hist = 当前道集**之前**访问过的道集，按访问先后（旧→新）排列，**不含**当前道集；
    显示的道集一变就往里压旧的，所以它是「浏览历史」而不是「标注顺序」。
    这正是不能用「我标注的」列表顺序代替的原因：那张表按**首次标注时间**排序
    （见 storage.LabelStore.upsert：已存在的 gid 不挪位），修正保存不会让记录回到末尾，
    于是「刚返回并修好 G3」之后按表顺序会翻到别的地方去。

    栈空 → (None, [])，调用方据此提示「已是最早一张」。
    cur 只作防御：万一当前道集被重复压在栈顶（不该发生），先剔除再取栈顶，
    免得「返回上一张」原地不动、看起来像按钮失灵。
    """
    h = [g for g in (hist or []) if g]
    if cur and h and h[-1] == cur:
        h.pop()
    if not h:
        return None, []
    return h[-1], h[:-1]


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
