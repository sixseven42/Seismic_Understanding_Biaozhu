# -*- coding: utf-8 -*-
"""
preprocess.py — 预处理管线（注册表模式，便于扩展）

新增预处理方法只需：
    @register("方法名", params={"参数名": 默认值})
    def my_step(data, **params): ...
然后在使用处写 {"name": "方法名", "params": {...}} 即可串入管线。
"""

from __future__ import annotations

import numpy as np

_REGISTRY: dict[str, dict] = {}


def register(name: str, params: dict | None = None, desc: str = ""):
    def deco(func):
        _REGISTRY[name] = {"func": func, "params": params or {}, "desc": desc}
        return func
    return deco


def available() -> dict:
    return {k: {"params": v["params"], "desc": v["desc"]} for k, v in _REGISTRY.items()}


@register("clip_percentile", params={"percentile": 99.0},
          desc="按绝对值的分位数截断：幅值超过 ±P 分位数的样点被压到边界")
def clip_percentile(data: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    if not (0 < percentile <= 100):
        raise ValueError("percentile 须在 (0, 100]")
    if percentile >= 100:
        return data
    c = np.percentile(np.abs(data), percentile)
    if c <= 0:
        return data
    return np.clip(data, -c, c)


def clip_bound(data: np.ndarray, percentile: float = 99.0) -> float:
    """求显示色标界限：|data| 的 percentile 分位值（与 clip_percentile 同口径）。

    用于把「未滤波」和「滤波后」两张图放到**同一色标**下对比：界限只从原始数据算一次，
    滤波图因此不会被重新拉伸 —— 面波被压掉的地方会真的变淡，这才看得出来滤波作用。
    """
    if not (0 < percentile <= 100):
        raise ValueError("percentile 须在 (0, 100]")
    if percentile >= 100:
        return float(np.abs(data).max()) or 1.0
    c = float(np.percentile(np.abs(data), percentile))
    return c if c > 0 else (float(np.abs(data).max()) or 1.0)


@register("clip_abs", params={"vlim": 1.0},
          desc="按绝对幅值 ±vlim 截断（滤波前后共用同一色标时用）")
def clip_abs(data: np.ndarray, vlim: float = 1.0) -> np.ndarray:
    v = abs(float(vlim))
    return np.clip(data, -v, v) if v > 0 else data


@register("normalize", params={},
          desc="按道集整体绝对最大值归一化到 [-1, 1]")
def normalize(data: np.ndarray) -> np.ndarray:
    m = np.abs(data).max()
    return data / m if m > 0 else data


@register("bandpass",
          params={"f1": 10.0, "f2": 20.0, "f3": 100.0, "f4": 150.0, "dt_ms": 1.0},
          desc="SeiSee 四角频率零相位带通（逐道 FFT 余弦过渡，与 bandpass.py 同一算法）")
def bandpass(data: np.ndarray, f1: float = 10.0, f2: float = 20.0,
             f3: float = 100.0, f4: float = 150.0, dt_ms: float = 1.0) -> np.ndarray:
    """对 (n_traces, ns) 逐道做带通滤波，返回同形 float32 数组。

    四角频率须满足 0 <= f1 < f2 < f3 < f4（f1 低截 / f2 低通拐点 / f3 高通拐点 /
    f4 高截），dt_ms 为采样间隔（毫秒）且必须 > 0 —— 传错会整体平移实际通带位置。
    注意：这是「看」的辅助（标注界面按需滤波），导出的训练图始终用未滤波数据。
    """
    from bandpass import bandpass_seisee
    try:
        f1, f2, f3, f4 = (float(f1), float(f2), float(f3), float(f4))
        dt_ms = float(dt_ms)
    except (TypeError, ValueError):
        raise ValueError("滤波参数须为数值")
    if not (0.0 <= f1 < f2 < f3 < f4):
        raise ValueError(
            f"滤波角频率须满足 0 ≤ f1 < f2 < f3 < f4，当前 {f1:g}/{f2:g}/{f3:g}/{f4:g} Hz")
    if not dt_ms > 0:
        raise ValueError(f"采样间隔须 > 0 ms，当前 {dt_ms:g}")

    dt = dt_ms / 1000.0
    # astype(float64) 必得副本，滤波不会就地改写调用方的数据（干净视图不被污染）
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 1:
        return bandpass_seisee(x, dt, f1, f2, f3, f4).astype(np.float32)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = bandpass_seisee(x[i], dt, f1, f2, f3, f4)
    return out.astype(np.float32)


def sample_clip_values(lo: float, hi: float, n: int) -> list[float]:
    """
    在 [lo, hi] 内按 0.1 为步长无放回随机采样 n 个 clip 分位值（升序）。
    无放回保证同一道集的 n 张增强图 clip 值互不相同，避免文件名/记录冲突。
    若区间内可选值不足 n 个，抛出 ValueError。
    """
    grid = np.round(np.arange(lo, hi + 1e-9, 0.1), 1)
    if len(grid) < n:
        raise ValueError(
            f"clip 范围 [{lo}, {hi}] 内只有 {len(grid)} 个可选值（0.1 步长），少于 {n} 个")
    return sorted(float(v) for v in np.random.choice(grid, n, replace=False))


def apply_pipeline(data: np.ndarray, steps: list[dict]) -> np.ndarray:
    """
    steps: [{"name": "clip_percentile", "params": {"percentile": 99}}, ...]
    按顺序执行，输入输出均为 (n_traces, ns) float32 数组。
    """
    out = data
    for step in steps:
        name = step["name"]
        if name not in _REGISTRY:
            raise KeyError(f"未注册的预处理方法: {name}（可选: {list(_REGISTRY)}）")
        out = _REGISTRY[name]["func"](out, **step.get("params", {}))
    return out
