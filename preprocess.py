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


@register("normalize", params={},
          desc="按道集整体绝对最大值归一化到 [-1, 1]")
def normalize(data: np.ndarray) -> np.ndarray:
    m = np.abs(data).max()
    return data / m if m > 0 else data


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
