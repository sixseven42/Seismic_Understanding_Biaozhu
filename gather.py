# -*- coding: utf-8 -*-
"""
gather.py — 多级道头排序 + 抽道集

用法示例：
    keys = [(95, 96), (13, 16)]      # 先按 95-96，再按 13-16（1-based，升序）
    gathers = extract_gathers(reader, sort_keys=keys, gather_key=(95, 96))
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from segy_reader import SegyReader


@dataclass
class Gather:
    """一个道集：某个键道头取值对应的所有道（已按排序键排好序）。"""
    key: str                      # 键道头描述，如 "95-96"
    value: int                    # 键值，如炮号 1373
    trace_indices: np.ndarray     # 0-based 道索引，已排序
    sort_values: dict = field(default_factory=dict)  # 各排序键的首道值（调试/展示用）
    prefix: str = ""              # gather_id 前缀（通常=来源文件名，防多文件冲突）

    @property
    def n_traces(self) -> int:
        return len(self.trace_indices)

    @property
    def gather_id(self) -> str:
        return f"{self.prefix}{self.key}_{self.value}"


def parse_key(text: str) -> tuple[int, int]:
    """把 '95-96' 或 '95' 解析为 (95, 96)。"""
    text = text.strip()
    if "-" in text:
        a, b = text.split("-", 1)
        return int(a), int(b)
    a = int(text)
    return a, a


def key_str(key: tuple[int, int]) -> str:
    return f"{key[0]}-{key[1]}" if key[0] != key[1] else str(key[0])


def sorted_order(reader: SegyReader, sort_keys: list[tuple[int, int]]) -> np.ndarray:
    """按多个道头键做稳定字典序升序，返回排序后的道索引。"""
    if not sort_keys:
        return np.arange(reader.n_traces)
    cols = [reader.header_field(a, b) for a, b in sort_keys]
    # np.lexsort 最后一个键是主键，所以反转
    return np.lexsort(tuple(cols[::-1]))


def extract_gathers(
    reader: SegyReader,
    sort_keys: list[tuple[int, int]],
    gather_key: tuple[int, int],
    values: list[int] | None = None,
) -> list[Gather]:
    """
    先按 sort_keys 全局排序，再按 gather_key 抽道集。
    values 为 None 时自动取所有唯一值；否则只抽指定值。
    """
    order = sorted_order(reader, sort_keys)
    gvals_all = reader.header_field(*gather_key)
    gvals_sorted = gvals_all[order]

    uniq, starts = np.unique(gvals_sorted, return_index=True)
    uniq = uniq[np.argsort(starts)]  # 保持排序后的出现顺序
    want = set(values) if values is not None else None

    kstr = key_str(gather_key)
    gathers = []
    boundaries = list(starts) + [len(order)]
    pos_of = {v: i for i, v in enumerate(uniq)}
    for v in uniq:
        if want is not None and int(v) not in want:
            continue
        i = pos_of[v]
        idx = order[boundaries[i]: boundaries[i + 1]]
        gathers.append(Gather(key=kstr, value=int(v), trace_indices=idx))
    return gathers


def gather_values(reader: SegyReader, gather_key: tuple[int, int]) -> np.ndarray:
    """列出键道头的所有唯一值（供界面勾选）。"""
    return np.unique(reader.header_field(*gather_key))
