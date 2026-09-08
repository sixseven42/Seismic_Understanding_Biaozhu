# -*- coding: utf-8 -*-
"""
gather.py — 多级道头排序 + 抽道集

用法示例：
    单字段抽道集：
        keys = [(95, 96), (13, 16)]      # 先按 95-96，再按 13-16（1-based，升序）
        gathers = extract_gathers(reader, sort_keys=keys, gather_key=(95, 96))

    多字段复合抽道集（如 FFID + Line 共同界定一个道集）：
        sort_keys  = [(9, 12), (189, 192), (13, 16)]   # 分组字段放前、组内排序字段放后
        gathers    = extract_gathers(reader, sort_keys=sort_keys,
                                     gather_key=[(9, 12), (189, 192)])
        # 每个 gather 的 value 为 tuple，如 (ffid, line)

约定：抽道集键的各字段必须出现在 sort_keys 的前部且升序一致，
保证同组道在排序后连续（工具按排序后连续同值块分组）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from segy_reader import SegyReader


@dataclass
class Gather:
    """一个道集：某个键道头取值对应的所有道（已按排序键排好序）。"""
    key: str                      # 键道头描述，如 "95-96" 或 "9-12,189-192"
    value: int | tuple            # 键值：单字段为 int；多字段为 tuple(int,...)
    trace_indices: np.ndarray     # 0-based 道索引，已排序
    sort_values: dict = field(default_factory=dict)  # 各排序键的首道值（调试/展示用）
    prefix: str = ""              # gather_id 前缀（通常=来源文件名，防多文件冲突）

    @property
    def n_traces(self) -> int:
        return len(self.trace_indices)

    @property
    def value_text(self) -> str:
        """键值展示文本：多字段用英文逗号连接。"""
        if isinstance(self.value, tuple):
            return ",".join(str(int(v)) for v in self.value)
        return str(int(self.value))

    @property
    def gather_id(self) -> str:
        return f"{self.prefix}{self.key}_{self.value_text}"


def parse_key(text: str) -> tuple[int, int]:
    """把 '95-96' 或 '95' 解析为 (95, 96)。"""
    text = text.strip()
    if "-" in text:
        a, b = text.split("-", 1)
        return int(a), int(b)
    a = int(text)
    return a, a


def parse_ranges(text: str) -> list[tuple[int, int]]:
    """把 '9-12, 189-192' 解析为 [(9,12),(189,192)]；单个 '95-96' 返回单元素列表。"""
    return [parse_key(t) for t in str(text).split(",") if t.strip()]


def key_str(key: tuple[int, int]) -> str:
    return f"{key[0]}-{key[1]}" if key[0] != key[1] else str(key[0])


def ranges_label(ranges: list[tuple[int, int]]) -> str:
    """多个键区间的展示名，如 '9-12,189-192'。"""
    return ",".join(key_str(r) for r in ranges)


def _norm_ranges(gather_key) -> list[tuple[int, int]]:
    """把各种形式的 gather_key 归一成 [(a,b), ...]。
    兼容：字符串 "95-96" / "9-12, 189-192"；(95,96)；[(9,12),(189,192)]；((9,12),(189,192))。"""
    if isinstance(gather_key, str):
        return parse_ranges(gather_key)
    if isinstance(gather_key, tuple):
        if len(gather_key) == 2 and not isinstance(gather_key[0], (tuple, list)):
            return [tuple(int(x) for x in gather_key)]
        return [tuple(int(x) for x in r) for r in gather_key]
    if isinstance(gather_key, list):
        if gather_key and not isinstance(gather_key[0], (tuple, list)):
            return [tuple(int(x) for x in gather_key)]
        return [tuple(int(x) for x in r) for r in gather_key]
    raise TypeError(f"无法解析的 gather_key: {gather_key!r}")


def sorted_order(reader: SegyReader, sort_keys: list[tuple[int, int]]) -> np.ndarray:
    """按多个道头键做稳定字典序升序，返回排序后的道索引。"""
    if not sort_keys:
        return np.arange(reader.n_traces)
    cols = [reader.header_field(a, b) for a, b in sort_keys]
    # np.lexsort 最后一个键是主键，所以反转；cols 顺序 = 主键在前
    return np.lexsort(tuple(cols[::-1]))


def _group_runs(mat: np.ndarray):
    """沿排序后道序，找出连续「各分组字段都不变」的块。
    返回 [(start, end, label), ...]；label 为 int 或 tuple(int,...)。"""
    n = len(mat)
    if n == 0:
        return []
    change = np.any(mat[1:] != mat[:-1], axis=1)
    starts = np.flatnonzero(np.concatenate(([True], change)))
    ends = np.concatenate((starts[1:], [n]))
    multi = mat.shape[1] > 1
    out = []
    for s, e in zip(starts, ends):
        if multi:
            label = tuple(int(x) for x in mat[s])
        else:
            label = int(mat[s, 0])
        out.append((int(s), int(e), label))
    return out


def extract_gathers(
    reader: SegyReader,
    sort_keys: list[tuple[int, int]],
    gather_key,
    values: list | None = None,
) -> list[Gather]:
    """
    先按 sort_keys 全局排序，再按 gather_key（单/多字段）抽道集。
    values 为 None 时自动取所有唯一组；否则只抽指定组。
    单字段 values 为 [int,...]；多字段为 [(a,b),...] 或 "a,b" 字符串列表。
    """
    gkeys = _norm_ranges(gather_key)
    order = sorted_order(reader, sort_keys)
    if len(order) == 0:
        return []
    cols = [reader.header_field(a, b)[order] for a, b in gkeys]
    mat = np.column_stack(cols)

    multi = len(gkeys) > 1
    runs = _group_runs(mat)
    label_f = ranges_label(gkeys)

    if values is not None:
        if multi:
            want = set()
            for item in values:
                if isinstance(item, (tuple, list)):
                    want.add(tuple(int(x) for x in item))
                else:
                    want.add(tuple(int(x) for x in str(item).split(",")))
        else:
            want = {int(v) for v in values}
    else:
        want = None

    gathers = []
    for s, e, lab in runs:
        if want is not None and lab not in want:
            continue
        if multi:
            key_disp = label_f
        else:
            key_disp = key_str(gkeys[0])
        gathers.append(Gather(key=key_disp, value=lab,
                              trace_indices=order[s:e]))
    return gathers


def gather_values(reader: SegyReader, gather_key):
    """列出键道头的所有唯一组（供界面勾选）。
    单字段返回 np.ndarray of int；多字段返回 list[tuple]。"""
    gkeys = _norm_ranges(gather_key)
    cols = [reader.header_field(a, b) for a, b in gkeys]
    if len(gkeys) == 1:
        return np.unique(cols[0])
    arr = np.column_stack(cols)
    return [tuple(int(x) for x in row) for row in np.unique(arr, axis=0)]
