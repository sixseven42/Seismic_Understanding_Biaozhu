# -*- coding: utf-8 -*-
"""
segy_reader.py — SEG-Y 读取层

特性：
- 支持大端 / 小端（标准 SEG-Y 为大端；实际生产中常见小端），可自动检测
- 支持采样格式码：1(IBM float32)、2(int32)、3(int16)、5(IEEE float32)、8(int8)
- 道头字节号采用 1-based（SEG-Y 标准文档惯例），如 (95, 96) 表示第 95~96 字节
- numpy memmap 惰性读取，不一次性载入全部数据
"""

from __future__ import annotations

import os
import numpy as np

HEADER_SIZE = 240          # 每道道头字节数
TEXT_HEADER_SIZE = 3200    # 文本头
BIN_HEADER_SIZE = 400      # 二进制头
FILE_HEADER_SIZE = TEXT_HEADER_SIZE + BIN_HEADER_SIZE

# 格式码 -> (numpy 类型, 每采样字节数, 是否IBM浮点)
FORMAT_MAP = {
    1: ("ibm4", 4, True),
    2: ("i4", 4, False),
    3: ("i2", 2, False),
    5: ("f4", 4, False),
    8: ("i1", 1, False),
}


class SegyReadError(Exception):
    pass


def _ibm_to_ieee(ibm: np.ndarray) -> np.ndarray:
    """IBM 360 浮点 -> IEEE float32。输入为 uint32 视图。"""
    ibm = ibm.astype("<u4") if ibm.dtype.byteorder == "<" else ibm.astype(">u4")
    sign = (ibm >> 31) & 0x01
    exponent = (ibm >> 24) & 0x7F
    mantissa = ibm & 0x00FFFFFF
    out = np.zeros(ibm.shape, dtype=np.float64)
    nz = mantissa != 0
    out[nz] = (mantissa[nz].astype(np.float64) / 0x1000000) * np.ldexp(
        1.0, (exponent[nz].astype(np.int32) - 64) * 4
    )
    out = np.where(sign == 1, -out, out)
    return out.astype(np.float32)


class SegyReader:
    """SEG-Y 文件读取器。"""

    def __init__(self, path: str, endian: str = "auto"):
        """
        endian: 'auto' | 'big' | 'little'
        """
        if not os.path.isfile(path):
            raise SegyReadError(f"文件不存在: {path}")
        self.path = path
        self.file_size = os.path.getsize(path)

        with open(path, "rb") as f:
            self._text_raw = f.read(TEXT_HEADER_SIZE)
            bh = f.read(BIN_HEADER_SIZE)
        if len(bh) < BIN_HEADER_SIZE:
            raise SegyReadError("文件过小，不是合法的 SEG-Y")

        self.endian = self._resolve_endian(bh, endian)
        e = "<" if self.endian == "little" else ">"

        # 二进制头字段（1-based 字节号见注释）
        self.ntr_field = int(np.frombuffer(bh[12:14], e + "i2")[0])   # 3213-3214
        self.dt_us = int(np.frombuffer(bh[16:18], e + "i2")[0])       # 3217-3218
        self.ns = int(np.frombuffer(bh[20:22], e + "i2")[0])          # 3221-3222
        self.format_code = int(np.frombuffer(bh[24:26], e + "i2")[0]) # 3225-3226

        if self.format_code not in FORMAT_MAP:
            raise SegyReadError(f"不支持的采样格式码: {self.format_code}")
        self._np_type, self.sample_bytes, self._is_ibm = FORMAT_MAP[self.format_code]
        # IBM 浮点 (ibm4) 无对应 numpy dtype：data 走 uint32 视图 + _ibm_to_ieee 转换，
        # 因此这里只给非 IBM 格式构造 dtype。
        if self._is_ibm:
            self._dtype = np.dtype(e + "u4")
        elif self.sample_bytes > 1:
            self._dtype = np.dtype(e + self._np_type)
        else:
            self._dtype = np.dtype("i1")

        self.trace_bytes = HEADER_SIZE + self.ns * self.sample_bytes
        body = self.file_size - FILE_HEADER_SIZE
        if body % self.trace_bytes != 0:
            raise SegyReadError(
                f"文件大小与道长度不一致: body={body}, trace_bytes={self.trace_bytes}"
            )
        self.n_traces = body // self.trace_bytes

        # memmap：二维视图，每行 = 道头+道数据
        self._raw = np.memmap(
            path, dtype=np.uint8, mode="r",
            offset=FILE_HEADER_SIZE, shape=(self.n_traces, self.trace_bytes),
        )
        self._headers = self._raw[:, :HEADER_SIZE]

    # ------------------------------------------------------------------
    def _resolve_endian(self, bh: bytes, endian: str) -> str:
        if endian in ("big", "little"):
            return endian
        for cand, e in (("big", ">"), ("little", "<")):
            ns = int(np.frombuffer(bh[20:22], e + "i2")[0])
            fmt = int(np.frombuffer(bh[24:26], e + "i2")[0])
            dt = int(np.frombuffer(bh[16:18], e + "i2")[0])
            if fmt not in FORMAT_MAP or not (0 < ns < 10_000_000) or dt <= 0:
                continue
            tb = HEADER_SIZE + ns * FORMAT_MAP[fmt][1]
            if (self.file_size - FILE_HEADER_SIZE) % tb == 0:
                return cand
        raise SegyReadError("无法自动检测字节序，请手动指定 endian='big' 或 'little'")

    # ------------------------------------------------------------------
    def info(self) -> dict:
        return {
            "path": self.path,
            "endian": self.endian,
            "n_traces": self.n_traces,
            "ns": self.ns,
            "dt_us": self.dt_us,
            "dt_ms": self.dt_us / 1000.0,
            "format_code": self.format_code,
            "file_size_mb": round(self.file_size / 1e6, 1),
        }

    def textual_header(self) -> str:
        """解码文本头（自动尝试 EBCDIC / ASCII）。"""
        try:
            txt = self._text_raw.decode("cp500")
        except Exception:
            txt = self._text_raw.decode("ascii", errors="replace")
        lines = [txt[i:i + 80] for i in range(0, len(txt), 80)]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def header_field(self, byte_start: int, byte_end: int | None = None) -> np.ndarray:
        """
        按 1-based 字节区间读取全部道的道头整数字段。
        byte_end 省略时与 byte_start 相同。区间长度须为 1/2/4/8 字节。
        返回 int64 数组（有符号，按文件端序解析）。
        """
        if byte_end is None:
            byte_end = byte_start
        if not (1 <= byte_start <= byte_end <= HEADER_SIZE):
            raise SegyReadError(
                f"字节区间 [{byte_start}, {byte_end}] 超出道头范围 1~{HEADER_SIZE}"
            )
        n = byte_end - byte_start + 1
        typemap = {1: "i1", 2: "i2", 4: "i4", 8: "i8"}
        if n not in typemap:
            raise SegyReadError(f"区间长度 {n} 字节不支持（须为 1/2/4/8）")
        e = "<" if self.endian == "little" else ">"
        buf = np.ascontiguousarray(self._headers[:, byte_start - 1: byte_end])
        return buf.view(e + typemap[n]).ravel().astype(np.int64)

    def header_field_range(self) -> tuple[int, int]:
        return (1, HEADER_SIZE)

    # ------------------------------------------------------------------
    def get_traces(self, indices: np.ndarray | list[int]) -> np.ndarray:
        """
        按道索引（0-based）取道数据，返回 float32 数组 shape=(len(indices), ns)。
        """
        idx = np.asarray(indices)
        data = np.asarray(self._raw[idx, HEADER_SIZE:])
        if self._is_ibm:
            e = "<" if self.endian == "little" else ">"
            return _ibm_to_ieee(np.ascontiguousarray(data).view(e + "u4").reshape(len(idx), self.ns))
        return np.ascontiguousarray(data).view(self._dtype).reshape(len(idx), self.ns).astype(np.float32)

    def close(self):
        if hasattr(self, "_raw") and self._raw is not None:
            self._raw._mmap.close()
            self._raw = None

    def compatible_with(self, other: "SegyReader") -> tuple[bool, str]:
        """Check that two SEG-Y files can be displayed trace-for-trace.

        Residual jobs use the trace indices from the first file for all three
        sources.  Comparing the complete trace headers (rather than only a
        few commonly used fields) prevents a subtle line/gather misalignment.
        Sample count and trace count are part of the layout contract as well.
        """
        if not isinstance(other, SegyReader):
            return False, "不是 SEG-Y 读取器"
        if self.n_traces != other.n_traces:
            return False, f"道数不一致（{self.n_traces} 与 {other.n_traces}）"
        if self.ns != other.ns:
            return False, f"每道采样点数不一致（{self.ns} 与 {other.ns}）"
        if self.format_code != other.format_code:
            return False, f"采样格式码不一致（{self.format_code} 与 {other.format_code}）"
        if self.dt_us != other.dt_us:
            return False, f"采样间隔不一致（{self.dt_us} 与 {other.dt_us} 微秒）"
        # Compare in chunks so a multi-million-trace file does not allocate a
        # second copy of every header just for this validation.
        for start in range(0, self.n_traces, 65536):
            stop = min(self.n_traces, start + 65536)
            if not np.array_equal(self._headers[start:stop], other._headers[start:stop]):
                return False, "道头不一致（请确认三个文件按同一顺序导出）"
        return True, ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
