# -*- coding: utf-8 -*-
"""tests/segy_factory.py — 合成最小 SEG-Y（小端，格式码 5 = IEEE float32）。
用于不依赖真实数据的 建作业/恢复 真链路测试。道头布局对齐示例文件习惯：
  9-12 FFID(int32), 13-16 道号(int32), 95-96 炮号(int16, 2 字节)。
"""
import numpy as np

TEXT = 3200
BIN = 400
HDR = 240

def _bin_header(ns, ntrace, endian="little", format_code=5, dt_us=2000):
    e = "<" if endian == "little" else ">"
    bh = bytearray(BIN)
    bh[12:14] = np.array([ntrace], dtype=e + "i2").tobytes()     # 每道字节数提示(非道数, 仅参考)
    bh[16:18] = np.array([dt_us], dtype=e + "i2").tobytes()      # 采样间隔 us
    bh[20:22] = np.array([ns], dtype=e + "i2").tobytes()         # ns
    bh[24:26] = np.array([format_code], dtype=e + "i2").tobytes()
    return bh

def write_sgy(path, n_gathers=3, traces_per=8, ns=64, endian="little",
              fld_gather=(95, 96), fld_ffid=(9, 12), fld_tr=(13, 16)):
    e = "<" if endian == "little" else ">"
    n_trace = n_gathers * traces_per
    rng = np.random.default_rng(0)
    rows = []
    gid = 0
    for gv in range(1, n_gathers + 1):
        for t in range(traces_per):
            hdr = bytearray(HDR)
            g = np.int16(gv)                                     # 炮号
            a, b = fld_gather
            hdr[a - 1:b] = g.tobytes() if (b - a + 1) >= 2 else bytes([g & 0xFF])
            a, b = fld_ffid
            hdr[a - 1:b] = np.array([37000 + gid], dtype=e + "i4").tobytes()
            a, b = fld_tr
            hdr[a - 1:b] = np.array([t + 1], dtype=e + "i4").tobytes()
            data = rng.normal(size=ns).astype(np.float32)
            rows.append(bytes(hdr) + data.tobytes())
    import os as _os
    _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x00" * TEXT)
        f.write(_bin_header(ns, n_trace, endian))
        for row in rows:
            f.write(row)
