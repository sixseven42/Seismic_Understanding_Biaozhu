#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeiSee-style band-pass filter for SEG-Y files
=============================================

Re-implements the *exact* band-pass algorithm that SeiSee's
"Processing -> Band Pass Filter" uses (reverse-engineered from SeiSee.exe
v2.22.2):

  * per trace, zero-phase, frequency-domain (FFT) filtering;
  * four corner frequencies F1 < F2 < F3 < F4 (Hz), stored per trace
    (F1,F2,F3,F4) = (low-cut, low-pass, high-pass, high-cut);
  * N = next power of two >= number of samples (padded with zeros);
  * bin_i = trunc(F_i * N * dt),  dt = sample interval in seconds;
  * cosine (Hann) taper mask:
        w(k) = 0                                    k <  k1  or  k >= k4
        w(k) = 0.5*(1 - cos(pi*(k-k1)/(k2-k1)))     k1 <= k  < k2
        w(k) = 1                                    k2 <= k  < k3
        w(k) = 0.5*(1 + cos(pi*(k-k3)/(k4-k3)))     k3 <= k  < k4
  * multiply the spectrum by w(k) for k = 0..N/2 and its mirror, then
    inverse FFT, take the first n samples back.

SeiSee defaults are F1..F4 = 10 / 20 / 100 / 150 Hz.

Input / output SEG-Y: the textual(3200 B) + binary(400 B) headers and every
240-byte trace header are copied byte-for-byte; only the trace sample values
are replaced, in the file's original sample format.

Supported SEG-Y sample format codes (binary header bytes 3225-3226):
    1 = IBM 32-bit float, 2 = 32-bit int, 3 = 16-bit int, 5 = IEEE 32-bit float.
Other codes (incl. 8-bit / fixed point w/ gain) raise an error.

Only dependency is numpy.

Usage:
    python seisee_bandpass.py IN.sgy OUT.sgy --f1 8 --f2 12 --f3 80 --f4 120
    python seisee_bandpass.py IN.sgy OUT.sgy                 # default 10/20/100/150
    python seisee_bandpass.py IN.sgy OUT.sgy --f1 5 --f2 10 --f3 60 --f4 80 \
        --dt-us 2000 --format 5
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

# ---------------------------------------------------------------------------
# The filter (same algorithm as SeiSee 0x4101f0 / 0x419dcc)
# ---------------------------------------------------------------------------


def _next_pow2(n: int) -> int:
    N = 1
    while N < n:
        N <<= 1
    return N


def bandpass_seisee(data, dt, f1, f2, f3, f4):
    """Zero-phase FFT band-pass with 4 cosine-tapered corners.

    data : 1-D array of float sample values
    dt   : sample interval in SECONDS (must be > 0)
    f1..f4 : corner frequencies in Hz, must satisfy 0 <= f1 < f2 < f3 < f4

    Returns a float64 array of the same length.  Mirrors SeiSee's algorithm.
    """
    x = np.asarray(data, dtype=np.float64).reshape(-1)
    n = x.size

    # -- guards (SeiSee returns without changing the trace on failure) -------
    if n <= 0 or dt <= 0 or not (0.0 <= f1 < f2 < f3 < f4):
        return x.copy()

    # FFT length: next power of two >= n, zero-padded (SeiSee uses the same).
    # (SeiSee.exe itself caps N at 16384/32768 because of its internal FFT;
    #  numpy has no such limit, so longer traces are filtered here too.)
    N = _next_pow2(n)

    # corner frequencies -> FFT bins :  k_i = trunc(F_i * N * dt)
    k1, k2, k3, k4 = (int(f * N * dt) for f in (f1, f2, f3, f4))
    if not (k1 < k2 < k3 < k4):
        return x.copy()

    # forward FFT of the (zero-padded) real trace
    spec = np.fft.rfft(x, n=N)          # bins 0 .. N//2

    # cosine taper mask on bins 0..N/2  (mirror half is implicit for real FFT)
    bins = np.arange(N // 2 + 1, dtype=np.float64)
    w = np.zeros_like(bins)
    low = (bins >= k1) & (bins < k2)    # rising edge  F1 -> F2
    mid = (bins >= k2) & (bins < k3)    # pass band    F2 -> F3
    hi = (bins >= k3) & (bins < k4)     # falling edge F3 -> F4
    w[low] = 0.5 * (1.0 - np.cos(np.pi * (bins[low] - k1) / (k2 - k1)))
    w[mid] = 1.0
    w[hi] = 0.5 * (1.0 + np.cos(np.pi * (bins[hi] - k3) / (k4 - k3)))

    spec *= w

    y = np.fft.irfft(spec, n=N)         # zero-phase inverse (implicit 1/N)
    return y[:n]


# ---------------------------------------------------------------------------
# Minimal SEG-Y I/O (big-endian, standard Rev-1 layout)
# ---------------------------------------------------------------------------

SAMPLE_FORMATS = {1: 4, 2: 4, 3: 2, 5: 4}   # format code -> bytes per sample
IBM = 1
INT32 = 2
INT16 = 3
IEEE = 5


# --- IBM 32-bit float <-> IEEE float64 (numpy vectorised) ------------------
def ibm_to_ieee(raw: bytes) -> np.ndarray:
    u = np.frombuffer(raw, dtype=">u4").astype(np.uint32)
    sign = np.where((u >> 31) & 1 == 1, -1.0, 1.0)
    exp = ((u >> 24) & 0x7F).astype(np.float64)
    frac = (u & 0xFFFFFF).astype(np.float64)
    return sign * frac * np.power(16.0, exp - 64.0) / 16777216.0


def ieee_to_ibm(vals: np.ndarray) -> bytes:
    v = np.asarray(vals, dtype=np.float64).reshape(-1)
    out = np.zeros(v.size, dtype=np.uint32)
    nz = v != 0.0
    if not nz.any():
        return out.astype(">u4").tobytes()

    x = np.abs(v[nz])
    m2, e2 = np.frexp(x)              # x = m2 * 2**e2, m2 in [0.5, 1)
    e16 = np.floor(e2 / 4.0).astype(np.int64)      # 16-base exponent (floor)
    t = (e2 - 4.0 * e16).astype(np.float64)
    m16 = m2 * np.power(2.0, t)       # in [0.5, 8)
    over = m16 >= 1.0                 # normalise fraction into [1/16, 1)
    m16 = np.where(over, m16 / 16.0, m16)
    e16 = np.where(over, e16 + 1, e16)

    frac = np.rint(m16 * 16777216.0).astype(np.int64)
    frac = np.clip(frac, 0, (1 << 24) - 1)

    e8 = e16 + 64
    zero = e8 < 0
    e8 = np.clip(e8, 0, 255)
    sign = np.where(v[nz] < 0, np.uint32(0x80000000), np.uint32(0))
    val = sign | (e8.astype(np.uint32) << 24) | frac.astype(np.uint32)
    val[zero] = 0
    out[nz] = val
    return out.astype(">u4").tobytes()


def read_trace_samples(raw: bytes, fmt: int) -> np.ndarray:
    if fmt == IEEE:
        return np.frombuffer(raw, dtype=">f4").astype(np.float64)
    if fmt == IBM:
        return ibm_to_ieee(raw)
    if fmt == INT32:
        return np.frombuffer(raw, dtype=">i4").astype(np.float64)
    if fmt == INT16:
        return np.frombuffer(raw, dtype=">i2").astype(np.float64)
    raise ValueError("unsupported SEG-Y sample format code %d" % fmt)


def write_trace_samples(vals: np.ndarray, fmt: int) -> bytes:
    if fmt == IEEE:
        return np.asarray(vals, dtype=np.float64).astype(">f4").tobytes()
    if fmt == IBM:
        return ieee_to_ibm(vals)
    if fmt == INT32:
        iv = np.rint(np.clip(vals, -2 ** 31, 2 ** 31 - 1)).astype(">i4")
        return iv.tobytes()
    if fmt == INT16:
        iv = np.rint(np.clip(vals, -2 ** 15, 2 ** 15 - 1)).astype(">i2")
        return iv.tobytes()
    raise ValueError("unsupported SEG-Y sample format code %d" % fmt)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _u16(buf, off):
    return int.from_bytes(buf[off:off + 2], "big")


def _resolve_trace_start(blob, fmt, nsamp):
    """Return the byte offset of the first trace.

    SEG-Y puts the trace area right after the 3600-byte header, optionally
    followed by N x 3200-byte *extended textual headers* (count in the binary
    header at offset 3502).  That counter is not always reliable (e.g. GeoEast
    files declare 1 but write no extra block), so we do not trust it blindly:
    we try the declared count first, then 0, and pick the layout under which
    the file divides into a whole number of fixed-size traces and the trace
    header sample count is consistent.
    """
    bps = SAMPLE_FORMATS[fmt]
    bpt = 240 + nsamp * bps
    declared = _u16(blob, 3502)

    cands = [declared] if declared > 0 else []
    if 0 not in cands:
        cands.append(0)

    for e in cands:
        off = 3600 + 3200 * e
        if off + 240 > len(blob):
            continue
        if (len(blob) - off) % bpt != 0:
            continue
        # trace-header "samples in this trace" should match (or be 0/absent)
        hns = _u16(blob, off + 114)
        if hns in (0, nsamp):
            return off

    # nothing clean -> use the declared layout (may be wrong, but best effort)
    return 3600 + 3200 * declared


def process_sgy(in_path: str, out_path: str, f1: float, f2: float,
                f3: float, f4: float, dt_us_override=None):
    with open(in_path, "rb") as fh:
        blob = bytearray(fh.read())

    if len(blob) < 3600:
        raise ValueError("file too short to be SEG-Y")

    # binary header fields (0-based file offsets)
    dt_us = _u16(blob, 3216)
    nsamp = _u16(blob, 3220)
    fmt = _u16(blob, 3224)

    if fmt not in SAMPLE_FORMATS:
        raise ValueError("unsupported SEG-Y sample format code %d" % fmt)
    if nsamp <= 0:
        raise ValueError("number of samples per trace is 0 in binary header")

    if dt_us_override is not None:
        dt_us = int(dt_us_override)
    if dt_us <= 0:
        raise ValueError("sample interval is 0 (pass --dt-us to override)")

    if not (0.0 <= f1 < f2 < f3 < f4):
        print("warning: F1..F4 must satisfy 0<=F1<F2<F3<F4; nothing filtered",
              file=sys.stderr)
        sys.exit(2)

    bps = SAMPLE_FORMATS[fmt]
    trace_off = _resolve_trace_start(blob, fmt, nsamp)
    pos = trace_off
    n = 0

    total = (len(blob) - trace_off) // (240 + nsamp * bps)
    print(f"SEG-Y: ~{total} trace(s), {nsamp} samples/trace, "
          f"dt={dt_us} us, format={fmt} | trace data starts at byte {trace_off}")

    while pos + 240 <= len(blob):
        # per-trace sample count (trace header offset 114), fall back to global
        tr_ns = _u16(blob, pos + 114)
        if tr_ns <= 0:
            tr_ns = nsamp
        # per-trace sample interval (trace header offset 116), fall back to global
        tr_dt_us = _u16(blob, pos + 116)
        tr_dt = (tr_dt_us if tr_dt_us > 0 else dt_us) * 1e-6

        need = 240 + tr_ns * bps
        if pos + need > len(blob):
            print(f"warning: truncated last trace at byte {pos}, stopping",
                  file=sys.stderr)
            break

        samples = read_trace_samples(bytes(blob[pos + 240: pos + need]), fmt)
        filt = bandpass_seisee(samples, tr_dt, f1, f2, f3, f4)
        blob[pos + 240: pos + need] = write_trace_samples(filt, fmt)

        pos += need
        n += 1
        if n % 2000 == 0:
            print(f"  ...{n} traces filtered", file=sys.stderr)

    with open(out_path, "wb") as fh:
        fh.write(blob)

    print(f"filtered {n} trace(s) with F1={f1}, F2={f2}, F3={f3}, F4={f4} Hz "
          f"-> {out_path}")
    return n


def main(argv=None):
    p = argparse.ArgumentParser(
        description="SeiSee-style FFT band-pass (F1..F4 cosine taper) on SEG-Y")
    p.add_argument("input", help="input SEG-Y file")
    p.add_argument("output", help="output SEG-Y file")
    p.add_argument("--f1", type=float, default=10.0,
                   help="low-cut frequency, Hz (SeiSee default 10)")
    p.add_argument("--f2", type=float, default=20.0,
                   help="low-pass corner, Hz (SeiSee default 20)")
    p.add_argument("--f3", type=float, default=100.0,
                   help="high-pass corner, Hz (SeiSee default 100)")
    p.add_argument("--f4", type=float, default=150.0,
                   help="high-cut frequency, Hz (SeiSee default 150)")
    p.add_argument("--dt-us", type=int, default=None,
                   help="override sample interval (microseconds)")
    args = p.parse_args(argv)

    process_sgy(args.input, args.output,
                args.f1, args.f2, args.f3, args.f4,
                dt_us_override=args.dt_us)


if __name__ == "__main__":
    main()
