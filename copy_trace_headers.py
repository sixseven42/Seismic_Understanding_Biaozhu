#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将一个 SEG-Y 文件的 240 字节道头原地复制到另一个 SEG-Y 文件。

示例：
    python copy_trace_headers.py source.sgy target.sgy --backup

只改写 target.sgy 的每道道头；target.sgy 的文本头、二进制头和道数据保持不变。
"""

from __future__ import annotations

import argparse
import mmap
import os
import shutil
import sys
from pathlib import Path

from segy_reader import FILE_HEADER_SIZE, HEADER_SIZE, SegyReadError, SegyReader


def _same_file(source: Path, target: Path) -> bool:
    """判断两个路径是否指向同一个文件（包括硬链接）。"""
    try:
        return os.path.samefile(source, target)
    except FileNotFoundError:
        return source.resolve() == target.resolve()


def _reader_info(path: Path) -> dict:
    reader = SegyReader(str(path))
    try:
        return {
            "n_traces": reader.n_traces,
            "trace_bytes": reader.trace_bytes,
            "ns": reader.ns,
            "format_code": reader.format_code,
            "endian": reader.endian,
        }
    finally:
        # 释放 SegyReader 的只读 memmap，避免 Windows 上后续打开文件失败。
        reader.close()


def _show_progress(stage: str, current: int, total: int, *, force: bool = False) -> None:
    """在终端原地显示处理进度，最多约每 1% 刷新一次。"""
    if total <= 0:
        percent = 100
    else:
        percent = min(100, int(current * 100 / total))
    step = max(1, total // 100)
    if not force and current != total and current % step != 0:
        return
    width = 30
    filled = width if total <= 0 else int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r{stage}: [{bar}] {percent:3d}% ({current}/{total})")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def copy_trace_headers(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    backup: bool = False,
    verify: bool = True,
    progress: bool = True,
) -> int:
    """把 source 的全部道头复制到 target，并返回复制的道数。"""
    source_path = Path(source).expanduser()
    target_path = Path(target).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"源文件不存在: {source_path}")
    if not target_path.is_file():
        raise FileNotFoundError(f"目标文件不存在: {target_path}")
    if _same_file(source_path, target_path):
        raise ValueError("源文件和目标文件不能是同一个文件（也不能是硬链接）")

    source_info = _reader_info(source_path)
    target_info = _reader_info(target_path)
    if source_info["n_traces"] != target_info["n_traces"]:
        raise ValueError(
            "道数不一致，拒绝写入: "
            f"源文件 {source_info['n_traces']} 道，"
            f"目标文件 {target_info['n_traces']} 道"
        )
    if source_info["endian"] != target_info["endian"]:
        raise ValueError(
            "源文件和目标文件端序不一致，拒绝原始道头复制: "
            f"源文件 {source_info['endian']}，目标文件 {target_info['endian']}"
        )

    backup_path = target_path.with_name(target_path.name + ".bak")
    if backup:
        if backup_path.exists():
            raise FileExistsError(f"备份文件已存在，为避免覆盖请先处理: {backup_path}")
        shutil.copy2(target_path, backup_path)

    n_traces = source_info["n_traces"]
    with source_path.open("rb") as source_file, target_path.open("r+b") as target_file:
        source_map = mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_READ)
        target_map = mmap.mmap(target_file.fileno(), 0, access=mmap.ACCESS_WRITE)
        try:
            if progress:
                _show_progress("复制道头", 0, n_traces, force=True)
            for index in range(n_traces):
                source_offset = FILE_HEADER_SIZE + index * source_info["trace_bytes"]
                target_offset = FILE_HEADER_SIZE + index * target_info["trace_bytes"]
                header = source_map[source_offset:source_offset + HEADER_SIZE]
                if len(header) != HEADER_SIZE:
                    raise IOError(f"读取源文件第 {index + 1} 道道头失败")
                target_map[target_offset:target_offset + HEADER_SIZE] = header
                if progress:
                    _show_progress("复制道头", index + 1, n_traces)

            if progress:
                sys.stdout.write("正在写入磁盘，请等待...\n")
                sys.stdout.flush()
            target_map.flush()
            os.fsync(target_file.fileno())
            if progress:
                sys.stdout.write("磁盘写入完成\n")
                sys.stdout.flush()

            if verify:
                if progress:
                    _show_progress("校验道头", 0, n_traces, force=True)
                for index in range(n_traces):
                    source_offset = FILE_HEADER_SIZE + index * source_info["trace_bytes"]
                    target_offset = FILE_HEADER_SIZE + index * target_info["trace_bytes"]
                    if target_map[target_offset:target_offset + HEADER_SIZE] != source_map[
                        source_offset:source_offset + HEADER_SIZE
                    ]:
                        raise IOError(f"写入校验失败：第 {index + 1} 道道头不一致")
                    if progress:
                        _show_progress("校验道头", index + 1, n_traces)
        finally:
            target_map.close()
            source_map.close()

    return n_traces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将源 SEG-Y 的 240 字节道头复制到目标 SEG-Y，目标文件原地修改。"
    )
    parser.add_argument("source", help="道头来源 SEG-Y 文件")
    parser.add_argument("target", help="要被原地修改的目标 SEG-Y 文件")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="修改前创建 target.sgy.bak；若备份已存在则拒绝覆盖",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过写入后的逐道校验（大文件可略微节省时间）",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示终端进度条",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source_info = _reader_info(Path(args.source).expanduser())
        target_info = _reader_info(Path(args.target).expanduser())
        print(
            f"源文件: {args.source} ({source_info['n_traces']} 道, "
            f"{source_info['endian']} endian)"
        )
        print(
            f"目标文件: {args.target} ({target_info['n_traces']} 道, "
            f"{target_info['endian']} endian)"
        )
        n_traces = copy_trace_headers(
            args.source,
            args.target,
            backup=args.backup,
            verify=not args.no_verify,
            progress=not args.no_progress,
        )
    except (FileNotFoundError, FileExistsError, PermissionError, ValueError, SegyReadError, IOError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if args.backup:
        print(f"已创建备份: {args.target}.bak")
    print(f"完成：已复制 {n_traces} 道道头到 {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
