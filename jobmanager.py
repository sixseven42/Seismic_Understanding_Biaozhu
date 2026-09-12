# -*- coding: utf-8 -*-
"""jobmanager.py — 作业(Job)、任务池、租约锁、保存协调（多用户中央服务核心）

不修改任何核心读取/出图模块。所有会改 store/租约的方法都在作业级锁内执行，
单进程内多线程安全；本模块是纯逻辑，可直接单测（用合成 Gather + fake provider）。
"""
from __future__ import annotations
import json, logging, os, random, tempfile, threading, time
from collections import deque
from datetime import datetime
import numpy as np

log = logging.getLogger("jobmanager")

from gather import Gather, extract_gathers, key_str, ranges_label
from preprocess import apply_pipeline, clip_bound, sample_clip_values
from storage import LabelStore
from imaging import render_data_only

TTL_SECONDS = 30 * 60

# 每次保存标注固定增强张数：同一道集随机采 N_AUG 个不同 clip 值各渲一张导出图。
# （用户拍板 v3.6：不再建作业定单个 clip 作为导出，保存即存多张随机 clip 图。）
N_AUG = 5

# 增强图默认后台渲染（见 AugRenderQueue）。测试把它置 False：断言「存完图就在」才稳定，
# 否则「images/ 应为空」这类断言会被上一张还在后台画的图偶发打脸。
# 测试模块里写一行 `jobmanager.AUG_ASYNC_DEFAULT = False` 即可全局切换。
AUG_ASYNC_DEFAULT = True

class JobError(Exception):
    pass


class AugRenderQueue:
    """保存之后的增强图渲染队列（守护线程 + 按道集去重，后来者覆盖）。

    为什么要有它：N_AUG 张增强图是给**训练导出 / 云回传**用的，跟标注者「下一张标什么」
    毫无关系，却曾经占掉「保存并释放」的绝大部分等待 —— 实测 528 道 × 4000 采样下
    0.64s / 0.78s（82%），本机之外只会更久。改成记录同步落盘、图后台补画：
    记录立刻可用于断点续标/进度/重开，图晚几百毫秒完全无妨。

    单线程是有意的：多用户并发保存时本来就是各自请求线程在并发调 matplotlib，
    这里串行反而比现状更保守。吞吐约 N_AUG×0.1s ≈ 0.5s/张，足够跟上标注节奏。
    """

    def __init__(self, render_one):
        """render_one(job, gather_id, aug_rec) -> rel_img|None，由 JobManager 注入。"""
        self._render_one = render_one
        self.on_done = None                       # fn(job, rels)，见 set_hook
        self._pending: dict[tuple, dict] = {}     # (job_id, gid) -> task（去重，后来者覆盖）
        self._order: deque = deque()
        self._in_flight = None
        self._cv = threading.Condition()
        self._thread = None
        self._stop = threading.Event()

    # ---- 入队 ----
    def submit(self, job, gather_id: str, aug_records: list[dict]) -> None:
        key = (job.job_id, gather_id)
        with self._cv:
            if key not in self._pending:
                self._order.append(key)
            # 同一张再存一次 → 直接顶掉排队中的旧任务（旧图已被 remove_augmented 删掉）
            self._pending[key] = {"job": job, "gid": gather_id, "recs": list(aug_records)}
            self._ensure_thread()
            self._cv.notify_all()

    def _ensure_thread(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="aug-render", daemon=True)
        self._thread.start()

    # ---- 工作线程 ----
    def _loop(self):
        while not self._stop.is_set():
            with self._cv:
                while not self._order and not self._stop.is_set():
                    self._cv.wait(0.5)
                if self._stop.is_set():
                    return
                key = self._order.popleft()
                task = self._pending.pop(key, None)
                self._in_flight = key
            if task is None:
                with self._cv:
                    self._in_flight = None
                    self._cv.notify_all()
                continue
            try:
                self._run(task)
            except Exception:                            # noqa: BLE001 —— 后台绝不许把线程打死
                log.exception("增强图后台渲染失败：%s", task["gid"])
            finally:
                with self._cv:
                    self._in_flight = None
                    self._cv.notify_all()

    def _run(self, task):
        job, gid = task["job"], task["gid"]
        done = []                                        # 本次真正画出来的图（用于云回传）
        for rec in task["recs"]:
            if job.store.get(rec["gather_id"]) is None:
                # 这张道集在这中间又被保存了一次（重开修正）→ remove_augmented 已删掉这条
                # 旧增强记录。跳过，否则会把刚删掉的旧图**又画回来**，images/ 里就多出
                # 没有记录对应的孤立图。这个判据比「任务代号」更准：记录在，图就该在。
                log.info("增强图记录已被取代，跳过：%s", rec["gather_id"])
                continue
            rel = self._render_one(job, gid, rec)
            if rel:
                done.append(rel)
        self.finish(job, done)

    def finish(self, job, rels: list[str]):
        """一批增强图画完了 → 交给回调（web_app 挂的是云回传）。

        **同步与异步两条路都要走这里**：云上传依赖那几张 PNG 真的存在，
        早先只在后台队列里调回调，同步模式下就整批不上传了（测试逮到过）。

        rels 只含图片；labels.jsonl 在这里统一补上 —— 它是同步落盘的、那会儿早就在了。
        """
        if not rels or self.on_done is None:
            return
        self.on_done(job, list(rels) + ["labels.jsonl"])

    def set_hook(self, fn):
        """图渲染完之后的回调 fn(job, rels) —— 由 web_app 挂云上传。"""
        self.on_done = fn

    # ---- 等待 / 收尾 ----
    def drain(self, timeout: float = 60.0) -> bool:
        """等队列排空（含在飞的那张）。测试与脚本要「存完图就在」时用。超时返回 False。"""
        end = time.time() + timeout
        with self._cv:
            while (self._order or self._in_flight) and time.time() < end:
                self._cv.wait(0.1)
            return not self._order and not self._in_flight

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)


def _json_values(values):
    return [list(v) if isinstance(v, (tuple, list)) else int(v) for v in values]

def _restore_values(values):
    return [tuple(v) if isinstance(v, list) else int(v) for v in values]


class Job:
    """一个作业 = 一份 sgy 抽出的一批道集 + 独立输出目录/独立 store。"""
    def __init__(self, meta: dict, gathers: list[Gather], ns: int, provider):
        self.meta = meta
        self.job_id = meta["job_id"]
        self.title = meta.get("title", "")
        self.output_dir = meta["output_dir"]
        self.state = meta.get("state", "open")
        # clip 语义 v3.6 起 = 仅「界面预览/画框参照」用（导出改为保存时 N_AUG 个随机 clip）
        self.clip = float(meta.get("clip_percentile", 99.0))
        # 增强 clip 采样上下限：旧作业（job.json 无 aug_lo/aug_hi）缺失时 aug_ok=False，
        # 视为「需重建」——不能保存、load_all 会将其标 broken。
        self.aug_lo, self.aug_hi = self._read_aug(meta)
        self.ns = int(ns)
        # 采样间隔（毫秒）：面波区「滤波」按它把角频率换算到 FFT 频点。0 = 未知 →
        # 拒绝滤波（宁可不出图，也不能按猜的 dt 画错通带位置误导标注）。
        try:
            self.dt_ms = float(meta.get("dt_ms") or 0.0)
        except (TypeError, ValueError):
            self.dt_ms = 0.0
        if self.dt_ms <= 0:
            self.dt_ms = 0.0
        self._gathers = gathers
        self._by_id = {g.gather_id: g for g in gathers}
        self._provider = provider
        # 领取池：把原始（按数据值排序的）顺序打乱一次。claim 只从该池下发，
        # 使每个标注者拿到的是乱序道集而非连续数据段，降低标注疲劳。
        # 每次进程内注册/加载作业时随机 shuffle 一次；同一次运行内顺序固定。
        self._pool = list(self._gathers)
        random.Random().shuffle(self._pool)
        self.store = LabelStore(os.path.join(self.output_dir, "labels.jsonl"))

    @staticmethod
    def _read_aug(meta: dict) -> tuple[float | None, float | None]:
        """读增强 clip 上下限；缺失/非法即返回 (None, None)（视为旧版作业）。"""
        try:
            lo, hi = float(meta["aug_lo"]), float(meta["aug_hi"])
        except (KeyError, TypeError, ValueError):
            return None, None
        return (lo, hi) if (0 < lo <= hi <= 100) else (None, None)

    @property
    def aug_ok(self) -> bool:
        return self.aug_lo is not None and self.aug_hi is not None

    # ---- 池 / 元数据 ----
    def gather_ids(self) -> list[str]:
        return [g.gather_id for g in self._gathers]

    def gather(self, gid: str) -> Gather:
        try:
            return self._by_id[gid]
        except KeyError:
            raise JobError(f"道集不在本作业中: {gid}")

    def gather_meta(self, gid: str) -> dict:
        g = self.gather(gid)
        return {"gather_id": gid, "key": g.key, "value_text": g.value_text,
                "n_traces": g.n_traces}

    # ---- 数据 / 出图 ----
    def raw(self, gid: str) -> np.ndarray:
        return np.asarray(self._provider(self.gather(gid)), dtype=np.float32)

    @staticmethod
    def _bandpass_params(bandpass) -> dict | None:
        """归一化滤波参数：无滤波返回 None，四角缺一/非数值抛 JobError。

        绝不静默回落到「不滤波」——否则界面点了滤波却给出未滤波的图。
        """
        if not bandpass:
            return None
        try:
            return {k: float(bandpass[k]) for k in ("f1", "f2", "f3", "f4")}
        except (KeyError, TypeError, ValueError):
            raise JobError("滤波参数缺失或非数值")

    def display_vlim(self, gid: str) -> float:
        """显示色标界限：**只从原始数据**按预览 clip 分位算一次。

        滤波前后共用同一个界限 → 两张图色标一致、可直接对比；滤波压掉的能量
        会真的显示为变淡，而不是被重新拉伸回满对比度。
        """
        return clip_bound(self.raw(gid), self.clip)

    def _display_data(self, gid: str, bandpass: dict | None, vlim: float) -> np.ndarray:
        bp = self._bandpass_params(bandpass)
        steps = []
        if bp is not None:
            if self.dt_ms <= 0:
                raise JobError("无法确定采样间隔（job.json 缺 dt_ms），不能对显示图做滤波")
            steps.append({"name": "bandpass", "params": {**bp, "dt_ms": self.dt_ms}})
        steps.append({"name": "clip_abs", "params": {"vlim": vlim}})
        try:
            return apply_pipeline(self.raw(gid), steps)
        except ValueError as e:                      # preprocess 的参数校验
            raise JobError(str(e))

    def display(self, gid: str, bandpass: dict | None = None) -> np.ndarray:
        """界面显示数据：原始道集 →（可选）带通滤波 → 按共用色标截断。

        滤波只作用于「看」：导出图在 save 时由 self.raw() 渲染，不受本参数影响。
        bandpass 为 {"f1","f2","f3","f4"}（Hz），非法或 dt 未知时抛 JobError。
        """
        return self._display_data(gid, bandpass, self.display_vlim(gid))

    @staticmethod
    def aug_image_rel(gather_id: str, clip_value: float) -> str:
        """增强导出图相对路径 images/<gather_id>__clip<值>.png。"""
        return f"images/{gather_id}__clip{clip_value:g}.png"

    def npy_rel(self, gid: str) -> str:
        return f"npy/{gid}.npy"

    @property
    def cache_dir(self) -> str:
        """界面显示图缓存目录（.cache/，不入导出结果）。"""
        return os.path.join(self.output_dir, ".cache")

    @staticmethod
    def _filter_suffix(bp: dict) -> str:
        """滤波显示图的缓存后缀，如 __flt10-20-100-150（bp 已归一化）。

        带后缀的缓存与干净图 <gid>.png 互不覆盖 —— 「还原」瞬间命中干净缓存，
        也不会把干净图顶掉成滤波图。
        """
        return "__flt" + "-".join(f"{float(bp[k]):g}" for k in ("f1", "f2", "f3", "f4"))

    def display_image(self, gid: str, bandpass: dict | None = None) -> str:
        """返回界面显示图绝对路径（.cache/<gid>[__flt…].png，缺则渲染）。

        显示图只用做标注/预览（预览 clip=self.clip），绝不写入导出 images/；
        导出图只在保存时生成 N_AUG 张 <gid>__clip<值>.png，从而保证 images/
        与 labels.jsonl 记录一一对应、无孤立图。
        """
        bp = self._bandpass_params(bandpass)      # 先校验，参数不全就报错而非回落到干净图
        suffix = "" if bp is None else self._filter_suffix(bp)
        abs_p = os.path.join(self.cache_dir, f"{gid}{suffix}.png")
        if not os.path.isfile(abs_p):
            vlim = self.display_vlim(gid)
            self._render_png(self._display_data(gid, bandpass, vlim), abs_p, vlim=vlim)
        return abs_p

    def _render_png(self, data: np.ndarray, abs_path: str,
                    vlim: float | None = None) -> str:
        """原子渲染 PNG：先写同目录临时文件再 os.replace，避免并发读到半截文件。

        vlim 给定则按该色标界限归一（界面显示图：滤波前后同一色标）；
        不给则由 render_data_only 取数据自身最大值（导出增强图沿用原行为）。
        """
        d = os.path.dirname(abs_path)
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".png")
        os.close(fd)
        try:
            render_data_only(data, out_path=tmp, vlim=vlim)
            os.replace(tmp, abs_path)
        finally:
            if os.path.isfile(tmp):
                os.remove(tmp)
        return abs_path

    def to_meta(self) -> dict:
        return dict(self.meta)


def open_job(meta: dict, gathers: list[Gather], ns: int, provider) -> Job:
    return Job(meta, gathers, ns, provider)


def finalize_regions(cfg, selection: dict, boxes: dict):
    """bbox 特征：选择「不存在」→ 区域置 None；否则必须已画框。返回 (regions, errs)。"""
    regions, errs = {}, []
    for f in cfg.features:
        if not f.bbox:
            continue
        lab = selection.get(f.name)
        if lab == "不存在":
            regions[f.key] = None
            continue
        box = boxes.get(f.key)
        if not box:
            errs.append(f"「{f.name}」为「{lab}」但未画框")
        regions[f.key] = box or None
    return regions, errs


class JobManager:
    def __init__(self, jobs_root: str, cfg, ttl_seconds: int = TTL_SECONDS,
                 aug_async: bool | None = None):
        self.jobs_root = os.path.abspath(jobs_root)
        self.cfg = cfg
        self.ttl = ttl_seconds
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._leases: dict[str, dict[str, list]] = {}   # job_id -> {gid: [user, deadline_ts]}
        # job_id -> {user: deque[gid]}：用户近期「跳过」的道集，发任务时对本用户避让，
        # 避免同一标注者反复在两个道集间循环跳过（只在对其没有其它任务时才退回）。
        self._skipq: dict[str, dict[str, deque]] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._lock = threading.RLock()                  # 注册表级
        self._readers: dict[tuple, object] = {}         # (abspath,endian) -> SegyReader
        # 增强图后台渲染（保存时把 PNG 画图挪出请求，见 AugRenderQueue）
        self.aug_queue = AugRenderQueue(self._render_one)
        self.aug_async = AUG_ASYNC_DEFAULT if aug_async is None else bool(aug_async)

    # ---- 增强图后台渲染 ----
    def set_aug_hook(self, fn):
        """图渲染完之后的回调 fn(job, rels)（web_app 用它挂云回传）。

        放在这里而不是让 jobmanager 直接依赖 cloudsync：本模块是纯逻辑，不该认识云。
        """
        self.aug_queue.set_hook(fn)

    def drain_aug(self, timeout: float = 60.0) -> bool:
        """等后台增强图队列排空（含在飞的那张）。超时返回 False。"""
        return self.aug_queue.drain(timeout)

    def sweep_missing_aug(self) -> int:
        """补画「有增强记录但 PNG 不在」的图，返回排队的道集数。

        什么时候会缺：进程在后台渲染途中被杀。窗口是每次保存约 N_AUG×0.1s，很小，
        但「记录在、图缺失」是这次改动新引入的状态，得有自动退路 ——
        否则就得靠人记得去跑 offline 的 backfill_augment.py。
        离线脚本仍然保留（它还能处理更早版本留下的单图记录）。
        """
        queued = 0
        for job in self.jobs():
            by_gather: dict[str, list[dict]] = {}
            for rec in list(job.store.records.values()):
                src = rec.get("augmented_from")
                rel = rec.get("image_path")
                if not src or not rel:
                    continue
                if os.path.isfile(os.path.join(job.output_dir, rel)):
                    continue
                by_gather.setdefault(src, []).append(rec)
            for gid, recs in by_gather.items():
                self.aug_queue.submit(job, gid, recs)
                queued += 1
        if queued:
            log.info("补画缺失的增强图：%d 个道集", queued)
        return queued

    # ---- 注册 / 恢复 ----
    def _job_lock(self, job_id: str) -> threading.RLock:
        return self._locks.setdefault(job_id, threading.RLock())

    def register_job(self, meta: dict, gathers: list[Gather], ns: int, provider) -> Job:
        job = open_job(meta, gathers, ns, provider)
        with self._lock:
            if job.job_id in self._jobs:
                raise JobError(f"作业已存在: {job.job_id}")
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
        return job

    def get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise JobError(f"作业不存在: {job_id}")

    def _get_reader(self, path: str, endian: str = "auto"):
        key = (os.path.abspath(path), endian)
        with self._lock:
            r = self._readers.get(key)
            if r is None:
                from segy_reader import SegyReader
                r = SegyReader(path, endian=endian)
                self._readers[key] = r
            return r

    def _extract(self, path, sort_keys, extract_keys, values, endian="auto"):
        r = self._get_reader(path, endian)
        gs = extract_gathers(r, sort_keys, extract_keys, values)
        stem = os.path.splitext(os.path.basename(path))[0] + "__"
        for g in gs:
            g.prefix = stem
        return r, gs

    @staticmethod
    def _filter_min_traces(gs, min_traces):
        """按道数下限滤去道集：保留 g.n_traces >= min_traces；min_traces<=0 不过滤。"""
        try:
            limit = int(min_traces or 0)
        except (TypeError, ValueError):
            limit = 0
        if limit <= 0:
            return gs
        return [g for g in gs if g.n_traces >= limit]

    @staticmethod
    def _decimate(gs, decimate_n):
        """抽稀：按抽取顺序等间隔删道集，每 decimate_n 个保留 1 个（留第 1、1+n… 个）。

        decimate_n <= 1 或缺失 = 不抽稀。返回新列表（不改原 gs）。
        """
        try:
            n = int(decimate_n or 0)
        except (TypeError, ValueError):
            n = 0
        if n <= 1:
            return gs
        return gs[::n]

    def _job_meta(self, job_id: str, title: str, path: str, endian: str,
                  sort_keys, extract_keys, values, clip: float,
                  aug_lo: float, aug_hi: float, created_by: str,
                  output_dir: str, min_traces: int = 0, dt_ms: float = 0.0,
                  decimate_n: int = 1) -> dict:
        return {"job_id": job_id, "title": title, "source_file": os.path.abspath(path),
                "endian": endian, "sort_keys": [list(k) for k in sort_keys],
                "extract_keys": [list(k) for k in extract_keys],
                "values": _json_values(values),
                # clip_percentile = 预览 clip（仅界面显示）；导出用 aug_lo~aug_hi 随机 N_AUG 张
                "clip_percentile": float(clip),
                "aug_lo": float(aug_lo), "aug_hi": float(aug_hi),
                "min_traces": int(min_traces or 0),
                # 采样间隔（ms）：界面「滤波」按它换算角频率；缺则滤波不可用
                "dt_ms": float(dt_ms or 0.0),
                "decimate_n": int(decimate_n or 1),
                "created_by": created_by,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "state": "open", "output_dir": output_dir}

    def create_job(self, source_file: str, sort_keys, extract_keys, values,
                   clip: float | None = None, title: str = "", created_by: str = "",
                   endian: str = "auto", min_traces: int = 0,
                   aug_lo: float = 90.0, aug_hi: float = 99.9,
                   decimate_n: int = 1) -> Job:
        """从真实 sgy 抽道集建作业：写 job.json + 注册。

        values: 单字段 list[int] / 多字段 list[tuple]。
        clip: 预览 clip（仅界面显示/画框参照，默认 99）。
        aug_lo/aug_hi: 保存时随机采样的 clip 范围（每张道集存 N_AUG 张）。
        min_traces: 道数 < 该值的道集从任务池滤去（不标注）；0=不过滤。
        decimate_n: 抽稀间隔 —— 按抽取顺序每 N 个道集保留 1 个进任务池；<=1=不抽稀。
                    顺序为先按 min_traces 过滤、再抽稀（劣质道集先出局，抽稀只削好道集）。
                    两个参数都写入 job.json，启动恢复时同样施加，保证池子一致。
        """
        clip = 99.0 if clip is None else clip
        if not (0 < aug_lo <= aug_hi <= 100):
            raise JobError("增强 clip 上下限须满足 0 < 下限 ≤ 上限 ≤ 100")
        r, gs = self._extract(source_file, sort_keys, extract_keys, values, endian)
        gs = self._filter_min_traces(gs, min_traces)
        gs = self._decimate(gs, decimate_n)
        if not gs:
            limit = int(min_traces or 0)
            raise JobError(
                "没有抽到任何道集" if limit <= 0 else f"抽到的道集道数都小于 {limit}，无可用道集")
        job_id = f"{os.path.splitext(os.path.basename(source_file))[0]}__{datetime.now():%Y%m%d%H%M%S}"
        output_dir = os.path.join(self.jobs_root, job_id)
        os.makedirs(output_dir, exist_ok=True)
        info = r.info()
        meta = self._job_meta(job_id, title, source_file, endian,
                              sort_keys, extract_keys, values, clip,
                              aug_lo, aug_hi, created_by, output_dir,
                              min_traces=min_traces, dt_ms=info["dt_ms"],
                              decimate_n=decimate_n)
        with open(os.path.join(output_dir, "job.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        ns = info["ns"]
        job = self.register_job(meta, gs, ns, lambda g, r=r: r.get_traces(g.trace_indices))
        return job

    def load_all(self) -> list[Job]:
        """启动恢复：扫 jobs_root 下含 job.json 的目录，重抽道集重新注册。

        旧版作业（job.json 无 aug_lo/aug_hi，v3.5 及以前建的）一律标 broken——
        用户已拍板「不兼容，一律重建」，避免按单个固定 clip 的旧格式继续标注。
        抽不出数据同样标 broken。
        """
        if not os.path.isdir(self.jobs_root):
            os.makedirs(self.jobs_root, exist_ok=True)
        loaded = []
        for name in sorted(os.listdir(self.jobs_root)):
            jd = os.path.join(self.jobs_root, name)
            jf = os.path.join(jd, "job.json")
            if not os.path.isfile(jf):
                continue
            try:
                with open(jf, encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue
            if "aug_lo" not in meta or "aug_hi" not in meta:
                meta = dict(meta); meta["state"] = "broken"
                job = self.register_job(meta, [], 0, lambda g: np.zeros((0, 0), np.float32))
                loaded.append(job)
                continue
            try:
                r, gs = self._extract(
                    meta["source_file"],
                    [tuple(x) for x in meta["sort_keys"]],
                    [tuple(x) for x in meta["extract_keys"]],
                    _restore_values(meta["values"]), meta.get("endian", "auto"))
                gs = self._filter_min_traces(gs, meta.get("min_traces", 0))
                gs = self._decimate(gs, meta.get("decimate_n", 1))
                info = r.info()
                # 旧作业（无 dt_ms）从源文件补齐采样间隔，使界面「滤波」可用；
                # 仍取不到就保持 0（未知），届时拒绝滤波而不是按猜的 dt 画错通带
                try:
                    known = float(meta.get("dt_ms") or 0.0) > 0
                except (TypeError, ValueError):
                    known = False
                if not known:
                    meta["dt_ms"] = info["dt_ms"]
                job = self.register_job(meta, gs, info["ns"],
                                        lambda g, r=r: r.get_traces(g.trace_indices))
            except Exception:
                meta = dict(meta); meta["state"] = "broken"
                job = self.register_job(meta, [], 0, lambda g: np.zeros((0, 0), np.float32))
            loaded.append(job)
        return loaded

    def set_state(self, job_id: str, state: str):
        with self._lock, self._job_lock(job_id):
            job = self.get(job_id)
            job.state = state
            job.meta["state"] = state
            jf = os.path.join(job.output_dir, "job.json")
            if os.path.isfile(jf):
                with open(jf, "w", encoding="utf-8") as f:
                    json.dump(job.meta, f, ensure_ascii=False, indent=2)

    # ---- 任务池 / 租约 ----
    def _leases_of(self, job_id: str) -> dict:
        return self._leases.setdefault(job_id, {})

    def _skip_list(self, job_id: str, user: str) -> deque:
        """user 在该作业的近期跳过队列（deque<gid>，只在本用户发任务时避让）。"""
        return self._skipq.setdefault(job_id, {}).setdefault(user, deque())

    def current(self, job_id: str, user: str) -> str | None:
        """该用户当前有效租约的 gid；顺带清过期租约。线程安全，自持锁。"""
        with self._lock, self._job_lock(job_id):
            now = time.time()
            L = self._leases_of(job_id)
            keep = {}
            for gid, (u, dl) in L.items():
                if (u == user and dl > now) or dl > now:
                    keep[gid] = [u, dl]
            self._leases[job_id] = keep
            for gid in keep:
                if keep[gid][0] == user:
                    return gid
            return None

    def claim(self, job_id: str, user: str) -> dict:
        with self._lock, self._job_lock(job_id):
            job = self.get(job_id)
            if job.state != "open":
                return {"gid": None, "resume": False, "reason": "作业未开放或已删除"}
            now = time.time()
            L = self._leases_of(job_id)
            # 自己已有有效租约 → resume
            for gid, (u, dl) in list(L.items()):
                if u == user and dl > now:
                    L[gid] = [user, now + self.ttl]
                    return {"gid": gid, "resume": True, "reason": None}
            # 清过期
            for gid in [g for g, (u, dl) in L.items() if dl <= now]:
                L.pop(gid, None)
            q = self._skip_list(job_id, user)     # 本人近期跳过项
            # 第 1 轮（避让）：不打乱后的派发顺序下发，但避开本人近期跳过的道集，
            # 防止“跳过”在两个道集间来回振荡、永远走不到后面的任务。
            for g in job._pool:
                gid = g.gather_id
                if job.store.is_labeled(gid) or gid in L:
                    continue
                if gid in q:
                    continue
                L[gid] = [user, now + self.ttl]
                return {"gid": gid, "resume": False, "reason": None}
            # 第 2 轮（回退）：其余要么已标、被占、要么皆为本用户近期跳过 → 允许回到近期跳过项
            for g in job._pool:
                gid = g.gather_id
                if job.store.is_labeled(gid) or gid in L:
                    continue
                if gid in q:
                    q.remove(gid)                # 消费：回到该张即解除避让
                L[gid] = [user, now + self.ttl]
                return {"gid": gid, "resume": False, "reason": None}
            return {"gid": None, "resume": False, "reason": "已全部标完或全部被占用"}

    # ---- 多作业作业池（标注者可同时选多个作业，随机从并集里抽一张）----
    def _free_gids(self, job_id: str, user: str, skip_aware: bool = True) -> list[str]:
        """该作业里 user 当前可领的 gid（按派发顺序）：未标注、未被占；
        skip_aware=True 时避开本人近期跳过项。顺带清掉过期租约。"""
        job = self.get(job_id)
        now = time.time()
        L = self._leases_of(job_id)
        for gid in [g for g, (_, dl) in L.items() if dl <= now]:
            L.pop(gid, None)                       # 过期租约回收
        q = self._skip_list(job_id, user) if skip_aware else ()
        out = []
        for g in job._pool:
            gid = g.gather_id
            if job.store.is_labeled(gid) or gid in L:
                continue
            if skip_aware and gid in q:
                continue
            out.append(gid)
        return out

    def claim_pool(self, job_ids, user: str) -> dict:
        """从**多个作业组成的池子**里随机抽一张道集。

        抽取口径（用户拍板）：把所有选中作业的可领道集**摊平成一个池**，每个道集等概率
        —— 等价于「按各作业的道集数量比例」抽，作业越大被抽中越多。

        返回 {"gid", "job_id", "resume", "reason"}：
          1. 若在这些作业里已持有有效租约 → 直接回到那一张（不给用户丢未保存的标注）；
          2. 否则先只从「本人近期没跳过的」候选里抽；抽不到才回到跳过项（并解除其避让），
             保留单作业版的防振荡语义 —— 否则一跳过就可能立刻又抽到同一张；
          3. 一个都领不到 → gid=None，reason 说明原因（全标完 / 都被占 / 未开放）。
        """
        with self._lock:
            ids = [j for j in dict.fromkeys(job_ids or []) if j in self._jobs]
            if not ids:
                return {"gid": None, "job_id": None, "resume": False,
                        "reason": "请先在「作业」里选择至少一个作业"}
            opened = [j for j in ids if self._jobs[j].state == "open"]
            now = time.time()
            # 1) 本人已有有效租约 → resume（多作业下也一样，避免切选择时丢未保存工作）
            for jid in opened:
                with self._job_lock(jid):
                    for gid, (u, dl) in self._leases_of(jid).items():
                        if u == user and dl > now:
                            self._leases_of(jid)[gid] = [user, now + self.ttl]
                            return {"gid": gid, "job_id": jid, "resume": True,
                                    "reason": None}
            # 2) 摊平候选后等概率抽
            for skip_aware in (True, False):
                pool = []
                for jid in opened:
                    with self._job_lock(jid):
                        pool.extend((jid, gid) for gid in
                                    self._free_gids(jid, user, skip_aware))
                if pool:
                    jid, gid = random.choice(pool)
                    with self._job_lock(jid):
                        q = self._skip_list(jid, user)
                        if gid in q:
                            q.remove(gid)          # 消费：回到该张即解除避让
                        self._leases_of(jid)[gid] = [user, now + self.ttl]
                    return {"gid": gid, "job_id": jid, "resume": False, "reason": None}
            # 3) 池子空了：给出可诊断的原因
            if not opened:
                return {"gid": None, "job_id": None, "resume": False,
                        "reason": "选中的作业都未开放或已删除"}
            n_lab, n_tot, busy = 0, 0, 0
            for jid in opened:
                with self._job_lock(jid):
                    lab, tot = self._raw_progress(jid)
                    n_lab += lab
                    n_tot += tot
                    busy += len(self._leases_of(jid))
            if n_tot and n_lab >= n_tot:
                reason = f"选中的作业已全部标注（{n_lab}/{n_tot}）"
            elif busy:
                reason = "选中的作业暂时没有可领的道集（其余正被别人标注）"
            else:
                reason = "选中的作业里没有可领的道集"
            return {"gid": None, "job_id": None, "resume": False, "reason": reason}

    def skip_release(self, job_id: str, user: str) -> str | None:
        """「跳过」的第一步：把 user 在该作业持有的道集放回池（不写记录），
        并计入本人近期跳过队列。返回被跳过的 gid（无则 None）。
        抽下一张交给调用方（多作业下要从**池子**抽，而不是从这个作业抽）。"""
        with self._lock, self._job_lock(job_id):
            L = self._leases_of(job_id)
            old = None
            for gid in [g for g, (u, _) in L.items() if u == user]:
                L.pop(gid, None)
                old = old or gid                       # 通常只有一个，取第一个
            if old:
                q = self._skip_list(job_id, user)
                if old not in q:
                    q.append(old)
            return old

    def pool_progress(self, job_ids) -> tuple[int, int, int]:
        """选中作业的汇总进度：(已标注总数, 道集总数, 作业个数)。"""
        n_lab = n_tot = n_job = 0
        with self._lock:
            for jid in dict.fromkeys(job_ids or []):
                if jid not in self._jobs:
                    continue
                with self._job_lock(jid):
                    lab, tot = self._raw_progress(jid)
                n_lab += lab
                n_tot += tot
                n_job += 1
        return n_lab, n_tot, n_job

    def renew(self, job_id: str, user: str) -> str | None:
        with self._lock, self._job_lock(job_id):
            gid = self.current(job_id, user)
            if gid:
                self._leases_of(job_id)[gid] = [user, time.time() + self.ttl]
            return gid

    def release(self, job_id: str, user: str) -> bool:
        """撤销 user 在该作业的全部租约（回池）。"""
        with self._lock, self._job_lock(job_id):
            L = self._leases_of(job_id)
            dropped = False
            for gid in [g for g, (u, _) in L.items() if u == user]:
                L.pop(gid, None); dropped = True
            return dropped

    def skip(self, job_id: str, user: str) -> dict:
        """「跳过」：把 user 当前持有的道集放回池（不写任何记录），
        并把该道集记入**本用户的近期跳过队列**，随后照常领取下一张——
        claim 对本用户会优先避开这些近期跳过项，直到没有其它任务才回到它们，
        避免“跳过”在两个道集之间来回振荡。

        返回与 claim 同构：{"gid", "resume", "reason"} + "skipped"。
        """
        with self._lock, self._job_lock(job_id):
            L = self._leases_of(job_id)
            old = None
            for gid in [g for g, (u, _) in L.items() if u == user]:
                L.pop(gid, None)
                old = old or gid                       # 通常只有一个，取第一个
            if old:
                q = self._skip_list(job_id, user)
                if old not in q:
                    q.append(old)
            res = self.claim(job_id, user)
            res["skipped"] = old
            return res

    # ---- 查询 ----
    def _info(self, job: Job) -> dict:
        with self._job_lock(job.job_id):
            ids = set(job.gather_ids())                 # 空池(如 broken) => 空集
            labeled = len(job.store.labeled_ids() & ids)
            total = len(ids) or len(job.store.records)  # broken 时以已有记录数兜底展示
        return {"job_id": job.job_id, "title": job.title, "state": job.state,
                "labeled": labeled, "total": total,
                "created_by": job.meta.get("created_by", ""),
                "created_at": job.meta.get("created_at", "")}

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [self._info(self._jobs[j]) for j in self._order
                    if self._jobs[j].state != "deleted"]

    def jobs(self) -> list[Job]:
        """全部作业对象（含已软删除的 —— 它们的记录与图还在本地，补画扫描要覆盖到）。"""
        with self._lock:
            return [self._jobs[j] for j in self._order]

    def deleted_jobs(self) -> list[dict]:
        """已软删除的作业（供管理区「恢复」）。本地文件仍保留。"""
        with self._lock:
            return [self._info(self._jobs[j]) for j in self._order
                    if self._jobs[j].state == "deleted"]

    def delete_job(self, job_id: str):
        """软删除：作业标记为 deleted，从 list_jobs/下拉/进度中隐藏；
        本地 jobs/<id> 结果文件一律不删除，可经 restore_job 恢复。"""
        self.set_state(job_id, "deleted")

    def restore_job(self, job_id: str):
        """把已软删除的作业恢复为 open（回到列表/进度，可继续领取标注）。"""
        self.set_state(job_id, "open")

    def user_counts(self) -> dict[str, int]:
        """各用户标注的**道集张数**（跨全部作业；已软删除的作业不计，与 list_jobs 口径一致）。

        只数**基础记录**（`image_path is None`）：一次保存会写 1 条基础 + N_AUG 条增强，
        增强记录也带 annotated_by，不过滤会把张数算成 ×(N_AUG+1)。
        annotated_by 在「他人覆盖/重开修改」时保留原标注者（见 save），所以归属稳定。
        """
        out: dict[str, int] = {}
        with self._lock:
            for jid in list(self._order):
                job = self._jobs.get(jid)
                if job is None or job.state == "deleted":
                    continue          # 软删除的作业与进度表一样不参与统计
                with self._job_lock(jid):
                    for rec in job.store.records.values():
                        if rec.get("image_path") is not None:
                            continue                       # 增强图记录，不是一张新道集
                        who = rec.get("annotated_by") or ""
                        if who:
                            out[who] = out.get(who, 0) + 1
        return out

    def user_count(self, user: str) -> int:
        """单个用户的标注总张数（跨全部作业）。"""
        return self.user_counts().get(user, 0)

    def _raw_progress(self, job_id: str) -> tuple[int, int]:
        """(已标注, 总数)；调用方需自带 job 锁。空池（broken）以已有记录数兜底。"""
        job = self.get(job_id)
        ids = set(job.gather_ids())
        labeled = len(job.store.labeled_ids() & ids)
        return labeled, (len(ids) or len(job.store.records))

    def progress(self, job_id: str):
        with self._lock, self._job_lock(job_id):
            info = self._info(self.get(job_id))
            return info["labeled"], info["total"]

    def record(self, job_id: str, gid: str) -> dict | None:
        with self._lock, self._job_lock(job_id):
            return self.get(job_id).store.get(gid)

    def mine(self, job_id: str, user: str) -> list[dict]:
        """该用户标过的记录（只含基础记录，不含增强副本，按写入顺序）。

        增强记录的 gather_id 带 __clip 后缀、不对应真实道集，供"我标注的"重开
        编辑会指向不存在的道集，故一律排除。
        """
        with self._lock, self._job_lock(job_id):
            job = self.get(job_id)
            return [job.store.records[g] for g in job.store.order
                    if job.store.records[g].get("annotated_by") == user
                    and "augmented_from" not in job.store.records[g]]

    # ---- 保存 ----
    def save(self, job_id: str, user: str, gather_id: str,
             selection: dict, boxes: dict, is_admin: bool = False,
             save_npy: bool = True, aug_sync: bool | None = None):
        """保存标注：每张道集写 1 条基础记录(image_path=null) + N_AUG 条增强记录。

        增强记录 = 从原始数据按 aug_lo~aug_hi 随机采 N_AUG 个不同 clip 值各渲一张
        images/<gather_id>__clip<值>.png（labels/sentence/regions 与基础一致）。
        重开/重复保存先删旧增强记录与其图片再写，避免堆积。

        **记录同步落盘，PNG 默认交给后台线程画**（见 AugRenderQueue）：返回时那些图可能
        还没生成，`aug_records` 里的 image_path 是「即将存在」的路径。要「返回时图就在」
        （测试、离线脚本）传 aug_sync=True，或先把 self.aug_async 置 False。

        返回 (ok, base_record, aug_records, err)；失败时 base/aug 均为 None/[]。
        """
        sync = (not self.aug_async) if aug_sync is None else bool(aug_sync)
        with self._lock, self._job_lock(job_id):
            job = self.get(job_id)
            try:
                g = job.gather(gather_id)
            except JobError as e:
                return False, None, [], str(e)

            if not job.aug_ok:
                return False, None, [], ("旧版作业未配置增强 clip 上下限（aug_lo/aug_hi），"
                                         "请用新版重新创建作业后再标注")

            errs = self.cfg.validate_selection(selection)
            if errs:
                return False, None, [], "尚有未选特征：" + "、".join(errs)
            regions, box_errs = finalize_regions(self.cfg, selection, boxes)
            if box_errs:
                return False, None, [], "；".join(box_errs)

            prev = job.store.get(gather_id)
            holding = self.current(job_id, user) == gather_id
            owner = bool(prev and prev.get("annotated_by") == user)
            if not (is_admin or holding or owner):
                # 找出占用者给提示
                holder = "未知"
                for gid, (u, dl) in self._leases_of(job_id).items():
                    if gid == gather_id and dl > time.time():
                        holder = u
                return False, None, [], f"该道集已被 {holder} 持有或标注，不能保存"

            try:
                clips = sample_clip_values(job.aug_lo, job.aug_hi, N_AUG)
            except ValueError as e:
                return False, None, [], f"增强 clip 范围过窄无法采 {N_AUG} 个值：{e}"

            # 清旧增强（重开/重复保存不堆积）：删旧增强记录 + 对应旧图
            img_dir = os.path.join(job.output_dir, "images")
            for old_id in job.store.remove_augmented(gather_id):
                p = os.path.join(img_dir, f"{old_id}.png")
                if os.path.isfile(p):
                    os.remove(p)

            raw = job.raw(gather_id)
            rel_npy = None
            if save_npy:
                abs_npy = os.path.join(job.output_dir, job.npy_rel(gather_id))
                os.makedirs(os.path.dirname(abs_npy), exist_ok=True)
                np.save(abs_npy, raw)
                rel_npy = job.npy_rel(gather_id)

            meta = job.meta
            # 重开修改/他人覆盖时保留原标注者（spec §5.4）；仅新记录用当前 user
            annotator = user
            if prev is not None and prev.get("annotated_by"):
                annotator = prev["annotated_by"]
            base = {
                "gather_id": gather_id,
                "gather_key": g.key,
                "gather_value": list(g.value) if isinstance(g.value, tuple) else int(g.value),
                "n_traces": g.n_traces,
                "labels": self.cfg.to_record_labels(selection),
                "sentence": self.cfg.render_sentence(selection),
                "regions": regions,
                "npy_path": rel_npy,
                "sort_keys": ", ".join(key_str(tuple(k)) for k in meta["sort_keys"]),
                "extract_key": ranges_label([tuple(k) for k in meta["extract_keys"]]),
                "source_file": meta["source_file"],
                "annotated_by": annotator,
            }
            # 增强记录**同步**落盘（纯 JSON，几毫秒）：断点续标/进度/重开读的都是
            # labels.jsonl，这些必须立刻正确 —— 晚的只是那几个 PNG。
            aug_records = []
            for cp in clips:
                aug_id = f"{gather_id}__clip{cp:g}"
                aug_rec = dict(base)
                aug_rec.update({"gather_id": aug_id,
                                "image_path": Job.aug_image_rel(gather_id, cp),
                                "augmented_from": gather_id, "clip_percentile": float(cp)})
                aug_records.append(job.store.upsert(aug_rec))
            # 基础记录：仅用于断点续标/进度/重开，无导出图
            base["image_path"] = None
            rec = job.store.upsert(base)
            # 释放本次租约（回改场景无租约，幂等）
            L = self._leases_of(job_id)
            if L.get(gather_id) and L[gather_id][0] == user:
                L.pop(gather_id, None)
            # PNG 画图挪出请求：标注者点完「保存并释放」立刻能进下一张
            self._render_aug(job, gather_id, aug_records, sync)
            return True, rec, aug_records, None

    def _render_aug(self, job, gather_id: str, aug_records: list[dict], sync: bool):
        """画增强图：sync 就地画完并回调，否则丢给后台队列（见 AugRenderQueue 的说明）。"""
        if not sync:
            self.aug_queue.submit(job, gather_id, aug_records)
            return
        done = []
        for rec in aug_records:
            rel = self._render_one(job, gather_id, rec)
            if rel:
                done.append(rel)
        self.aug_queue.finish(job, done)

    @staticmethod
    def _render_one(job, gather_id: str, rec: dict):
        """渲染一张增强图。返回相对路径；数据读不出来就记日志跳过（不炸后台线程）。"""
        try:
            raw = job.raw(gather_id)
            d = apply_pipeline(raw, [{"name": "clip_percentile",
                                      "params": {"percentile": float(rec["clip_percentile"])}}])
            rel = rec["image_path"]
            job._render_png(d, os.path.join(job.output_dir, rel))
            return rel
        except Exception:                                # noqa: BLE001
            log.exception("增强图渲染失败：%s", rec.get("gather_id"))
            return None
