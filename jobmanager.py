# -*- coding: utf-8 -*-
"""jobmanager.py — 作业(Job)、任务池、租约锁、保存协调（多用户中央服务核心）

不修改任何核心读取/出图模块。所有会改 store/租约的方法都在作业级锁内执行，
单进程内多线程安全；本模块是纯逻辑，可直接单测（用合成 Gather + fake provider）。
"""
from __future__ import annotations
import json, os, random, tempfile, threading, time
from collections import deque
from datetime import datetime
import numpy as np

from gather import Gather, extract_gathers, key_str, ranges_label
from preprocess import apply_pipeline, sample_clip_values
from storage import LabelStore
from imaging import render_data_only

TTL_SECONDS = 30 * 60

# 每次保存标注固定增强张数：同一道集随机采 N_AUG 个不同 clip 值各渲一张导出图。
# （用户拍板 v3.6：不再建作业定单个 clip 作为导出，保存即存多张随机 clip 图。）
N_AUG = 5

class JobError(Exception):
    pass


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

    def display(self, gid: str) -> np.ndarray:
        return apply_pipeline(self.raw(gid),
                              [{"name": "clip_percentile", "params": {"percentile": self.clip}}])

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

    def display_image(self, gid: str) -> str:
        """返回界面显示图绝对路径（.cache/<gid>.png，缺则渲染）。

        显示图只用做标注/预览（预览 clip=self.clip），绝不写入导出 images/；
        导出图只在保存时生成 N_AUG 张 <gid>__clip<值>.png，从而保证 images/
        与 labels.jsonl 记录一一对应、无孤立图。
        """
        abs_p = os.path.join(self.cache_dir, f"{gid}.png")
        if not os.path.isfile(abs_p):
            self._render_png(self.display(gid), abs_p)
        return abs_p

    def _render_png(self, data: np.ndarray, abs_path: str) -> str:
        """原子渲染 PNG：先写同目录临时文件再 os.replace，避免并发读到半截文件。"""
        d = os.path.dirname(abs_path)
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".png")
        os.close(fd)
        try:
            render_data_only(data, out_path=tmp)
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
    def __init__(self, jobs_root: str, cfg, ttl_seconds: int = TTL_SECONDS):
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

    def _job_meta(self, job_id: str, title: str, path: str, endian: str,
                  sort_keys, extract_keys, values, clip: float,
                  aug_lo: float, aug_hi: float, created_by: str,
                  output_dir: str, min_traces: int = 0) -> dict:
        return {"job_id": job_id, "title": title, "source_file": os.path.abspath(path),
                "endian": endian, "sort_keys": [list(k) for k in sort_keys],
                "extract_keys": [list(k) for k in extract_keys],
                "values": _json_values(values),
                # clip_percentile = 预览 clip（仅界面显示）；导出用 aug_lo~aug_hi 随机 N_AUG 张
                "clip_percentile": float(clip),
                "aug_lo": float(aug_lo), "aug_hi": float(aug_hi),
                "min_traces": int(min_traces or 0),
                "created_by": created_by,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "state": "open", "output_dir": output_dir}

    def create_job(self, source_file: str, sort_keys, extract_keys, values,
                   clip: float | None = None, title: str = "", created_by: str = "",
                   endian: str = "auto", min_traces: int = 0,
                   aug_lo: float = 90.0, aug_hi: float = 99.9) -> Job:
        """从真实 sgy 抽道集建作业：写 job.json + 注册。

        values: 单字段 list[int] / 多字段 list[tuple]。
        clip: 预览 clip（仅界面显示/画框参照，默认 99）。
        aug_lo/aug_hi: 保存时随机采样的 clip 范围（每张道集存 N_AUG 张）。
        min_traces: 道数 < 该值的道集从任务池滤去（不标注）；0=不过滤。
        """
        clip = 99.0 if clip is None else clip
        if not (0 < aug_lo <= aug_hi <= 100):
            raise JobError("增强 clip 上下限须满足 0 < 下限 ≤ 上限 ≤ 100")
        r, gs = self._extract(source_file, sort_keys, extract_keys, values, endian)
        gs = self._filter_min_traces(gs, min_traces)
        if not gs:
            limit = int(min_traces or 0)
            raise JobError(
                "没有抽到任何道集" if limit <= 0 else f"抽到的道集道数都小于 {limit}，无可用道集")
        job_id = f"{os.path.splitext(os.path.basename(source_file))[0]}__{datetime.now():%Y%m%d%H%M%S}"
        output_dir = os.path.join(self.jobs_root, job_id)
        os.makedirs(output_dir, exist_ok=True)
        meta = self._job_meta(job_id, title, source_file, endian,
                              sort_keys, extract_keys, values, clip,
                              aug_lo, aug_hi, created_by, output_dir,
                              min_traces=min_traces)
        with open(os.path.join(output_dir, "job.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        ns = r.info()["ns"]
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
                job = self.register_job(meta, gs, r.info()["ns"],
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
             save_npy: bool = True):
        """保存标注：每张道集写 1 条基础记录(image_path=null) + N_AUG 条增强记录。

        增强记录 = 从原始数据按 aug_lo~aug_hi 随机采 N_AUG 个不同 clip 值各渲一张
        images/<gather_id>__clip<值>.png（labels/sentence/regions 与基础一致）。
        重开/重复保存先删旧增强记录与其图片再写，避免堆积。

        返回 (ok, base_record, aug_records, err)；失败时 base/aug 均为 None/[]。
        """
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
            aug_records = []
            for cp in clips:
                d = apply_pipeline(raw, [{"name": "clip_percentile",
                                          "params": {"percentile": float(cp)}}])
                aug_id = f"{gather_id}__clip{cp:g}"
                rel_img = Job.aug_image_rel(gather_id, cp)
                job._render_png(d, os.path.join(job.output_dir, rel_img))
                aug_rec = dict(base)
                aug_rec.update({"gather_id": aug_id, "image_path": rel_img,
                                "augmented_from": gather_id, "clip_percentile": float(cp)})
                aug_records.append(job.store.upsert(aug_rec))
            # 基础记录：仅用于断点续标/进度/重开，无导出图
            base["image_path"] = None
            rec = job.store.upsert(base)
            # 释放本次租约（回改场景无租约，幂等）
            L = self._leases_of(job_id)
            if L.get(gather_id) and L[gather_id][0] == user:
                L.pop(gather_id, None)
            return True, rec, aug_records, None
