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
from preprocess import apply_pipeline
from storage import LabelStore
from imaging import render_data_only

TTL_SECONDS = 30 * 60

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
        self.clip = float(meta.get("clip_percentile", 99.0))
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

    def image_rel(self, gid: str) -> str:
        return f"images/{gid}.png"

    def npy_rel(self, gid: str) -> str:
        return f"npy/{gid}.npy"

    @property
    def cache_dir(self) -> str:
        """界面显示图缓存目录（.cache/，不入导出结果）。"""
        return os.path.join(self.output_dir, ".cache")

    def display_image(self, gid: str) -> str:
        """返回界面显示图绝对路径（.cache/<gid>.png，缺则渲染）。

        显示图只用做标注/预览，绝不写入导出 images/；导出图只在保存时由
        ensure_image() 生成，从而保证 images/ 与 labels.jsonl 一一对应。
        """
        abs_p = os.path.join(self.cache_dir, f"{gid}.png")
        if not os.path.isfile(abs_p):
            self._render_png(self.display(gid), abs_p)
        return abs_p

    def ensure_image(self, gid: str) -> str:
        """导出图 images/<gid>.png（缺则渲染）。

        仅应在保存标注（JobManager.save）时调用：保存即把该道集固化为结果图。
        任何「只看不保存」的显示路径不得调用本方法，否则会在 images/ 产生
        无标注记录对应的孤立图（显示请用 display_image）。
        """
        abs_p = os.path.join(self.output_dir, self.image_rel(gid))
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
                  sort_keys, extract_keys, values, clip: float, created_by: str,
                  output_dir: str, min_traces: int = 0) -> dict:
        return {"job_id": job_id, "title": title, "source_file": os.path.abspath(path),
                "endian": endian, "sort_keys": [list(k) for k in sort_keys],
                "extract_keys": [list(k) for k in extract_keys],
                "values": _json_values(values), "clip_percentile": float(clip),
                "min_traces": int(min_traces or 0),
                "created_by": created_by,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "state": "open", "output_dir": output_dir}

    def create_job(self, source_file: str, sort_keys, extract_keys, values,
                   clip: float, title: str, created_by: str,
                   endian: str = "auto", min_traces: int = 0) -> Job:
        """从真实 sgy 抽道集建作业：写 job.json + 注册。values: 单字段 list[int] / 多字段 list[tuple]。
        min_traces: 道数 < 该值的道集从任务池滤去（不标注）；0=不过滤。"""
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
                              sort_keys, extract_keys, values, clip, created_by, output_dir,
                              min_traces=min_traces)
        with open(os.path.join(output_dir, "job.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        ns = r.info()["ns"]
        job = self.register_job(meta, gs, ns, lambda g, r=r: r.get_traces(g.trace_indices))
        return job

    def load_all(self) -> list[Job]:
        """启动恢复：扫 jobs_root 下含 job.json 的目录，重抽道集重新注册；抽不出标 broken。"""
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
        """该用户标过的记录，按写入顺序。"""
        with self._lock, self._job_lock(job_id):
            job = self.get(job_id)
            return [job.store.records[g] for g in job.store.order
                    if job.store.records[g].get("annotated_by") == user]

    # ---- 保存 ----
    def save(self, job_id: str, user: str, gather_id: str,
             selection: dict, boxes: dict, is_admin: bool = False,
             save_npy: bool = True):
        """返回 (ok, record, err)。成功即渲染缓存图+npy 并写 store、释放租约。"""
        with self._lock, self._job_lock(job_id):
            job = self.get(job_id)
            try:
                g = job.gather(gather_id)
            except JobError as e:
                return False, None, str(e)

            errs = self.cfg.validate_selection(selection)
            if errs:
                return False, None, "尚有未选特征：" + "、".join(errs)
            regions, box_errs = finalize_regions(self.cfg, selection, boxes)
            if box_errs:
                return False, None, "；".join(box_errs)

            prev = job.store.get(gather_id)
            holding = self.current(job_id, user) == gather_id
            owner = bool(prev and prev.get("annotated_by") == user)
            if not (is_admin or holding or owner):
                # 找出占用者给提示
                holder = "未知"
                for gid, (u, dl) in self._leases_of(job_id).items():
                    if gid == gather_id and dl > time.time():
                        holder = u
                return False, None, f"该道集已被 {holder} 持有或标注，不能保存"

            # 渲染缓存导出图（已存在则复用）
            abs_img = job.ensure_image(gather_id)
            rel_img = job.image_rel(gather_id)
            rel_npy = None
            if save_npy:
                abs_npy = os.path.join(job.output_dir, job.npy_rel(gather_id))
                os.makedirs(os.path.dirname(abs_npy), exist_ok=True)
                np.save(abs_npy, job.raw(gather_id))
                rel_npy = job.npy_rel(gather_id)

            meta = job.meta
            record = {
                "gather_id": gather_id,
                "gather_key": g.key,
                "gather_value": list(g.value) if isinstance(g.value, tuple) else int(g.value),
                "n_traces": g.n_traces,
                "labels": self.cfg.to_record_labels(selection),
                "sentence": self.cfg.render_sentence(selection),
                "regions": regions,
                "image_path": rel_img,
                "npy_path": rel_npy,
                "sort_keys": ", ".join(key_str(tuple(k)) for k in meta["sort_keys"]),
                "extract_key": ranges_label([tuple(k) for k in meta["extract_keys"]]),
                "source_file": meta["source_file"],
                "annotated_by": user,
            }
            # 重开修改/他人覆盖时保留原标注者（spec §5.4）；仅新记录用当前 user
            if prev is not None and prev.get("annotated_by"):
                record["annotated_by"] = prev["annotated_by"]
            rec = job.store.upsert(record)
            # 释放本次租约（回改场景无租约，幂等）
            L = self._leases_of(job_id)
            if L.get(gather_id) and L[gather_id][0] == user:
                L.pop(gather_id, None)
            return True, rec, None
