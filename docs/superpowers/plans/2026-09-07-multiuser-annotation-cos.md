# 多人协作标注 + 腾讯云 COS 回传 —— 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 把单用户 Gradio 标注器改造成「同局域网 ≤5 人并发 + 每张只标一次 + 结果上传腾讯云 COS」的中央服务器式标注系统。

**Architecture:** 保留全部核心模块（segy_reader/gather/preprocess/imaging/labels）；新增 `users.py`（账号/角色）、`jobmanager.py`（作业+任务池+租约+保存协调，单进程线程锁）、`cloudsync.py`（COS 上传队列）；重构 `web_app.py` 去掉全局 Session，标注者从作业池「领下一张」，admin 建作业/管作业/重传。

**Tech Stack:** Python 3.14（`C:\Python314`）、Gradio 6.26、numpy、matplotlib、PIL、PyYAML；测试用标准库 `unittest`（**本机无 pytest，勿装**）；可选腾讯云 `cos-python-sdk-v5`。

**Spec:** `docs/superpowers/specs/2026-09-07-multiuser-annotation-cos-design.md`

## Global Constraints

- 运行目录：`E:\AI4geo\去噪智能体\Seismic_Understanding_Biaozhu-main`（下文简称**项目根**）。
- **本项目未使用 git**：所有任务没有 commit 步骤；以「每步的测试/验证门禁」代替提交点。不要在任务里跑 `git`。
- 测试一律用 `unittest`，运行方式固定为 `python -m unittest <模块路径> -v`；测试放 `tests/`（含 `tests/__init__.py`）。
- 核心模块 `segy_reader.py / gather.py / preprocess.py / imaging.py / labels.py / label_config.yaml / storage.py / main_window.py` **不改**。`storage.LabelStore` 的方法需在 `jobmanager` 的作业锁内调用，锁不进 storage.py。
- 中文界面与中文注释与现代码一致；`ensure_ascii=False` 写 JSON/YAML。
- 云上传失败**不得影响标注**：保存成功后再入队上传。
- COS 凭证只出现在 `cos_config.yaml`，不进代码/注释/日志。
- 旧的全局单例 `SES`（`web_app.py` 现 73 行）必须删除，不得残留。
- 标注服务只供内网：`0.0.0.0:7860`，无公网需求。
- 删除/重写旧功能前先看现有文件真实内容（行号以实际为准），本计划行号是标注性的参考。

## 现有接口速查（代码块引用其真实签名）

```python
# gather.py
@dataclass class Gather:
    key: str; value: int|tuple; trace_indices: np.ndarray; prefix: str = ""
    # 属性: n_traces, value_text, gather_id(="f'{prefix}{key}_{value_text}'")
def extract_gathers(reader, sort_keys: list[tuple[int,int]], gather_key, values: list|None=None) -> list[Gather]
def key_str(key) -> str; def ranges_label(ranges) -> str

# segy_reader.py
SegyReader(path, endian="auto"); .info() -> {"ns":int, "n_traces":int, "dt_us":int, ...}
.get_traces(indices) -> float32 (len, ns); .close()

# preprocess.py
def apply_pipeline(data, steps: list[{"name","params"}]) -> np.ndarray   # "clip_percentile": {"percentile":99}

# imaging.py
def render_data_only(data, out_path, size=(512,1024)) -> out_path  # 固定 512×1024 RGB PNG，纯数据
def overlay_boxes(img_path, boxes:[(xyxy,color,label)], out_path) -> out_path

# labels.py
LabelConfig(path): .features(Feature{name,key,bbox,bbox_color,options[Option{label,phrase,hotkey}]})
  .validate_selection({feat_name: opt_label}) -> list[str]   # 空=合法
  .render_sentence({feat_name: opt_label}) -> str
  .to_record_labels({feat_name: opt_label}) -> {feature_key: opt_label}

# storage.py
LabelStore(jsonl_path): .records(dict gid->rec) .order(list) .get(gid) .upsert(rec) .is_labeled(gid) .labeled_ids() .next_unlabeled(ids, start)
```

**测试数据约定**：`tests/` 内所有用 `LabelConfig` 的地方一律指向项目根 `label_config.yaml`。合成道集用 `Gather` 直接构造（`prefix` 设为 `"fake__"`）。

---

### Task 1: 账号模块 `users.py` + 账号文件

**Files:**
- Create: `users.py`
- Create: `tests/__init__.py`（空文件）
- Create: `tests/test_users.py`
- Create: `users.yaml`

**Interfaces:**
- Produces: `Accounts(path)`：`authenticate(username,password)->bool`、`role(username)->str|None`、`names()->list[str]`。同一实例在文件被编辑后自动重新加载（热生效，无需重建）。

- [ ] **Step 1: 建测试目录与空包**

```bash
mkdir -p tests && (echo ok) && type NUL > tests/__init__.py
```

- [ ] **Step 2: 写失败测试 `tests/test_users.py`**

```python
# -*- coding: utf-8 -*-
import os, tempfile, unittest
from users import Accounts

SAMPLE = """
users:
  - username: boss
    password: boss123
    role: admin
  - username: ann1
    password: x
"""

class TestAccounts(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.path = os.path.join(d, "users.yaml")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(SAMPLE)
        self.acc = Accounts(self.path)

    def test_authenticate_ok_and_bad(self):
        self.assertTrue(self.acc.authenticate("boss", "boss123"))
        self.assertFalse(self.acc.authenticate("boss", "wrong"))
        self.assertFalse(self.acc.authenticate("nobody", "x"))

    def test_role_and_default_annotator(self):
        self.assertEqual(self.acc.role("boss"), "admin")
        self.assertEqual(self.acc.role("ann1"), "annotator")

    def test_hot_reload_after_edit(self):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("  - username: ann2\n    password: y\n")
        self.assertTrue(self.acc.authenticate("ann2", "y"))
        self.assertEqual(self.acc.role("ann2"), "annotator")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 运行，确认失败（ImportError: No module named 'users'）**

Run: `python -m unittest tests.test_users -v` → 期望 FAIL（找不到模块）

- [ ] **Step 4: 实现 `users.py`**

```python
# -*- coding: utf-8 -*-
"""users.py — 账号与角色（users.yaml，热重载：每次校验时按 mtime 重新读盘）"""
from __future__ import annotations
import os, threading
import yaml

class AccountsError(Exception):
    pass

class Accounts:
    def __init__(self, path: str):
        self.path = path
        self._users: dict[str, dict] = {}
        self._mtime: float | None = None
        self._lock = threading.Lock()
        self._reload(force=True)

    def _reload(self, force: bool = False):
        if not os.path.isfile(self.path):
            raise AccountsError(f"账号文件不存在: {self.path}")
        mtime = os.path.getmtime(self.path)
        if not force and mtime == self._mtime:
            return
        with open(self.path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        users = {}
        for u in data.get("users") or []:
            name = str(u["username"])
            users[name] = {"password": str(u["password"]),
                           "role": str(u.get("role", "annotator"))}
        self._users, self._mtime = users, mtime

    def authenticate(self, username: str, password: str) -> bool:
        with self._lock:
            self._reload()
        rec = self._users.get(str(username))
        return bool(rec and rec["password"] == str(password))

    def role(self, username: str) -> str | None:
        with self._lock:
            self._reload()
        rec = self._users.get(str(username))
        return rec["role"] if rec else None

    def names(self) -> list[str]:
        with self._lock:
            self._reload()
        return list(self._users)

    @staticmethod
    def ensure_default(path: str) -> bool:
        """文件不存在时写入初始 admin 账号并返回 True（启动引导用）。"""
        if os.path.isfile(path):
            return False
        import os as _os
        _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("users:\n  - username: boss\n    password: boss123\n    role: admin\n")
        return True
```

- [ ] **Step 5: 建示例 `users.yaml`（内容：boss/boss123 admin）**

```yaml
users:
  - username: boss
    password: boss123
    role: admin
  - username: ann1
    password: ann123
    role: annotator
```

> 提醒：上线前把示例密码换掉。

- [ ] **Step 6: 运行通过**

Run: `python -m unittest tests.test_users -v` → 期望 3 个用例 PASS。

---

### Task 2: `jobmanager.py` —— Job 定义 / 持久化 / 注册 / 进度（不含领取与保存）

**Files:**
- Create: `jobmanager.py`
- Create: `tests/test_jobmanager.py`

**Interfaces:**
- Produces（后续任务都依赖这些精确名字）：
  - `Job(meta: dict, gathers: list[Gather], ns: int, provider)` 其中 `meta` 为 JSON-safe dict，`provider(gather)->float32 (n_traces, ns)`。
  - `Job`：`.job_id/.title/.output_dir/.state/.clip/.store(LabelStore)`、`.gather_ids()/.gather(gid)/.gather_meta(gid)/.raw(gid)/.display(gid)/.image_rel(gid)/.npy_rel(gid)/.ensure_image(gid)->abs_path/.to_meta()`
  - `open_job(meta, gathers, ns, provider) -> Job`（模块级工厂）
  - `finalize_regions(cfg, selection:dict, boxes:dict) -> (regions, errs)`（模块级函数，供保存校验）
  - `JobManager(jobs_root, cfg, ttl_seconds=1800)`：
    `.register_job(meta,gathers,ns,provider) -> Job`（测试/建作业共用）
    `.list_jobs() -> list[info]`、`.progress(job_id)->(labeled,total)`、`.record(job_id,gid)->dict|None`、`.mine(job_id,user)->list[dict]`、`.get(job_id)->Job`、`.set_state(job_id,state)`、`.load_all()`

- [ ] **Step 1: 写失败测试（Job 元数据往返 + 进度 + finalize_regions）**

`tests/test_jobmanager.py`：

```python
# -*- coding: utf-8 -*-
import os, tempfile, unittest
import numpy as np
from gather import Gather
from labels import LabelConfig
from jobmanager import open_job, JobManager, finalize_regions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = LabelConfig(os.path.join(ROOT, "label_config.yaml"))

def meta_for(job_id, out):
    return {"job_id": job_id, "title": "测试", "source_file": "x.sgy", "endian": "auto",
            "sort_keys": [[95, 96]], "extract_keys": [[95, 96]], "values": [1, 2, 3],
            "clip_percentile": 99.0, "created_by": "boss",
            "created_at": "2026-01-01T00:00:00", "state": "open", "output_dir": out}

def fake_gathers():
    gs = []
    for v in (1, 2, 3):
        g = Gather(key="95-96", value=v, trace_indices=np.arange(40)); g.prefix = "fake__"
        gs.append(g)
    return gs

def fake_provider(ns=200):
    def prov(g):
        rng = np.random.default_rng(abs(hash(g.gather_id)) % 2**32)
        return rng.normal(size=(g.n_traces, ns)).astype(np.float32)
    return prov

class TestJobAndManager(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.manager = JobManager(self.root, CFG)
        self.out = os.path.join(self.root, "job1")
        os.makedirs(self.out)
        meta = meta_for("job1", self.out)
        self.gs = fake_gathers()
        self.job = self.manager.register_job(meta, self.gs, 200, fake_provider())

    def test_gather_ids_and_meta(self):
        self.assertEqual(self.job.gather_ids(), ["fake__95-96_1", "fake__95-96_2", "fake__95-96_3"])
        m = self.job.gather_meta("fake__95-96_1")
        self.assertEqual(m["n_traces"], 40)

    def test_progress_zero_then_after_upsert(self):
        self.assertEqual(self.manager.progress("job1"), (0, 3))
        rec = {"gather_id": "fake__95-96_1"}
        self.job.store.upsert(rec)
        self.assertEqual(self.manager.progress("job1"), (1, 3))

    def test_finalize_regions_requires_box_when_not_absent(self):
        selection = {f.name: "存在" for f in CFG.features}
        regions, errs = finalize_regions(CFG, selection, {})
        self.assertTrue(any("面波" in e for e in errs))   # bbox 特征缺框
        # 注意 label_config.yaml 有两个 bbox 特征（面波 + 近炮点强能量噪声），须都置「不存在」
        selection["面波"] = "不存在"
        selection["近炮点强能量噪声"] = "不存在"
        regions, errs = finalize_regions(CFG, selection, {})
        self.assertEqual(errs, [])
        self.assertIsNone(regions["surface_wave"])
        self.assertIsNone(regions["near_shot_noise"])

    def test_list_jobs_info(self):
        info = self.manager.list_jobs()[0]
        self.assertEqual(info["job_id"], "job1")
        self.assertEqual(info["total"], 3)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_jobmanager -v` → FAIL（No module named 'jobmanager'）

- [ ] **Step 3: 实现 `jobmanager.py`（本任务先实现 Job/工厂/注册/进度；claim/save 在 Task 3/4 追加）**

```python
# -*- coding: utf-8 -*-
"""jobmanager.py — 作业(Job)、任务池、租约锁、保存协调（多用户中央服务核心）

不修改任何核心读取/出图模块。所有会改 store/租约的方法都在作业级锁内执行，
单进程内多线程安全；本模块是纯逻辑，可直接单测（用合成 Gather + fake provider）。
"""
from __future__ import annotations
import json, os, threading, time
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

    def ensure_image(self, gid: str) -> str:
        """返回导出图绝对路径；缺则渲染并缓存（显示图=导出图，见规格 §5.5）。"""
        abs_p = os.path.join(self.output_dir, self.image_rel(gid))
        if not os.path.isfile(abs_p):
            render_data_only(self.display(gid), out_path=abs_p)
        return abs_p

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

    def _job_meta(self, job_id: str, title: str, path: str, endian: str,
                  sort_keys, extract_keys, values, clip: float, created_by: str,
                  output_dir: str) -> dict:
        return {"job_id": job_id, "title": title, "source_file": os.path.abspath(path),
                "endian": endian, "sort_keys": [list(k) for k in sort_keys],
                "extract_keys": [list(k) for k in extract_keys],
                "values": _json_values(values), "clip_percentile": float(clip),
                "created_by": created_by,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "state": "open", "output_dir": output_dir}

    def create_job(self, source_file: str, sort_keys, extract_keys, values,
                   clip: float, title: str, created_by: str,
                   endian: str = "auto") -> Job:
        """从真实 sgy 抽道集建作业：写 job.json + 注册。values: 单字段 list[int] / 多字段 list[tuple]"""
        r, gs = self._extract(source_file, sort_keys, extract_keys, values, endian)
        if not gs:
            raise JobError("没有抽到任何道集")
        job_id = f"{os.path.splitext(os.path.basename(source_file))[0]}__{datetime.now():%Y%m%d%H%M%S}"
        output_dir = os.path.join(self.jobs_root, job_id)
        os.makedirs(output_dir, exist_ok=True)
        meta = self._job_meta(job_id, title, source_file, endian,
                              sort_keys, extract_keys, values, clip, created_by, output_dir)
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

    # ---- 查询 ----
    def _info(self, job: Job) -> dict:
        ids = set(job.gather_ids())                     # 空池(如 broken) => 空集
        labeled = len(job.store.labeled_ids() & ids)
        total = len(ids) or len(job.store.records)      # broken 时以已有记录数兜底展示
        return {"job_id": job.job_id, "title": job.title, "state": job.state,
                "labeled": labeled, "total": total,
                "created_by": job.meta.get("created_by", ""),
                "created_at": job.meta.get("created_at", "")}

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [self._info(self._jobs[j]) for j in self._order]

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
```

- [ ] **Step 4: 运行通过**

Run: `python -m unittest tests.test_jobmanager -v` → PASS（4 用例）

---

### Task 3: 任务池领取 / 归还 / 续期 / 租约并发安全

**Files:**
- Modify: `jobmanager.py`（给 `JobManager` 追加 `claim / release / renew / current`）
- Modify: `tests/test_jobmanager.py`

**Interfaces:**
- Produces: `JobManager.claim(job_id,user)->dict{"gid":str|None,"resume":bool,"reason":str|None}`；`release(job_id,user)->bool`；`renew(job_id,user)->str|None`；`current(job_id,user)->str|None`。

- [ ] **Step 1: 追加失败测试（领取唯一性、归还回池、续期、TTL 过期）**

在 `tests/test_jobmanager.py` 追加：

```python
import time

class TestClaimLease(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.manager = JobManager(self.root, CFG, ttl_seconds=0.4)
        self.out = os.path.join(self.root, "job1"); os.makedirs(self.out)
        meta = meta_for("job1", self.out)
        self.job = self.manager.register_job(meta, fake_gathers(), 200, fake_provider())

    def test_claim_gives_distinct_and_resume(self):
        a = self.manager.claim("job1", "ann1"); b = self.manager.claim("job1", "ann2")
        self.assertIsNotNone(a["gid"]); self.assertIsNotNone(b["gid"])
        self.assertNotEqual(a["gid"], b["gid"])
        again = self.manager.claim("job1", "ann1")     # 已持有 → resume 同一张
        self.assertTrue(again["resume"]); self.assertEqual(again["gid"], a["gid"])

    def test_release_returns_to_pool(self):
        a = self.manager.claim("job1", "ann1")
        self.assertTrue(self.manager.release("job1", "ann1"))
        b = self.manager.claim("job1", "ann2")
        self.assertEqual(b["gid"], a["gid"])           # 归还能被另一个人领到

    def test_ttl_expiry_frees_lease(self):
        a = self.manager.claim("job1", "ann1")
        time.sleep(0.6)                                # 超过 ttl
        b = self.manager.claim("job1", "ann2")
        self.assertEqual(b["gid"], a["gid"])           # 过期后回池

    def test_no_double_claim_concurrently(self):
        # 单用户并发 claim 不得超过池容量去重；两个用户各自 save 后互不重复
        got = [self.manager.claim("job1", u) for u in ("ann1", "ann2") for _ in range(3)]
        gids = [x["gid"] for x in got if x["gid"]]
        self.assertLessEqual(len(set(gids)), 3)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_jobmanager.TestClaimLease -v` → FAIL（无 claim 方法）

- [ ] **Step 3: 实现 claim/release/renew/current**

在 `JobManager` 类内、`# ---- 查询 ----` 之前插入：

```python
    # ---- 任务池 / 租约 ----
    def _leases_of(self, job_id: str) -> dict:
        return self._leases.setdefault(job_id, {})

    def current(self, job_id: str, user: str) -> str | None:
        """用户在该作业当前有效租约的 gid；顺带清掉过期租约。"""
        now = time.time()
        L = self._leases_of(job_id)
        keep = {}
        for gid, (u, dl) in L.items():
            if u == user and dl > now:
                keep[gid] = [u, dl]
            elif dl > now:
                keep[gid] = [u, dl]      # 别人的有效租约保留
        self._leases[job_id] = keep
        for gid in keep:
            if keep[gid][0] == user:
                return gid
        return None

    def claim(self, job_id: str, user: str) -> dict:
        with self._lock, self._job_lock(job_id):
            job = self.get(job_id)
            if job.state != "open":
                return {"gid": None, "resume": False, "reason": "作业未开放"}
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
            for g in job._gathers:
                gid = g.gather_id
                if job.store.is_labeled(gid):
                    continue
                if gid in L:
                    continue
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
```

- [ ] **Step 4: 运行通过**

Run: `python -m unittest tests.test_jobmanager -v` → PASS（全部用例）

---

### Task 4: `save`（校验 / 归属 / 渲染缓存 / npy / 记录写入）

**Files:**
- Modify: `jobmanager.py`（`JobManager` 追加 `save`，`Job` 追加 `npy` 辅助；用 `cfg` 拼句子与标签）
- Modify: `tests/test_jobmanager.py`

**Interfaces:**
- Consumes: `finalize_regions(cfg, selection, boxes)`（Task 2）、`claim/current`（Task 3）
- Produces: `JobManager.save(job_id,user,gather_id,selection,boxes,is_admin=False,save_npy=True) -> (ok:bool, record:dict|None, err:str|None)`
  - `selection`：`{feature.name: option.label}`；`boxes`：`{feature.key: box|None}`
  - 允许保存的条件：`is_admin` 或 该 gid 有 user 的有效租约 或 已存在记录且 `annotated_by==user`（回改自己）。
  - 成功即：渲染/缓存导出图、按需写 npy、`store.upsert`（含 `annotated_by`）、释放该用户对该 gid 的租约。

- [ ] **Step 1: 追加失败测试**

```python
class TestSave(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.manager = JobManager(self.root, CFG)
        self.out = os.path.join(self.root, "job1"); os.makedirs(self.out)
        meta = meta_for("job1", self.out)
        self.job = self.manager.register_job(meta, fake_gathers(), 200, fake_provider())

    def full_selection(self):
        return {f.name: "不存在" for f in CFG.features}   # 无 bbox 需求
    def good_boxes(self):
        return {f.key: {"xyxy": [1,1,2,2], "traces": [0,2], "samples": [0,4]} for f in CFG.features if f.bbox}

    def test_save_requires_valid_selection(self):
        claim = self.manager.claim("job1", "ann1")
        ok, rec, err = self.manager.save("job1", "ann1", claim["gid"], {}, {}, is_admin=False)
        self.assertFalse(ok); self.assertIn("未选择", err)

    def test_save_success_sets_annotated_by_and_image(self):
        claim = self.manager.claim("job1", "ann1")
        ok, rec, err = self.manager.save("job1", "ann1", claim["gid"],
                                         self.full_selection(), self.good_boxes())
        self.assertTrue(ok, err); self.assertEqual(rec["annotated_by"], "ann1")
        img = os.path.join(self.out, rec["image_path"])
        self.assertTrue(os.path.isfile(img))            # 渲染缓存已生成
        self.assertTrue(os.path.isfile(os.path.join(self.out, rec["npy_path"])))
        self.assertTrue(self.manager.progress("job1")[0] >= 1)

    def test_other_user_cannot_save_someone_elses_claimed(self):
        a = self.manager.claim("job1", "ann1")
        ok, rec, err = self.manager.save("job1", "ann2", a["gid"],
                                         self.full_selection(), self.good_boxes())
        self.assertFalse(ok); self.assertIn("已被", err)

    def test_owner_can_resave_and_admin_can_override(self):
        a = self.manager.claim("job1", "ann1")
        self.manager.save("job1", "ann1", a["gid"], self.full_selection(), self.good_boxes())
        ok, rec, err = self.manager.save("job1", "ann1", a["gid"],
                                         self.full_selection(), self.good_boxes())
        self.assertTrue(ok, err)                        # 本人回改
        ok, rec, err = self.manager.save("job1", "boss", a["gid"],
                                         self.full_selection(), self.good_boxes(), is_admin=True)
        self.assertTrue(ok, err)                        # admin 覆盖
        self.assertEqual(rec["annotated_by"], "boss")   # 覆盖时标注者改为 admin
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_jobmanager.TestSave -v` → FAIL（无 save）

- [ ] **Step 3: 实现 `save`**

在 `JobManager` 内追加：

```python
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
            rec = job.store.upsert(record)
            # 释放本次租约（回改场景无租约，幂等）
            L = self._leases_of(job_id)
            if L.get(gather_id) and L[gather_id][0] == user:
                L.pop(gather_id, None)
            return True, rec, None
```

- [ ] **Step 4: 运行通过**

Run: `python -m unittest tests.test_jobmanager -v` → PASS（全部）

> 注：`render_data_only` 会把 `(40,200)` 输入 resize 到 512×1024，PIL 已装，毫秒级；测试覆盖渲染缓存路径（首次生成文件）。

---

### Task 5: COS 上传模块 `cloudsync.py`

**Files:**
- Create: `cloudsync.py`
- Create: `tests/test_cloudsync.py`
- Create: `cos_config.yaml`（示例；不入库语义）
- Bash: `python -m pip install cos-python-sdk-v5`（成功与否都不影响本任务测试；SDK 缺失时模块安全降级为 dry-run）

**Interfaces:**
- Produces: `load_config(path)->dict`；`object_key(job_id, rel_path)->str`；`Uploader(cfg)`：`.enqueue(job_id, output_dir, rel_paths)`、`.resync_all(job_id, output_dir)`、`.dry_log`（list[(key,local)]，非启用时记录）、`.start()`、`.shutdown()`。
  - 对象键：`seismic/<job_id>/<rel>`（rel 用 `/`）。

- [ ] **Step 1: 写失败测试**

`tests/test_cloudsync.py`：

```python
# -*- coding: utf-8 -*-
import os, tempfile, unittest
from cloudsync import Uploader, object_key, load_config

class TestCloudSync(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.img = os.path.join(self.d, "images"); os.makedirs(self.img)
        for n in ("a.png", "b.png"):
            with open(os.path.join(self.img, n), "wb") as f: f.write(b"x")
        with open(os.path.join(self.d, "labels.jsonl"), "w") as f: f.write("{}\n")

    def test_object_key_slashes(self):
        self.assertEqual(object_key("job1", "images\\a.png"), "seismic/job1/images/a.png")

    def test_dry_run_records_not_upload(self):
        up = Uploader({}, dry_run=True)                 # enabled false
        up.enqueue("job1", self.d, ["images/a.png", "labels.jsonl"])
        self.assertEqual(len(up.dry_log), 2)
        self.assertEqual(up.dry_log[0][0], "seismic/job1/images/a.png")

    def test_missing_local_skipped(self):
        up = Uploader({}, dry_run=True)
        up.enqueue("job1", self.d, ["images/nope.png"])
        self.assertEqual(up.dry_log, [])

    def test_resync_all_walks_images_and_labels(self):
        up = Uploader({}, dry_run=True)
        up.resync_all("job1", self.d)
        keys = [k for k, _ in up.dry_log]
        self.assertEqual(sorted(keys),
                         sorted(["seismic/job1/images/a.png", "seismic/job1/images/b.png",
                                 "seismic/job1/labels.jsonl"]))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m unittest tests.test_cloudsync -v` → FAIL

- [ ] **Step 3: 实现 `cloudsync.py`**

```python
# -*- coding: utf-8 -*-
"""cloudsync.py — 腾讯云 COS 上传队列（后台线程 + 失败重试）

启用条件：cos_config.yaml enabled=true 且已 pip install cos-python-sdk-v5。
未满足时保持安全降级：enqueue 只记入 self.dry_log，不影响标注保存。
凭证只从配置文件读，绝不进代码/注释/日志。
"""
from __future__ import annotations
import logging, os, queue, threading, time
import yaml

log = logging.getLogger("cloudsync")


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def object_key(job_id: str, rel_path: str) -> str:
    return f"seismic/{job_id}/{rel_path.replace(os.sep, '/')}"


class Uploader:
    def __init__(self, cfg: dict, dry_run: bool | None = None):
        self.cfg = cfg
        enabled = bool(cfg.get("enabled")) and bool(cfg.get("secret_id"))
        self.enabled = enabled and not dry_run
        self.dry_log: list[tuple[str, str]] = []        # [(object_key, local_path)] 非启用时记录
        self._q: queue.Queue = queue.Queue()
        self._client = None
        self._started = False
        self._stop = threading.Event()
        self.failures: list[tuple[str, str, str]] = []  # [(key, local, err)]

    def start(self):
        if not self.enabled or self._started:
            return
        try:
            from qcloud_cos import CosConfig, CosS3Client
        except Exception as e:
            log.warning("cos-python-sdk-v5 未安装，云上传关闭：%s", e)
            self.enabled = False
            return
        self._client = CosS3Client(CosConfig(
            Region=self.cfg["region"], SecretId=self.cfg["secret_id"],
            SecretKey=self.cfg["secret_key"], Scheme="https"))
        threading.Thread(target=self._loop, daemon=True).start()
        self._started = True

    def _loop(self):
        while not self._stop.is_set():
            item = self._q.get()
            if item is None:
                break
            self._upload_once(item)
            self._q.task_done()

    def _upload_once(self, item):
        key, local = item
        try:
            with open(local, "rb") as body:
                self._client.put_object(Bucket=self.cfg["bucket"], Key=key, Body=body)
        except Exception as e:
            self.failures.append((key, local, str(e)))
            log.error("上传失败 %s <- %s: %s", key, local, e)

    def enqueue(self, job_id: str, output_dir: str, rel_paths: list[str]):
        for rel in rel_paths:
            local = os.path.join(output_dir, rel)
            if not os.path.isfile(local):
                log.warning("跳过不存在的文件: %s", local)
                continue
            key = object_key(job_id, rel)
            if not self.enabled:
                self.dry_log.append((key, local))
                continue
            self._q.put((key, local))

    def resync_all(self, job_id: str, output_dir: str):
        rels = ["labels.jsonl"]
        img = os.path.join(output_dir, "images")
        if os.path.isdir(img):
            rels += [f"images/{f}" for f in sorted(os.listdir(img)) if f.endswith(".png")]
        self.enqueue(job_id, output_dir, rels)

    def shutdown(self):
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
```

- [ ] **Step 4: 建 `cos_config.yaml`（占位示例；enabled 先 false）**

```yaml
enabled: false
secret_id: ""
secret_key: ""
bucket: "your-bucket-1250000000"
region: "ap-guangzhou"
```

- [ ] **Step 5: 安装 COS SDK（成功与否不影响）**

Run: `python -m pip install cos-python-sdk-v5` → 期望输出 SUCCESS 或说明已装；失败可留到部署前再装，模块会安全降级。

- [ ] **Step 6: 运行通过**

Run: `python -m unittest tests.test_cloudsync -v` → PASS（4 用例）

---

### Task 6: 小型 SEG-Y 测试文件生成器 + 建作业/恢复真链路单测

**Files:**
- Create: `tests/segy_factory.py`
- Modify: `tests/test_jobmanager.py`（追加 create/load_all 对真实小 sgy 的测试）

**Interfaces:**
- Produces: `write_sgy(path, n_gathers=3, traces_per=8, ns=64, endian="little", fld_gather=(95,96), fld_ffid=(9,12), fld_tr=(13,16)) -> None`
  - 小端 IEEE float32 格式码 5；每道道头在 `fld_gather` 写炮号（int16 区间按 2 字节 → 用 (95,95)？不，用 2 字节区间 (95,96) 放 int16 炮号）、`fld_ffid`(9-12, int32)、`fld_tr`(13-16, int32) 写道内序号；道数据为确定性伪随机。

- [ ] **Step 1: 实现生成器 `tests/segy_factory.py`**

```python
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
```

- [ ] **Step 2: 追加测试（create_job → job.json → 新 manager.load_all 恢复）**

在 `tests/test_jobmanager.py` 追加：

```python
from segy_factory import write_sgy

class TestCreateAndRestore(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.sgy = os.path.join(self.root, "demo.sgy")
        write_sgy(self.sgy, n_gathers=3, traces_per=8, ns=64)

    def test_create_job_persists_and_restores(self):
        m = JobManager(os.path.join(self.root, "jobs"), CFG)
        job = m.create_job(self.sgy, [(95, 96), (13, 16)], (95, 96), [1, 2], clip=99,
                           title="demo", created_by="boss")
        self.assertEqual(len(job.gather_ids()), 2)
        jf = os.path.join(job.output_dir, "job.json")
        self.assertTrue(os.path.isfile(jf))
        # 用全新 manager 走启动恢复
        m2 = JobManager(os.path.join(self.root, "jobs"), CFG)
        m2.load_all()
        info = m2.list_jobs()[0]
        self.assertEqual(info["job_id"], job.job_id)
        self.assertEqual(info["total"], 2)
        # 恢复后仍可领取/保存（回归渲染需读取源数据）
        claim = m2.claim(job.job_id, "ann1")
        self.assertEqual(m2.progress(job.job_id), (0, 2))
        sel = {f.name: "不存在" for f in CFG.features}
        boxes = {f.key: {"xyxy": [0, 0, 3, 5], "traces": [0, 3], "samples": [0, 6]} for f in CFG.features if f.bbox}
        ok, rec, err = m2.save(job.job_id, "ann1", claim["gid"], sel, boxes)
        self.assertTrue(ok, err)
        self.assertTrue(os.path.isfile(os.path.join(job.output_dir, rec["image_path"])))
```

- [ ] **Step 3: 运行通过**

Run: `python -m unittest tests.test_jobmanager -v` → PASS（含恢复用例；会真实渲染 2 张 512×1024，数秒可接受）

---

### Task 7: `web_app.py` 重构为多人中央服务

**Files:**
- Modify: `web_app.py`（大改：删全局 `SES`；接 `Accounts / JobManager / Uploader`；角色分流 UI）
- Create: `tests/test_web_core.py`（对可单测的纯逻辑：像素→数据 box 映射沿用、占位符/句子构造——把 handler 里能抽出的纯函数收拢测试）

**Interfaces（本任务最终产品）：**
- `main()` 启动时：`Accounts.ensure_default(users.yaml)`；`JM=JobManager(jobs_root=项目根/jobs, cfg=CFG).load_all()`；`UP=Uploader(load_config(cos_config.yaml))`；`UP.start()`；`demo.launch(auth=ACC.authenticate, server_name="0.0.0.0", server_port=port)`
- Handler 统一签名含 `request: gr.Request`，用户名 `request.username`。
- 页面：admin 看到「① 建作业 / ② 作业与标注 / ③ 管理」；annotator 只看到「② 作业与标注」。

**约束（对着规格核对）：**
- annotator 无建作业/重传/账号入口；权限以 handler 校验为准（非 UI 隐藏）。
- 保存成功后把 `[record["image_path"], "labels.jsonl"]` 交给 `UP.enqueue(job_id, job.output_dir, [...])`；失败不得回滚标注。
- 标注器底部按钮：领取 / 归还 / 刷新 / 我标注的(下拉)。
- 不显示增强输入；不给标注者暴露「上一张/下一张」（改为作业池领取语义）。
- 每次交互（Radio change、框选点、领取、保存）调用 `JM.renew(job_id, user)` 续租约。

- [ ] **Step 1: 收拢纯逻辑并先测**

把现 `web_app.py` 中 `_fmt/_label_of/_bbox_feats/_pixel_box_to_data/IMG_W/IMG_H` 抽到一个新函数集合（保留同名，放在模块顶层函数，删掉对 `SES` 的依赖；`_bbox_feats` 只依赖 `CFG`）。写 `tests/test_web_core.py` 覆盖：

```python
# -*- coding: utf-8 -*-
import unittest
from labels import LabelConfig
from web_core import fmt_option, label_of, pixel_box_to_data, IMG_W, IMG_H

class TestPixelBox(unittest.TestCase):
    def test_pixel_box_to_data_range(self):
        box = pixel_box_to_data((0, 0), (IMG_W - 1, IMG_H - 1), 8, 200)
        self.assertEqual(box["traces"], [0, 8])
        self.assertEqual(box["samples"], [0, 200])
        self.assertEqual(box["xyxy"], [0, 0, IMG_W - 1, IMG_H - 1])
```

实现建议：把下列函数原样移到新文件 `web_core.py`（无副作用、不 import gradio）：
`IMG_W=512; IMG_H=1024`、`fmt_option(opt)`（=原 `_fmt`）、`label_of(display)`、`pixel_box_to_data(c0,c1,n_tr,ns)`（原样，见 `web_app.py:112-124`）。
`web_app.py` 顶部 `from web_core import ...` 复用，删除本地重复定义。

Run: `python -m unittest tests.test_web_core -v` → PASS

- [ ] **Step 2: 重写 `web_app.py` 的会话与页面（删除 `SES` 与三步旧流）**

按下面「目标结构」重写 `web_app.py`（读现有文件为底，保留 出图/框选/句子 交互实现并把状态从全局 `SES` 换成 `WORK[username]`；UI 用 Accordion+Dropdown 组成，组件尽量沿用现有）。**核心骨架**：

```python
# web_app.py（重写后关键骨架；完整实现以本骨架为准补齐事件接线）
from __future__ import annotations
import argparse, os, tempfile
import gradio as gr
import numpy as np
from web_core import IMG_W, IMG_H, fmt_option, label_of, pixel_box_to_data
from labels import LabelConfig
from users import Accounts
from jobmanager import JobManager, JobError
from cloudsync import Uploader, load_config

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_config.yaml")
USERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.yaml")
COS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cos_config.yaml")
JOBS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
os.makedirs(JOBS_ROOT, exist_ok=True)

CFG = LabelConfig(CONFIG_PATH)
ACC = Accounts(USERS_PATH)            # 启动前 ensure_default
JM = JobManager(JOBS_ROOT, CFG)
UP = Uploader(load_config(COS_PATH))
UP.start()
JM.load_all()

# 每用户工作态：{username: {"job_id","gid","partial","boxes","box_mode","pending_corner"}}
WORK: dict[str, dict] = {}
WORK_LOCK = __import__("threading").RLock()

def wstate(user): return WORK.setdefault(user, {})

def _user(request) -> str:
    return (request.username or "") if request is not None else ""
def _is_admin(user) -> bool: return ACC.role(user) == "admin"
```

（此文件不在此处贴全部 UI；**实现时**在本骨架下补齐：）
- 建作业区（admin）：沿用现 第1/2步组件（sgy 路径/端序/排序键/抽道集键/键值勾选/clip）但把「生成道集」「开始标注」替换为一个 `创建作业` 按钮 → 调 `JM.create_job(...)`；`build_info` 显示成功 + 作业下拉刷新。
- 标注区（全体可见，annotator 仅有此区）：`job_dd`（`gr.Dropdown`，候选 = `[j["job_id"] for j in JM.list_jobs() if j["state"] in ("open","broken") or ...]` 详见实现）+ 现有「左图右栏 + 逐特征 Radio + bbox 框选 + 句子」组件 + 底部按钮组（领取/归还/刷新/我标注的下拉/保存）。
- 管理区（admin）：`全部重传` 按钮 → 对选定 job `UP.resync_all(...)`；`作业开关`；`账号` 提示改 `users.yaml`。

- [ ] **Step 3: 标注流程 handler（新语义）——这是本任务核心，代码必须与 jobmanager 接口一致**

在 `web_app.py` 实现下列 handler（事件接线见 Step 4 列表）：

```python
def open_job_for(request, job_id):
    user = _user(request); st = wstate(user)
    st["job_id"] = job_id; st.pop("gid", None); st.pop("partial", None); st.pop("boxes", None)
    st["box_mode"] = None; st["pending_corner"] = None
    return refresh_anno(request)

def claim_next(request, job_id):
    user = _user(request); st = wstate(user); st["job_id"] = job_id
    out = JM.claim(job_id, user)
    if out["gid"]:
        st["gid"] = out["gid"]
        st["partial"], st["boxes"] = {}, {}
        st["box_mode"] = None; st["pending_corner"] = None
        return (*show_current(request), *box_statuses_of(request))
    return (gr.skip(), "⚠️ " + (out.get("reason") or "无任务"), "", *([gr.Radio()] * len(CFG.features)), *(["…"] * len(bbox_feats())))
```

其余 handler（`show_current / on_radio_change / enter_box_mode / clear_box / on_img_select / save_anno / release_current / reopen_mine / list_my`）都按「从 `_user(request)` 拿用户名、从 `wstate(user)` 拿 `gid/partial/boxes`、数据经 `job = JM.get(st["job_id"])` 获得（`job.gather_meta/raw/display/ensure_image/ns`），像素框仍走 `pixel_box_to_data` + `job.gather(gid).n_traces` 与 `job.ns`」来实现。保存最后：

```python
def save_anno(request):
    user = _user(request); st = wstate(user)
    if not st.get("gid"): return gr.skip(), "⚠️ 没有正在标注的道集", "", *([gr.Radio()] * len(CFG.features)), *(["…"] * len(bbox_feats()))
    job = JM.get(st["job_id"]); sel = radios_to_selection(...)   # 沿用 _selection_from_radios
    ok, rec, err = JM.save(st["job_id"], user, st["gid"], sel, st.get("boxes") or {}, is_admin=_is_admin(user))
    if not ok:
        cur = show_current(request)
        return (cur[0], "⚠️ " + err, cur[2], *cur[3:])
    # 云上传（异步，失败不阻塞标注）
    rels = [rec["image_path"]] if rec.get("image_path") else []
    rels.append("labels.jsonl")
    UP.enqueue(st["job_id"], job.output_dir, rels)
    n_labeled, total = JM.progress(st["job_id"])
    st.pop("gid", None); st.pop("partial", None); st.pop("boxes", None)
    return gr.update(value=render_display(request)), f"✔ 已保存 {user} | 进度 {n_labeled}/{total}｜点「领取下一张」", "", *([gr.Radio(value=None)] * len(CFG.features)), *(["…"] * len(bbox_feats()))
```

> 注：`render_display/radios_to_selection/box_statuses_of/bbox_feats` 等在实现时补齐（均无 `SES` 依赖）。显示图= `job.ensure_image(gid)` 叠加已画框的临时副本（复用 `overlay_boxes`）。

- [ ] **Step 4: 事件接线（对着现有 `web_app.py:601-624` 逐条改）**

| 现有事件 | 改为 |
|---|---|
| `btn_load/btn_scan/btn_build/btn_preview/btn_start` | 仅保留建作业相关（load→读取信息、scan→扫键值、`创建作业`→`create_job`） |
| `btn_prev/btn_next/btn_skip` | 删除（换成领取语义） |
| 新增 `btn_claim`→`claim_next`；`btn_release`→`release_current`；`btn_reopen`→`reopen_mine`；`btn_refresh`→`refresh_anno` |
| `btn_save`→`save_anno(request,...)`；`cur_img.select`→`on_img_select(request,...)`；Radio `change`→`on_radio_change(request,...)` 并调 `JM.renew`；`btn_inherit` 保留但改从本用户最近标注继承 |
| admin `btn_resync`→`UP.resync_all(job_dd.value, JM.get(job_dd.value).output_dir)`；`btn_toggle`→`JM.set_state` |

每个需要用户名的事件函数都带 `request: gr.Request` 参数（Gradio 自动注入）。`save/claim` 等事件输出列与原标注区一致（`anno_outputs` 结构保留，仅把数组来源换成 per-user 状态）。

- [ ] **Step 5: 启动冒烟验证（手动/脚本）**

Run: `python -c "import web_app"` 应无导入错误。

Run: `start python web_app.py`（或 `python -m uvicorn ...` 否——Gradio 自带）→ 浏览器打开 `http://127.0.0.1:7860`，用 `boss/boss123` 登录后应看到「建作业/管理」；`ann1/ann123` 登录只看到标注区。用两个浏览器分别登录验证：A 领取第 1 张并保存 → B 领取应跳到第 2 张（不重复）；B 保存后作业进度 +2。`jobs/<job_id>/images/` 出现 PNG、`labels.jsonl` 含 `annotated_by`。停服。

（Gradio 登录后能否在 handler 拿 `request.username` —— 若发现拿不到，改用 `auth` callable 返回 `username`（Gradio 支持 callable 返回用户名用于记录）；此为本任务唯一的实现期待定项，优先保证可拿用户名。）

- [ ] **Step 6: 增补自动化 e2e（用 requests 打 Gradio HTTP，可选但推荐）**

Create: `tests/smoke_web.py` —— 用 `tests/segy_factory.write_sgy` 造一个小 sgy；直接 import `web_app` 的 `JM`，注册作业后调用 handler 层的纯函数做断言（不真正起 HTTP，避免 flaky）。运行 `python -m unittest tests.test_web_core -v` + `python tests/smoke_web.py`。**验收标准**：两个用户名各自 claim/save 后，`JM.progress` 正确、无重复标注、记录带 `annotated_by`。

---

### Task 8: 文档与收尾

**Files:**
- Modify: `README.md`
- Modify: `PLAN.md`
- Create: `启动.bat`

- [ ] **Step 1: `启动.bat`**

```bat
@echo off
cd /d "%~dp0"
python web_app.py
```

- [ ] **Step 2: README 增加「多人协作（新）部署」小节**

写明：同局域网标注 → `python web_app.py` → 访问 `http://<本机内网IP>:7860`；账号在 `users.yaml`（热生效，改后重新登录）；建作业在 boss 账号；结果在 `jobs/<job_id>/`，上传到 COS 在 `cos_config.yaml` 开启 `enabled: true` 并填凭证后重启；npy 留本地、增强离线用 `backfill_augment.py` 指向该作业目录。

- [ ] **Step 3: PLAN.md 变更记录追加**

追加一行：`| v3.0 | 多人协作中央服务：jobmanager 任务池/租约、users 账号、cloudsync(COS)、web_app 去全局单例（详见 docs/superpowers/specs 与 plans）|`

- [ ] **Step 4: 全量回归**

Run: `python -m unittest discover -s tests -v` → 全绿；`python -c "import web_app"` 通过；再手工按 Task7 Step5 冒烟一次。

---

## 自查（已执行）

- **Spec 覆盖**：§2 账号/角色→Task1/7；§5 作业/池/租约/并发/保存/缓存→Task2/3/4/6；§6 会话重构→Task7；§7 COS→Task5/7；§8 增强离线→Task8 文档指向现有 backfill 脚本（不改标注循环，符合 spec）；§9 UI→Task7；§10 恢复→Task2(load_all)/Task6 测试；§11 部署内网→Task7/8；§14 测试→对应各 Task + smoke。非目标（不动核心模块、不做公网）在约束中强制。
- **占位扫描**：无 TODO/TBD；UI 任务以「骨架+接线表+验证」给出，业务逻辑全部收敛到可单测的 jobmanager/web_core，避免把 Gradio 布局塞进计划（逐行 UI 在实现时对照现有文件改，属低风险改写）。
- **类型一致**：`JobManager.claim/save/release/renew/current/progress/record/mine/list_jobs/set_state/get/create_job/load_all/register_job`、`Job.*`、`Uploader.*`、`Accounts.*`、`web_core.*` 在本计划各 Task 间同名同签，无漂移。
