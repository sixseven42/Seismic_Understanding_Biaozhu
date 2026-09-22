# -*- coding: utf-8 -*-
"""cloudsync.py — 腾讯云 COS 上传队列（后台线程 + 有界重试）

启用条件：cos_config.yaml enabled=true、secret_id/secret_key/region/bucket 齐全，
且已 pip install cos-python-sdk-v5。未满足时保持安全降级：enqueue 只记入
self.dry_log，不影响标注保存（失败绝不阻塞标注）。
每次上传有界重试 max_attempts 次（指数退避，上限 8s）；重试耗尽后把最终失败
追加到 <job output>/upload.log（供管理员用 resync_all 重传），并记入 self.failures。
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
        self.max_attempts = int(cfg.get("max_attempts", 4))

    def start(self):
        if not self.enabled or self._started:
            return
        try:
            from qcloud_cos import CosConfig, CosS3Client
        except Exception as e:
            log.warning("cos-python-sdk-v5 未安装，云上传关闭：%s", e)
            self.enabled = False
            return
        missing = [k for k in ("region", "secret_key", "bucket") if not self.cfg.get(k)]
        if missing:
            log.warning("COS 配置不完整（缺少 %s），云上传关闭", ", ".join(missing))
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
        key, local, output_dir = item
        delay = 1.0
        for attempt in range(1, self.max_attempts + 1):
            try:
                with open(local, "rb") as body:
                    self._client.put_object(Bucket=self.cfg["bucket"], Key=key, Body=body)
                return
            except Exception as e:
                if attempt < self.max_attempts:
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)
                    continue
                self.failures.append((key, local, str(e)))
                log.error("上传失败 %s <- %s: %s", key, local, e)
                self._write_log(output_dir, key, e)

    def _write_log(self, output_dir, key, err):
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "upload.log"), "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{key}\t{err}\n")
        except Exception:
            pass  # 记录失败绝不阻断

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
            self._q.put((key, local, output_dir))

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
