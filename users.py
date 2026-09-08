# -*- coding: utf-8 -*-
"""users.py — 账号与角色（users.yaml，热重载：每次校验时按 (mtime, size) 判断是否重新读盘；读取/解析失败时沿用最近一次成功加载的数据，不向调用方抛错）"""
from __future__ import annotations
import logging
import os, threading
import yaml

logger = logging.getLogger(__name__)

class AccountsError(Exception):
    pass

class Accounts:
    def __init__(self, path: str):
        self.path = path
        self._users: dict[str, dict] = {}
        self._mtime: float | None = None
        self._size: int | None = None
        self._lock = threading.Lock()
        if not os.path.isfile(path):
            raise AccountsError(f"账号文件不存在: {path}")
        self._reload(force=True)

    def _reload(self, force: bool = False):
        if not os.path.isfile(self.path):
            logger.warning("账号文件不存在，沿用最近一次成功加载的数据: %s", self.path)
            return
        try:
            mtime = os.path.getmtime(self.path)
            size = os.path.getsize(self.path)
            if not force and mtime == self._mtime and size == self._size:
                return
            with open(self.path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            users = {}
            for u in data.get("users") or []:
                name = str(u["username"])
                users[name] = {"password": str(u["password"]),
                               "role": str(u.get("role", "annotator"))}
        except Exception:
            logger.warning("读取账号文件失败，沿用最近一次成功加载的数据: %s", self.path, exc_info=True)
            return
        self._users, self._mtime, self._size = users, mtime, size

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
