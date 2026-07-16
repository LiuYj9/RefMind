"""SQLite 连接管理与数据库初始化。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from ..config import settings
from .models import SCHEMA


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """提供一个自动提交并关闭的数据库连接（按调用创建，适配 Streamlit）。"""
    settings.ensure_dirs()
    # 每个 worker 使用独立连接；busy timeout 让并行入库时的短写事务排队，
    # 而不是立即以 "database is locked" 失败。
    conn = sqlite3.connect(settings.database_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # 开启外键级联，删除组时自动清理关联数据
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """若表不存在则创建。"""
    with connect() as conn:
        # WAL 允许读请求与短写事务并行；不支持 WAL 的特殊文件系统继续使用默认模式。
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.DatabaseError:
            pass
        conn.executescript(SCHEMA)
