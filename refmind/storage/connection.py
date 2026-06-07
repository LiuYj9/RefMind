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
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    # 开启外键级联，删除组时自动清理关联数据
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """若表不存在则创建。"""
    with connect() as conn:
        conn.executescript(SCHEMA)
