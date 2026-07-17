"""SQLite 连接管理与数据库初始化。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from ..config import settings
from .models import SCHEMA


def _migrate_documents_schema(conn: sqlite3.Connection) -> None:
    """为旧数据库补齐论文题名与稳定库内序号，不要求用户重建文献库。"""
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(documents)").fetchall()
    }
    if "paper_title" not in columns:
        conn.execute("ALTER TABLE documents ADD COLUMN paper_title TEXT")
    if "library_index" not in columns:
        conn.execute(
            "ALTER TABLE documents ADD COLUMN library_index INTEGER NOT NULL DEFAULT 0"
        )

    # 旧数据全部为 0。按最早入库顺序分配稳定序号；后续删除不重排，避免历史引用漂移。
    group_rows = conn.execute(
        "SELECT DISTINCT group_id FROM documents ORDER BY group_id"
    ).fetchall()
    for group_row in group_rows:
        group_id = int(group_row["group_id"])
        rows = conn.execute(
            "SELECT id, library_index FROM documents "
            "WHERE group_id = ? ORDER BY id ASC",
            (group_id,),
        ).fetchall()
        next_index = max(
            (int(row["library_index"] or 0) for row in rows),
            default=0,
        ) + 1
        for row in rows:
            if int(row["library_index"] or 0) > 0:
                continue
            conn.execute(
                "UPDATE documents SET library_index = ? WHERE id = ?",
                (next_index, int(row["id"])),
            )
            next_index += 1

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_group_library_index "
        "ON documents(group_id, library_index)"
    )


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
        _migrate_documents_schema(conn)
