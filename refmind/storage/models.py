"""数据行模型（dataclass）与建表 SQL。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# 数据库建表语句
SCHEMA = """
CREATE TABLE IF NOT EXISTS groups (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id          INTEGER NOT NULL,
    filename          TEXT NOT NULL,
    original_path     TEXT,
    parsed_json_path  TEXT,
    summary           TEXT,
    num_chunks        INTEGER DEFAULT 0,
    status            TEXT DEFAULT 'pending',
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id    INTEGER NOT NULL,
    name        TEXT DEFAULT '新对话',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
"""


@dataclass
class Group:
    """用户组。"""

    id: int
    name: str
    created_at: str


@dataclass
class DocumentRow:
    """文档元信息。"""

    id: int
    group_id: int
    filename: str
    original_path: Optional[str]
    parsed_json_path: Optional[str]
    summary: Optional[str]
    num_chunks: int
    status: str
    created_at: str


@dataclass
class Session:
    """对话会话。"""

    id: int
    group_id: int
    name: str
    created_at: str


@dataclass
class Message:
    """单条对话消息。"""

    id: int
    session_id: int
    role: str
    content: str
    created_at: str
