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
    paper_title       TEXT,
    library_index     INTEGER NOT NULL DEFAULT 0,
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

CREATE TABLE IF NOT EXISTS long_term_memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    group_id        INTEGER NOT NULL,
    session_id      INTEGER,
    content         TEXT NOT NULL,
    memory_type     TEXT NOT NULL CHECK(memory_type IN ('semantic', 'episodic')),
    subtype         TEXT DEFAULT 'other',
    memory_key      TEXT,
    content_hash    TEXT NOT NULL,
    importance      REAL NOT NULL DEFAULT 0.5,
    confidence      REAL NOT NULL DEFAULT 0.5,
    embedding       TEXT,
    source          TEXT NOT NULL DEFAULT 'conversation',
    is_active       INTEGER NOT NULL DEFAULT 1,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMP,
    expires_at      TIMESTAMP,
    superseded_by   INTEGER,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE,
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE SET NULL,
    FOREIGN KEY(superseded_by) REFERENCES long_term_memories(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_long_term_memory_scope
ON long_term_memories(user_id, group_id, is_active);

CREATE INDEX IF NOT EXISTS idx_long_term_memory_key
ON long_term_memories(user_id, group_id, memory_key, is_active);

CREATE INDEX IF NOT EXISTS idx_long_term_memory_hash
ON long_term_memories(user_id, group_id, content_hash, is_active);
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
    paper_title: Optional[str]
    library_index: int
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


@dataclass
class LongTermMemory:
    """跨会话的用户长期记忆；论文事实不进入此表。"""

    id: int
    user_id: str
    group_id: int
    session_id: Optional[int]
    content: str
    memory_type: str
    subtype: str
    memory_key: Optional[str]
    content_hash: str
    importance: float
    confidence: float
    embedding: Optional[str]
    source: str
    is_active: int
    access_count: int
    last_accessed_at: Optional[str]
    expires_at: Optional[str]
    superseded_by: Optional[int]
    created_at: str
    updated_at: str
