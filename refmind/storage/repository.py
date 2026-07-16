"""数据访问层（CRUD）：用户组、文档、会话、消息。"""

from __future__ import annotations

from typing import Optional

from .connection import connect
from .models import DocumentRow, Group, LongTermMemory, Message, Session


# 用户组
def create_group(name: str) -> Group:
    """创建用户组。"""
    with connect() as conn:
        cur = conn.execute("INSERT INTO groups (name) VALUES (?)", (name,))
        gid = cur.lastrowid
    return get_group(gid)


def get_group(group_id: int) -> Optional[Group]:
    """按 ID 获取用户组。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM groups WHERE id = ?", (group_id,)
        ).fetchone()
    return Group(**row) if row else None


def list_groups() -> list[Group]:
    """列出所有用户组（按创建时间倒序）。"""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM groups ORDER BY created_at DESC"
        ).fetchall()
    return [Group(**r) for r in rows]


def delete_group(group_id: int) -> None:
    """删除用户组（级联删除其文档、会话与消息）。"""
    with connect() as conn:
        conn.execute("DELETE FROM groups WHERE id = ?", (group_id,))


# 文档
def create_document(
    group_id: int,
    filename: str,
    original_path: str | None = None,
    status: str = "pending",
) -> int:
    """新增文档记录，返回文档 ID。"""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO documents (group_id, filename, original_path, status) "
            "VALUES (?, ?, ?, ?)",
            (group_id, filename, original_path, status),
        )
        return cur.lastrowid


def update_document(
    doc_id: int,
    *,
    original_path: str | None = None,
    parsed_json_path: str | None = None,
    summary: str | None = None,
    num_chunks: int | None = None,
    status: str | None = None,
) -> None:
    """更新文档字段（仅更新非 None 的字段）。"""
    fields, values = [], []
    if original_path is not None:
        fields.append("original_path = ?")
        values.append(original_path)
    if parsed_json_path is not None:
        fields.append("parsed_json_path = ?")
        values.append(parsed_json_path)
    if summary is not None:
        fields.append("summary = ?")
        values.append(summary)
    if num_chunks is not None:
        fields.append("num_chunks = ?")
        values.append(num_chunks)
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if not fields:
        return
    values.append(doc_id)
    with connect() as conn:
        conn.execute(
            f"UPDATE documents SET {', '.join(fields)} WHERE id = ?", values
        )


def get_document(doc_id: int) -> Optional[DocumentRow]:
    """按 ID 获取文档。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
    return DocumentRow(**row) if row else None


def list_documents(group_id: int) -> list[DocumentRow]:
    """列出某组的所有文档。"""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE group_id = ? ORDER BY created_at DESC",
            (group_id,),
        ).fetchall()
    return [DocumentRow(**r) for r in rows]


def delete_document(doc_id: int) -> None:
    """删除文档记录。"""
    with connect() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


# 会话
def create_session(group_id: int, name: str = "新对话") -> Session:
    """创建对话会话。"""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (group_id, name) VALUES (?, ?)", (group_id, name)
        )
        sid = cur.lastrowid
    return get_session(sid)


def get_session(session_id: int) -> Optional[Session]:
    """按 ID 获取会话。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return Session(**row) if row else None


def list_sessions(group_id: int) -> list[Session]:
    """列出某组的所有会话。"""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE group_id = ? ORDER BY created_at DESC",
            (group_id,),
        ).fetchall()
    return [Session(**r) for r in rows]


def rename_session(session_id: int, name: str) -> None:
    """重命名会话。"""
    with connect() as conn:
        conn.execute(
            "UPDATE sessions SET name = ? WHERE id = ?", (name, session_id)
        )


def delete_session(session_id: int) -> None:
    """删除会话（级联删除其消息）。"""
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


# 消息
def add_message(session_id: int, role: str, content: str) -> int:
    """新增一条消息，返回消息 ID。"""
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        return cur.lastrowid


def list_messages(session_id: int, limit: int | None = None) -> list[Message]:
    """列出某会话的消息（按时间正序）。

    指定 ``limit`` 时仅返回最近 ``limit`` 条，但仍保持正序。
    """
    query = "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC"
    params: tuple = (session_id,)
    if limit is not None:
        # 先取最近 limit 条，再按正序排列
        query = (
            "SELECT * FROM (SELECT * FROM messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?) ORDER BY id ASC"
        )
        params = (session_id, limit)
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [Message(**r) for r in rows]


def clear_messages(session_id: int) -> None:
    """清空某会话的全部消息。"""
    with connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


# 用户长期记忆
def create_long_term_memory(
    *,
    user_id: str,
    group_id: int,
    session_id: int | None,
    content: str,
    memory_type: str,
    subtype: str,
    memory_key: str | None,
    content_hash: str,
    importance: float,
    confidence: float,
    embedding: str | None,
    source: str = "conversation",
    expires_at: str | None = None,
) -> LongTermMemory:
    """新增一条已通过价值判断的长期记忆。"""
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO long_term_memories (
                user_id, group_id, session_id, content, memory_type, subtype,
                memory_key, content_hash, importance, confidence, embedding,
                source, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                group_id,
                session_id,
                content,
                memory_type,
                subtype,
                memory_key,
                content_hash,
                importance,
                confidence,
                embedding,
                source,
                expires_at,
            ),
        )
        memory_id = int(cur.lastrowid)
    memory = get_long_term_memory(memory_id)
    if memory is None:  # pragma: no cover - SQLite INSERT 后的防御边界
        raise RuntimeError("长期记忆写入后无法读取")
    return memory


def get_long_term_memory(memory_id: int) -> LongTermMemory | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM long_term_memories WHERE id = ?", (memory_id,)
        ).fetchone()
    return LongTermMemory(**row) if row else None


def list_long_term_memories(
    user_id: str,
    group_id: int,
    *,
    active_only: bool = True,
    limit: int | None = None,
) -> list[LongTermMemory]:
    """按用户与文献库边界列出长期记忆。"""
    clauses = ["user_id = ?", "group_id = ?"]
    params: list[object] = [user_id, group_id]
    if active_only:
        clauses.append("is_active = 1")
    query = (
        "SELECT * FROM long_term_memories WHERE "
        + " AND ".join(clauses)
        + " ORDER BY updated_at DESC, id DESC"
    )
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [LongTermMemory(**row) for row in rows]


def update_long_term_memory(memory_id: int, **changes: object) -> None:
    """更新允许变更的记忆字段，并刷新 ``updated_at``。"""
    allowed = {
        "session_id",
        "content",
        "memory_type",
        "subtype",
        "memory_key",
        "content_hash",
        "importance",
        "confidence",
        "embedding",
        "source",
        "is_active",
        "expires_at",
        "superseded_by",
    }
    items = [(key, value) for key, value in changes.items() if key in allowed]
    if not items:
        return
    assignments = [f"{key} = ?" for key, _ in items]
    values = [value for _, value in items]
    values.append(memory_id)
    with connect() as conn:
        conn.execute(
            f"UPDATE long_term_memories SET {', '.join(assignments)}, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            values,
        )


def touch_long_term_memories(memory_ids: list[int]) -> None:
    """记录记忆被实际召回，供衰减策略判断是否仍有价值。"""
    if not memory_ids:
        return
    placeholders = ",".join("?" for _ in memory_ids)
    with connect() as conn:
        conn.execute(
            f"""
            UPDATE long_term_memories
            SET access_count = access_count + 1,
                last_accessed_at = CURRENT_TIMESTAMP
            WHERE id IN ({placeholders}) AND is_active = 1
            """,
            memory_ids,
        )


def deactivate_long_term_memory(
    memory_id: int, *, superseded_by: int | None = None
) -> None:
    """软失效记忆，保留来源审计与冲突演进记录。"""
    update_long_term_memory(
        memory_id, is_active=0, superseded_by=superseded_by
    )
