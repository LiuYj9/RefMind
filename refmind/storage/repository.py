"""数据访问层（CRUD）：用户组、文档、会话、消息。"""

from __future__ import annotations

from typing import Optional

from .connection import connect
from .models import DocumentRow, Group, Message, Session


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
