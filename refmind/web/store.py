"""Additive team/workflow tables. Legacy records are never implicitly shared."""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError

from .. import storage
from ..storage.connection import connect

PASSWORDS = PasswordHasher()
DUMMY_HASH = PASSWORDS.hash(secrets.token_urlsafe(24))
TERMINAL = {"completed", "partial", "failed", "cancelled", "needs_review", "interrupted"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS rm_users (
 id INTEGER PRIMARY KEY, username TEXT UNIQUE COLLATE NOCASE NOT NULL,
 password_hash TEXT NOT NULL, is_admin INTEGER NOT NULL DEFAULT 0,
 disabled INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS rm_tokens (
 token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES rm_users(id),
 csrf TEXT NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS rm_libraries (
 group_id INTEGER PRIMARY KEY REFERENCES groups(id) ON DELETE CASCADE,
 owner_id INTEGER NOT NULL REFERENCES rm_users(id), title TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('private','shared','session','temporary')));
CREATE TABLE IF NOT EXISTS rm_members (
 group_id INTEGER NOT NULL REFERENCES rm_libraries(group_id) ON DELETE CASCADE,
 user_id INTEGER NOT NULL REFERENCES rm_users(id),
 role TEXT NOT NULL CHECK(role IN ('viewer','editor')), PRIMARY KEY(group_id,user_id));
CREATE TABLE IF NOT EXISTS rm_sessions (
 session_id INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
 owner_id INTEGER NOT NULL REFERENCES rm_users(id));
CREATE TABLE IF NOT EXISTS rm_tasks (
 id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES rm_users(id), kind TEXT NOT NULL,
 status TEXT NOT NULL, stage TEXT NOT NULL DEFAULT '', progress REAL NOT NULL DEFAULT 0,
 payload TEXT NOT NULL, result TEXT NOT NULL DEFAULT '{}', error TEXT,
 cancel_requested INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS rm_tasks_owner ON rm_tasks(owner_id,created_at);
CREATE TABLE IF NOT EXISTS rm_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL REFERENCES rm_tasks(id) ON DELETE CASCADE,
 data TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS rm_assets (
 id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES rm_users(id),
 session_id INTEGER NOT NULL REFERENCES rm_sessions(session_id) ON DELETE CASCADE,
 group_id INTEGER REFERENCES groups(id), doc_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
 paper TEXT NOT NULL, file_info TEXT NOT NULL, status TEXT NOT NULL,
 expires_at REAL NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS rm_document_sources (
 doc_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
 group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
 sha256 TEXT NOT NULL, metadata TEXT NOT NULL,
 UNIQUE(group_id,sha256));
CREATE TABLE IF NOT EXISTS rm_answer_evidence (
 message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
 snapshot TEXT NOT NULL);
"""


def initialize() -> None:
    storage.init_db()
    with connect() as conn:
        conn.executescript(SCHEMA)
        conn.execute("UPDATE rm_tasks SET status='interrupted',stage='服务重启，可重试',updated_at=? WHERE status IN ('queued','running')", (time.time(),))
        conn.execute("DELETE FROM rm_tokens WHERE expires_at < ?", (time.time(),))


def create_user(username: str, password: str, *, admin: bool = False) -> dict:
    username = username.strip()
    if not re.fullmatch(r"[\w.@-]{2,64}", username) or not 12 <= len(password) <= 256:
        raise ValueError("用户名需为 2–64 个字母、数字或 .@_-；密码需为 12–256 个字符")
    hashed = PASSWORDS.hash(password)
    with connect() as conn:
        cur = conn.execute("INSERT INTO rm_users(username,password_hash,is_admin,created_at) VALUES(?,?,?,?)", (username, hashed, int(admin), time.time()))
        return {"id": cur.lastrowid, "username": username, "is_admin": admin}


def login(username: str, password: str) -> tuple[str, str, dict] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM rm_users WHERE username=?", (username,)).fetchone()
    try:
        PASSWORDS.verify(row["password_hash"] if row else DUMMY_HASH, password)
    except VerificationError:
        return None
    if row is None or row["disabled"]:
        return None
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
    with connect() as conn:
        if PASSWORDS.check_needs_rehash(row["password_hash"]):
            conn.execute("UPDATE rm_users SET password_hash=? WHERE id=?", (PASSWORDS.hash(password), row["id"]))
        conn.execute("INSERT INTO rm_tokens VALUES(?,?,?,?)", (hashlib.sha256(token.encode()).hexdigest(), row["id"], csrf, time.time() + 12 * 3600))
    return token, csrf, {"id": row["id"], "username": row["username"], "is_admin": bool(row["is_admin"])}


def authenticate(token: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT u.id,u.username,u.is_admin,t.csrf FROM rm_tokens t JOIN rm_users u ON u.id=t.user_id WHERE token_hash=? AND expires_at>? AND u.disabled=0", (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
    return dict(row) if row else None


def logout(token: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM rm_tokens WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))


def create_library(user_id: int, title: str, kind: str = "private") -> dict:
    if kind not in {"private", "shared", "session", "temporary"}:
        raise ValueError("无效文献库类型")
    title = title.strip()[:120]
    if not title:
        raise ValueError("文献库名称不能为空")
    with connect() as conn:
        group_id = conn.execute("INSERT INTO groups(name) VALUES(?)", (f"web:{secrets.token_hex(12)}",)).lastrowid
        conn.execute("INSERT INTO rm_libraries VALUES(?,?,?,?)", (group_id, user_id, title, kind))
    return {"id": group_id, "title": title, "kind": kind, "role": "owner"}


def library_access(user_id: int, group_id: int, *, write: bool = False, allow_temporary: bool = False) -> dict:
    with connect() as conn:
        row = conn.execute("SELECT l.*,m.role FROM rm_libraries l LEFT JOIN rm_members m ON m.group_id=l.group_id AND m.user_id=? WHERE l.group_id=?", (user_id, group_id)).fetchone()
    if not row or (row["owner_id"] != user_id and row["role"] is None):
        raise PermissionError("无权访问该文献库")
    if row["kind"] in {"session", "temporary"} and not allow_temporary:
        raise PermissionError("会话资料不可作为正式文献库访问")
    role = "owner" if row["owner_id"] == user_id else row["role"]
    if write and role not in {"owner", "editor"}:
        raise PermissionError("需要文献库编辑权限")
    return {"id": group_id, "title": row["title"], "kind": row["kind"], "role": role}


def libraries(user_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT l.*,m.role FROM rm_libraries l LEFT JOIN rm_members m ON m.group_id=l.group_id AND m.user_id=? WHERE (l.owner_id=? OR m.user_id=?) AND l.kind IN ('private','shared') ORDER BY l.group_id DESC", (user_id, user_id, user_id)).fetchall()
    return [{"id": r["group_id"], "title": r["title"], "kind": r["kind"], "role": "owner" if r["owner_id"] == user_id else r["role"]} for r in rows]


def grant_member(owner_id: int, group_id: int, user_id: int, role: str) -> None:
    access = library_access(owner_id, group_id, write=True)
    if access["role"] != "owner" or access["kind"] != "shared" or role not in {"viewer", "editor"}:
        raise PermissionError("只有共享库所有者可以管理成员")
    with connect() as conn:
        conn.execute("INSERT INTO rm_members VALUES(?,?,?) ON CONFLICT(group_id,user_id) DO UPDATE SET role=excluded.role", (group_id, user_id, role))


def create_session(user_id: int, title: str, group_id: int | None = None) -> dict:
    if group_id is None:
        group_id = create_library(user_id, "会话资料", "session")["id"]
    else:
        library_access(user_id, group_id)
    with connect() as conn:
        sid = conn.execute("INSERT INTO sessions(group_id,name) VALUES(?,?)", (group_id, title[:120] or "新研读")).lastrowid
        conn.execute("INSERT INTO rm_sessions VALUES(?,?)", (sid, user_id))
    return {"id": sid, "group_id": group_id, "name": title or "新研读"}


def session_access(user_id: int, session_id: int) -> dict:
    with connect() as conn:
        row = conn.execute("SELECT s.* FROM sessions s JOIN rm_sessions a ON a.session_id=s.id WHERE s.id=? AND a.owner_id=?", (session_id, user_id)).fetchone()
    if not row:
        raise PermissionError("无权访问该会话")
    return dict(row)


def sessions(user_id: int) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT s.* FROM sessions s JOIN rm_sessions a ON a.session_id=s.id WHERE a.owner_id=? ORDER BY s.id DESC", (user_id,))]



class TaskQueueFull(ValueError):
    pass


def _queue_capacity(conn, user_id: int) -> None:
    active = conn.execute("SELECT count(*) FROM rm_tasks WHERE owner_id=? AND status IN ('queued','running','maintenance_pending')", (user_id,)).fetchone()[0]
    if active >= 10:
        raise TaskQueueFull("待处理任务较多，请稍后再试")

def _answer_slot(conn, user_id: int, payload: dict, *, exclude: str | None = None) -> None:
    session_id = payload.get("session_id")
    if type(session_id) is not int or session_id <= 0:
        raise ValueError("无效的会话编号")
    owned = conn.execute("SELECT session_id FROM rm_sessions WHERE session_id=? AND owner_id=?", (session_id, user_id)).fetchone()
    if not owned:
        raise PermissionError("无权访问该会话")
    # Manual JSON decoding also supports SQLite builds without JSON1.
    rows = conn.execute("SELECT id,payload FROM rm_tasks WHERE owner_id=? AND kind='answer' AND status IN ('queued','running')", (user_id,)).fetchall()
    for row in rows:
        if row["id"] != exclude and json.loads(row["payload"]).get("session_id") == session_id:
            raise ValueError("当前会话仍有问答处理中，请等待该轮完成后再提问或重试")


def create_task(user_id: int, kind: str, payload: dict) -> dict:
    task_id, now = secrets.token_hex(16), time.time()
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _queue_capacity(conn, user_id)
        if kind == "answer":
            _answer_slot(conn, user_id, payload)
        conn.execute("INSERT INTO rm_tasks(id,owner_id,kind,status,payload,created_at,updated_at) VALUES(?,?,?,'queued',?,?,?)", (task_id, user_id, kind, json.dumps(payload, ensure_ascii=False), now, now))
    return task(task_id, user_id)


def requeue_task(task_id: str, user_id: int) -> tuple[dict, bool]:
    """Compare-and-set prevents concurrent retries resetting a running/completed job."""
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM rm_tasks WHERE id=? AND owner_id=?", (task_id, user_id)).fetchone()
        if not row:
            raise PermissionError("无权访问该任务")
        changed = False
        if row["status"] in {"failed", "interrupted", "cancelled"}:
            _queue_capacity(conn, user_id)
            payload = json.loads(row["payload"])
            if row["kind"] == "answer":
                _answer_slot(conn, user_id, payload, exclude=task_id)
            changed = bool(conn.execute("UPDATE rm_tasks SET status='queued',cancel_requested=0,error=NULL,updated_at=? WHERE id=? AND owner_id=? AND status IN ('failed','interrupted','cancelled')", (time.time(), task_id, user_id)).rowcount)
    return task(task_id, user_id), changed


def task(task_id: str, user_id: int | None = None) -> dict:
    with connect() as conn:
        row = conn.execute("SELECT * FROM rm_tasks WHERE id=?", (task_id,)).fetchone()
    if row is None or (user_id is not None and row["owner_id"] != user_id):
        raise PermissionError("无权访问该任务")
    out = dict(row)
    for key in ("payload", "result"):
        out[key] = json.loads(out[key])
    return out


def update_task(task_id: str, *, status: str | None = None, stage: str | None = None, progress: float | None = None, result: dict | None = None, error: str | None = None) -> None:
    changes: dict[str, Any] = {"updated_at": time.time()}
    for key, value in (("status", status), ("stage", stage), ("progress", progress), ("result", result), ("error", error)):
        if value is not None:
            changes[key] = json.dumps(value, ensure_ascii=False) if key == "result" else value
    with connect() as conn:
        conn.execute("UPDATE rm_tasks SET " + ",".join(f"{key}=?" for key in changes) + " WHERE id=?", (*changes.values(), task_id))
        conn.execute("INSERT INTO rm_events(task_id,data,created_at) VALUES(?,?,?)", (task_id, json.dumps({k: v for k, v in changes.items() if k != "result"}, ensure_ascii=False), time.time()))


def asset(asset_id: str, user_id: int, *, require_ready: bool = False) -> dict:
    with connect() as conn:
        row = conn.execute("SELECT * FROM rm_assets WHERE id=? AND owner_id=?", (asset_id, user_id)).fetchone()
    if not row:
        raise PermissionError("无权访问该会话资料")
    item = dict(row)
    item["paper"], item["file_info"] = json.loads(item["paper"]), json.loads(item["file_info"])
    if item["expires_at"] <= time.time():
        item["status"] = "expired"
    if require_ready and item["status"] != "ready":
        raise ValueError("全文尚未就绪或已过期")
    return item


def public_task(item: dict) -> dict:
    return {k: v for k, v in item.items() if k not in {"owner_id", "payload", "cancel_requested"}}
