"""User-requested, offline-only legacy index maintenance."""
from __future__ import annotations

import json
import secrets
import time
from uuid import uuid4
from pathlib import Path

from .. import storage
from ..config import settings
from ..services import reindex
from ..storage.connection import connect
from ..storage.runtime_lock import acquire_runtime_lock, release_runtime_lock
from . import store

KIND = "legacy_reindex"
PENDING = "maintenance_pending"


def _document(user_id: int, doc_id: int, *, write: bool = False):
    with connect() as conn:
        enabled = conn.execute("SELECT 1 FROM rm_users WHERE id=? AND disabled=0", (user_id,)).fetchone()
    if not enabled:
        raise PermissionError("账号不可用")
    row = storage.get_document(doc_id)
    if row is None:
        raise ValueError("文献不存在")
    library = store.library_access(user_id, row.group_id, write=write)
    return row, library


def inspect_document(user_id: int, doc_id: int) -> dict:
    row, library = _document(user_id, doc_id)
    protected = row.status == "ready" and bool(row.index_version and row.file_sha256)
    result = {"doc_id": row.id, "status": row.status, "protected": protected,
              "can_request": False, "reason": "当前索引已有版本校验。" if protected else "",
              "expected_sha256": None}
    if protected:
        return result
    if row.index_version or row.file_sha256:
        result["reason"] = "索引身份记录不完整或不适用于历史重建，请管理员核对。"
        return result
    if row.status not in {"ready", *reindex.REINDEX_STATES}:
        result["reason"] = "当前处理尚未结束，暂不能申请索引维护。"
        return result
    try:
        # Independent persisted provenance, never the current index contents.
        parsed = json.loads(Path(row.parsed_json_path or "").read_text(encoding="utf-8"))
        expected = parsed.get("file_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("缺少可核对的历史文件指纹，请管理员检查入库记录。")
        reindex._verify_source(row, expected)
        result.update(expected_sha256=expected, can_request=library["role"] in {"owner", "editor"},
                      reason="原文和历史配置已核对，可申请离线维护。")
    except ValueError as exc:
        result["reason"] = str(exc) if not isinstance(exc, json.JSONDecodeError) else "历史解析记录无法读取，请管理员核对。"
    except (OSError, TypeError, AttributeError):
        result["reason"] = "历史原文或入库记录不可用，请管理员核对。"
    if library["role"] == "viewer":
        result["reason"] += " 需要文献库编辑权限才能申请。"
    return result


def request_rebuild(user_id: int, doc_id: int, expected_sha256: str, *, retry_id: str | None = None) -> dict:
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row, _ = _document(user_id, doc_id, write=True)
        if retry_id:
            old = store.task(retry_id, user_id)
            if old["kind"] != KIND or old["status"] not in {"failed", "interrupted", "cancelled"}:
                raise ValueError("该维护申请不需要重试")
            if old["payload"]["doc_id"] != doc_id or old["payload"]["expected_sha256"] != expected_sha256:
                raise ValueError("维护申请与当前文献不一致")
        for task in conn.execute("SELECT id,owner_id,payload FROM rm_tasks WHERE kind=? AND status IN (?, 'running')", (KIND, PENDING)):
            if json.loads(task["payload"]).get("doc_id") == doc_id:
                if task["owner_id"] == user_id:
                    return store.task(task["id"], user_id)
                raise ValueError("这篇文献已有待处理的维护申请")
        info = inspect_document(user_id, doc_id)
        committed_retry = bool(retry_id and info["protected"] and row.file_sha256 == expected_sha256)
        if not info["can_request"] and not committed_retry:
            raise ValueError(info["reason"])
        if not committed_retry and info["expected_sha256"] != expected_sha256:
            raise ValueError("原文指纹已变化，请刷新后重新核对")
        active = conn.execute("SELECT count(*) FROM rm_tasks WHERE owner_id=? AND status IN ('queued','running',?)", (user_id,PENDING)).fetchone()[0]
        if active >= 10:
            raise store.TaskQueueFull("待处理任务较多，请稍后再试")
        now, tid = time.time(), retry_id or secrets.token_hex(16)
        payload = {"doc_id":row.id,"group_id":row.group_id,"expected_sha256":expected_sha256}
        result = {"doc_id":row.id,"title":row.paper_title or row.filename,"offline_only":True}
        if retry_id:
            conn.execute("UPDATE rm_tasks SET status=?,stage='等待管理员安排离线维护',progress=0,error=NULL,cancel_requested=0,result=?,updated_at=? WHERE id=?",
                         (PENDING,json.dumps(result,ensure_ascii=False),now,tid))
        else:
            conn.execute("INSERT INTO rm_tasks(id,owner_id,kind,status,stage,payload,result,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                         (tid,user_id,KIND,PENDING,"等待管理员安排离线维护",json.dumps(payload),json.dumps(result,ensure_ascii=False),now,now))
        conn.execute("INSERT INTO rm_events(task_id,data,created_at) VALUES(?,?,?)",(tid,json.dumps({"status":PENDING}),now))
    return store.task(tid,user_id)


def cancel_request(user_id: int, task_id: str) -> None:
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        task = store.task(task_id,user_id)
        if task["kind"] != KIND:
            raise ValueError("不是索引维护申请")
        if task["status"] == PENDING:
            conn.execute("UPDATE rm_tasks SET status='cancelled',stage='维护申请已撤回',cancel_requested=1,updated_at=? WHERE id=?",
                         (time.time(),task_id))
            conn.execute("INSERT INTO rm_events(task_id,data,created_at) VALUES(?,?,?)",
                         (task_id,json.dumps({"status":"cancelled"}),time.time()))
        elif task["status"] not in store.TERMINAL:
            raise ValueError("维护已由管理员执行，请等待当前阶段结束")


def retry_request(user_id: int, task_id: str) -> dict:
    task = store.task(task_id,user_id)
    if task["kind"] != KIND:
        raise ValueError("不是索引维护申请")
    return request_rebuild(user_id,task["payload"]["doc_id"],task["payload"]["expected_sha256"],retry_id=task_id)


def execute_request(task_id: str, *, expected_sha256: str) -> dict:
    """Local operator confirms SHA, stops all writers and invokes exactly one request."""
    acquire_runtime_lock(settings.database_path, "maintenance-" + uuid4().hex)
    try:
        task = store.task(task_id)
        if task["kind"] != KIND or task["payload"].get("expected_sha256") != expected_sha256:
            raise ValueError("任务类型或管理员核对的原文指纹不一致")
        if task["status"] == "completed":
            return task
        if task["status"] != PENDING:
            raise ValueError("申请未在等待维护，请在工作台显式重试")
        try:
            row, _ = _document(task["owner_id"], task["payload"]["doc_id"], write=True)
            if row.group_id != task["payload"]["group_id"]:
                raise PermissionError("文献所属库已变化")
            reindex._verify_source(row, expected_sha256)
            store.update_task(task_id,status="running",stage="管理员正在重建索引",progress=0.1)
            rebuilt = reindex._rebuild_legacy_index_locked(row.id, expected_sha256=expected_sha256)
            store.update_task(task_id,status="completed",stage="索引重建完成，原文与历史引用保留",progress=1,
                              result={**task["result"],"index_version":rebuilt.index_version,"file_sha256":rebuilt.file_sha256})
        except BaseException:
            # Avoid exposing filesystem paths or provider responses in user-facing errors.
            store.update_task(task_id,status="failed",stage="维护未完成，可重新申请",
                              error="请管理员核对权限、原文、配置和本次维护日志后重试。原文与历史引用保留。")
            raise
        return store.task(task_id)
    finally:
        release_runtime_lock(settings.database_path)
