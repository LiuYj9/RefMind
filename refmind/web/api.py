"""Same-origin API for the private RefMind research workspace."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from threading import RLock

from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import storage
from ..config import settings
from ..contracts import ResearchScope
from ..storage.connection import connect
from ..storage.runtime_lock import acquire_runtime_lock, release_runtime_lock
from . import store
from .jobs import JobRunner, _candidate, expire_assets, public_asset, workspace

COOKIE = "refmind_session"
_attempts: dict[str, deque] = defaultdict(deque)
_attempt_lock = RLock()


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class UserBody(LoginBody):
    is_admin: bool = False


class LibraryBody(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    shared: bool = False


class MemberBody(BaseModel):
    username: str
    role: str = "viewer"


class SessionBody(BaseModel):
    name: str = Field(default="新研读", max_length=120)
    group_id: int | None = None


class SearchBody(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    provider: str = "auto"
    year_from: int | None = Field(default=None, ge=1600, le=2100)
    year_to: int | None = Field(default=None, ge=1600, le=2100)
    open_access_only: bool = False
    publication_types: list[str] = Field(default_factory=list, max_length=10)
    limit: int = Field(default=20, ge=1, le=50)
    cursor: str | None = Field(default=None, max_length=10000)


class ReadingBody(BaseModel):
    search_id: str
    paper_id: str
    session_id: int
    location_index: int = Field(default=0, ge=0, le=100)


class ImportBody(BaseModel):
    group_id: int


class ConfirmBody(BaseModel):
    identity_confirmed: bool = False
    accept_partial: bool = False


class ReindexRequestBody(BaseModel):
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AnswerBody(ResearchScope):
    session_id: int
    question: str = Field(min_length=1, max_length=8000)


def current_user(request: Request) -> dict:
    user = store.authenticate(request.cookies.get(COOKIE, ""))
    if not user:
        raise HTTPException(401, "请先登录")
    if request.method not in {"GET", "HEAD", "OPTIONS"} and not secrets.compare_digest(request.headers.get("X-CSRF-Token", ""), user["csrf"]):
        raise HTTPException(403, "会话校验失效，请刷新后重试")
    return user


def admin_user(user: dict = Depends(current_user)) -> dict:
    if not user["is_admin"]:
        raise HTTPException(403, "需要管理员权限")
    return user


def _public_document(row) -> dict:
    return {k: v for k, v in asdict(row).items() if k not in {"original_path", "parsed_json_path"}}


def create_app(*, start_workers: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        acquire_runtime_lock(settings.database_path, "api")
        previous_openai_library = settings.allow_openai_library
        settings.allow_openai_library = True
        runner = None
        cleanup = None
        async def cleanup_loop():
            while True:
                await asyncio.to_thread(expire_assets)
                await asyncio.sleep(3600)
        try:
            store.initialize()
            from ..services.ingestion import recover_incomplete_ingestions
            app.state.startup_recovery = await asyncio.to_thread(recover_incomplete_ingestions)
            runner = JobRunner()
            app.state.runner = runner
            app.state.submit = runner.submit if start_workers else lambda task_id: None
            cleanup = asyncio.create_task(cleanup_loop()) if start_workers else None
            yield
        finally:
            if cleanup:
                cleanup.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cleanup
            try:
                if runner is not None:
                    await asyncio.to_thread(runner.close)
            finally:
                settings.allow_openai_library = previous_openai_library
                release_runtime_lock(settings.database_path)

    app = FastAPI(title="RefMind Research Workspace", version="0.2.0", lifespan=lifespan)

    @app.middleware("http")
    async def browser_boundaries(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
            from urllib.parse import urlsplit
            if urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "请求来源不匹配"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(PermissionError)
    async def forbidden(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(store.TaskQueueFull)
    async def queue_full(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=429)

    @app.exception_handler(ValueError)
    async def invalid(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(sqlite3.IntegrityError)
    async def duplicate(_request, _exc):
        return JSONResponse({"detail": "记录已存在或关联记录无效"}, status_code=409)

    def queue(request: Request, uid: int, kind: str, payload: dict) -> dict:
        with connect() as conn:
            active = conn.execute("SELECT count(*) FROM rm_tasks WHERE owner_id=? AND status IN ('queued','running','maintenance_pending')", (uid,)).fetchone()[0]
        if active >= 10:
            raise HTTPException(429, "待处理任务较多，请稍后再试")
        item = store.create_task(uid, kind, payload)
        request.app.state.submit(item["id"])
        return store.public_task(item)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.2.0"}

    @app.post("/api/auth/login")
    def login(body: LoginBody, request: Request, response: Response):
        key = (request.client.host if request.client else "local") + ":" + body.username.casefold()
        now = time.monotonic()
        with _attempt_lock:
            history = _attempts[key]
            while history and history[0] < now - 60:
                history.popleft()
            if len(history) >= 5:
                raise HTTPException(429, "登录尝试过于频繁，请稍后重试")
            history.append(now)
            if len(_attempts) > 2000:
                for old in list(_attempts)[:1000]:
                    if old != key:
                        _attempts.pop(old, None)
        result = store.login(body.username, body.password)
        if result is None:
            raise HTTPException(401, "用户名或密码不正确")
        token, csrf, user = result
        response.set_cookie(COOKIE, token, httponly=True, samesite="strict", secure=os.getenv("REFMIND_COOKIE_SECURE", "false").lower() == "true", max_age=12 * 3600, path="/")
        return {**user, "csrf": csrf}

    @app.get("/api/auth/me")
    def me(user: dict = Depends(current_user)):
        return user

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response, user: dict = Depends(current_user)):
        store.logout(request.cookies.get(COOKIE, ""))
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @app.post("/api/admin/users")
    def create_user(body: UserBody, user: dict = Depends(admin_user)):
        return store.create_user(body.username, body.password, admin=body.is_admin)

    @app.get("/api/admin/users")
    def users(user: dict = Depends(admin_user)):
        with connect() as conn:
            return [dict(r) for r in conn.execute("SELECT id,username,is_admin,disabled,created_at FROM rm_users ORDER BY id")]

    @app.get("/api/libraries")
    def libraries(user: dict = Depends(current_user)):
        return store.libraries(user["id"])

    @app.post("/api/libraries")
    def create_library(body: LibraryBody, user: dict = Depends(current_user)):
        return store.create_library(user["id"], body.title, "shared" if body.shared else "private")

    @app.post("/api/libraries/{group_id}/members")
    def member(group_id: int, body: MemberBody, user: dict = Depends(current_user)):
        with connect() as conn:
            target = conn.execute("SELECT id FROM rm_users WHERE username=? AND disabled=0", (body.username,)).fetchone()
        if not target:
            raise ValueError("未找到可加入的账号")
        store.grant_member(user["id"], group_id, target["id"], body.role)
        return {"ok": True}

    @app.get("/api/libraries/{group_id}/members")
    def members(group_id: int, user: dict = Depends(current_user)):
        if store.library_access(user["id"], group_id)["role"] != "owner":
            raise PermissionError("只有所有者可以查看成员")
        with connect() as conn:
            return [dict(r) for r in conn.execute("SELECT m.user_id,u.username,m.role FROM rm_members m JOIN rm_users u ON u.id=m.user_id WHERE m.group_id=? ORDER BY u.username", (group_id,))]

    @app.delete("/api/libraries/{group_id}/members/{member_id}")
    def remove_member(group_id: int, member_id: int, user: dict = Depends(current_user)):
        if store.library_access(user["id"], group_id)["role"] != "owner":
            raise PermissionError("只有所有者可以管理成员")
        with connect() as conn:
            conn.execute("DELETE FROM rm_members WHERE group_id=? AND user_id=?", (group_id, member_id))
        return {"ok": True}

    @app.get("/api/libraries/{group_id}/documents")
    def documents(group_id: int, user: dict = Depends(current_user)):
        store.library_access(user["id"], group_id)
        return [_public_document(d) for d in storage.list_documents(group_id)]

    @app.post("/api/libraries/{group_id}/documents")
    async def upload(group_id: int, file: UploadFile, request: Request, user: dict = Depends(current_user)):
        store.library_access(user["id"], group_id, write=True)
        filename = Path((file.filename or "paper.pdf").replace("\\", "/")).name
        if not filename.lower().endswith(".pdf"):
            raise ValueError("请上传 PDF 文件")
        upload_dir = workspace() / "uploads"
        upload_dir.mkdir(exist_ok=True)
        path = upload_dir / (secrets.token_hex(16) + ".pdf")
        digest, count = hashlib.sha256(), 0
        try:
            with path.open("xb") as handle:
                while block := await file.read(1024 * 1024):
                    count += len(block)
                    if count > 50 * 1024 * 1024:
                        raise ValueError("文件超过 50 MB")
                    digest.update(block)
                    handle.write(block)
            import fitz
            with fitz.open(path) as pdf:
                if not pdf.is_pdf or pdf.is_encrypted or len(pdf) == 0:
                    raise ValueError("需要可读取且未加密的 PDF")
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        try:
            return queue(request, user["id"], "upload", {"group_id": group_id, "path": str(path), "filename": filename, "sha256": digest.hexdigest()})
        except BaseException:
            # A rejected queue has no task to own or expire this file. If task
            # persistence succeeded but submission failed, retain it for retry.
            with connect() as conn:
                owned = any(json.loads(row["payload"]).get("path") == str(path) for row in conn.execute("SELECT payload FROM rm_tasks WHERE owner_id=? AND kind='upload'", (user["id"],)))
            if not owned:
                path.unlink(missing_ok=True)
            raise

    @app.get("/api/documents/{doc_id}/index-maintenance")
    def index_maintenance(doc_id: int, user: dict = Depends(current_user)):
        from .maintenance import inspect_document
        return inspect_document(user["id"], doc_id)

    @app.post("/api/documents/{doc_id}/reindex-requests")
    def reindex_request(doc_id: int, body: ReindexRequestBody, user: dict = Depends(current_user)):
        from .maintenance import request_rebuild
        return store.public_task(request_rebuild(user["id"], doc_id, body.expected_sha256))

    @app.get("/api/documents/{doc_id}/pdf")
    def document_pdf(doc_id: int, user: dict = Depends(current_user)):
        row = storage.get_document(doc_id)
        if not row:
            raise HTTPException(404, "文献不存在")
        store.library_access(user["id"], row.group_id)
        path = Path(row.original_path or "")
        if not path.is_file() or not path.resolve().is_relative_to(settings.upload_dir.resolve()):
            raise HTTPException(404, "原文不可用")
        return FileResponse(path, media_type="application/pdf")

    @app.get("/api/sessions")
    def sessions(user: dict = Depends(current_user)):
        return store.sessions(user["id"])

    @app.post("/api/sessions")
    def new_session(body: SessionBody, user: dict = Depends(current_user)):
        return store.create_session(user["id"], body.name, body.group_id)

    @app.patch("/api/sessions/{session_id}")
    def rename_session(session_id: int, body: SessionBody, user: dict = Depends(current_user)):
        store.session_access(user["id"], session_id)
        storage.rename_session(session_id, body.name)
        return store.session_access(user["id"], session_id)

    @app.get("/api/sessions/{session_id}/messages")
    def messages(session_id: int, user: dict = Depends(current_user)):
        store.session_access(user["id"], session_id)
        with connect() as conn:
            snapshots = {r["message_id"]: json.loads(r["snapshot"]) for r in conn.execute("SELECT e.* FROM rm_answer_evidence e JOIN messages m ON m.id=e.message_id WHERE m.session_id=?", (session_id,))}
        return [{**asdict(m), "result": snapshots.get(m.id)} for m in storage.list_messages(session_id)]

    @app.get("/api/sessions/{session_id}/assets")
    def assets(session_id: int, user: dict = Depends(current_user)):
        store.session_access(user["id"], session_id)
        with connect() as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM rm_assets WHERE session_id=? AND owner_id=? ORDER BY created_at DESC", (session_id, user["id"]))]
        return [public_asset(store.asset(i, user["id"])) for i in ids]

    @app.post("/api/academic-searches")
    def search(body: SearchBody, request: Request, user: dict = Depends(current_user)):
        if not settings.academic_search_enabled:
            raise HTTPException(409, "外部检索已由管理员关闭")
        if body.provider not in {"auto", "semantic_scholar", "openalex", "crossref"}:
            raise ValueError("不支持的学术来源")
        if body.year_from and body.year_to and body.year_from > body.year_to:
            raise ValueError("起始年份不能晚于截止年份")
        return queue(request, user["id"], "search", {"query": body.query, "provider": body.provider, "limit": body.limit, "cursor": body.cursor, "filters": {"year_from": body.year_from, "year_to": body.year_to, "open_access_only": body.open_access_only, "publication_types": body.publication_types}})

    @app.post("/api/reading-tasks")
    def reading(body: ReadingBody, request: Request, user: dict = Depends(current_user)):
        store.session_access(user["id"], body.session_id)
        _candidate(user["id"], body.search_id, body.paper_id)
        return queue(request, user["id"], "reading", body.model_dump())

    @app.get("/api/academic-searches/{task_id}")
    @app.get("/api/reading-tasks/{task_id}")
    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str, user: dict = Depends(current_user)):
        return store.public_task(store.task(task_id, user["id"]))

    @app.get("/api/tasks")
    def task_list(user: dict = Depends(current_user)):
        with connect() as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM rm_tasks WHERE owner_id=? ORDER BY created_at DESC LIMIT 100", (user["id"],))]
        return [store.public_task(store.task(tid, user["id"])) for tid in ids]

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel(task_id: str, user: dict = Depends(current_user)):
        task = store.task(task_id, user["id"])
        if task["kind"] == "legacy_reindex":
            from .maintenance import cancel_request
            cancel_request(user["id"], task_id)
            return {"ok": True}
        if task["status"] not in store.TERMINAL:
            with connect() as conn:
                conn.execute("UPDATE rm_tasks SET cancel_requested=1 WHERE id=?", (task_id,))
        return {"ok": True}

    @app.post("/api/tasks/{task_id}/retry")
    def retry(task_id: str, request: Request, user: dict = Depends(current_user)):
        task = store.task(task_id, user["id"])
        if task["status"] not in {"failed", "interrupted", "cancelled"}:
            raise ValueError("该任务不需要重试")
        if task["kind"] == "legacy_reindex":
            from .maintenance import retry_request
            return store.public_task(retry_request(user["id"], task_id))
        current, changed = store.requeue_task(task_id, user["id"])
        if changed:
            request.app.state.submit(task_id)
        return store.public_task(current)

    @app.get("/api/tasks/{task_id}/events")
    async def events(task_id: str, request: Request, after: int = 0, user: dict = Depends(current_user)):
        store.task(task_id, user["id"])
        async def stream():
            cursor = max(0, after)
            while not await request.is_disconnected():
                # Revocation is honored on an already-open event stream.
                if not store.authenticate(request.cookies.get(COOKIE, "")):
                    return
                with connect() as conn:
                    rows = conn.execute("SELECT * FROM rm_events WHERE task_id=? AND id>? ORDER BY id LIMIT 100", (task_id, cursor)).fetchall()
                for row in rows:
                    cursor = row["id"]
                    yield f"id: {cursor}\nevent: progress\ndata: {row['data']}\n\n"
                task = store.task(task_id, user["id"])
                if task["status"] in store.TERMINAL:
                    yield "event: complete\ndata: " + json.dumps(store.public_task(task), ensure_ascii=False) + "\n\n"
                    return
                if not rows:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.get("/api/assets/{asset_id}/pdf")
    def asset_pdf(asset_id: str, user: dict = Depends(current_user)):
        item = store.asset(asset_id, user["id"])
        if item["status"] == "expired":
            raise HTTPException(410, "临时原文已过期，历史证据快照仍保留")
        path = Path(item["file_info"]["path"])
        if not path.is_file() or not path.resolve().is_relative_to((workspace() / "downloads").resolve()):
            raise HTTPException(404, "原文不可用")
        return FileResponse(path, media_type="application/pdf")

    @app.post("/api/reading-tasks/{asset_id}/confirm")
    def confirm(asset_id: str, body: ConfirmBody, request: Request, user: dict = Depends(current_user)):
        item = store.asset(asset_id, user["id"])
        if item["status"] == "needs_review" and body.identity_confirmed:
            info = item["file_info"]
            if info["identity_status"] != "unknown":
                raise ValueError("该文件不能通过人工确认继续")
            info["identity_status"] = "user_confirmed"
            info["confirmed_by"] = user["id"]
            info["confirmed_at"] = time.time()
            with connect() as conn:
                conn.execute("UPDATE rm_assets SET file_info=?,status='downloaded' WHERE id=?", (json.dumps(info, ensure_ascii=False), asset_id))
            store.update_task(asset_id, status="queued")
            request.app.state.submit(asset_id)
        elif item["status"] == "partial" and body.accept_partial:
            info = item["file_info"]
            info["partial_accepted"] = True
            with connect() as conn:
                conn.execute("UPDATE rm_assets SET status='ready',file_info=? WHERE id=?", (json.dumps(info, ensure_ascii=False), asset_id))
            store.update_task(asset_id, status="completed", stage="仅研读已解析页面", progress=1, result=public_asset(store.asset(asset_id, user["id"])))
        else:
            raise ValueError("确认内容与当前资料状态不匹配")
        return store.public_task(store.task(asset_id, user["id"]))

    @app.post("/api/reading-tasks/{asset_id}/imports")
    def import_asset(asset_id: str, body: ImportBody, request: Request, user: dict = Depends(current_user)):
        store.asset(asset_id, user["id"], require_ready=True)
        store.library_access(user["id"], body.group_id, write=True)
        return queue(request, user["id"], "import", {"asset_id": asset_id, "group_id": body.group_id})

    @app.post("/api/answers")
    def answer(body: AnswerBody, request: Request, user: dict = Depends(current_user)):
        store.session_access(user["id"], body.session_id)
        if body.mode not in {"library", "external", "mixed"}:
            raise ValueError("无效研读模式")
        if body.mode == "external" and not (body.asset_ids or body.abstracts):
            raise ValueError("请先选择摘要或已就绪的全文")
        return queue(request, user["id"], "answer", body.model_dump())

    @app.get("/api/config")
    def config(user: dict = Depends(current_user)):
        return {"external_search_enabled": settings.academic_search_enabled,
                "chat_provider": os.getenv("CHAT_PROVIDER", "dashscope"),
                "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "dashscope"),
                "model_configured": settings.has_api_key, "temporary_retention_days": 7,
                "max_pdf_mb": 50}

    dist = settings.project_root / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
