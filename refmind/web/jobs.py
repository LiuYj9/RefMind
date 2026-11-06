"""Durable, owner-scoped jobs; expensive work happens outside HTTP requests."""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import RLock

from .. import storage
from ..config import settings
from ..storage.connection import connect
from . import store

LOG = logging.getLogger(__name__)
_IMPORT_LOCK = RLock()


class Cancelled(Exception):
    pass


def workspace() -> Path:
    target = settings.database_path.parent / "research_workspace"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _candidate(user_id: int, search_id: str, paper_id: str) -> dict:
    search = store.task(search_id, user_id)
    if search["kind"] != "search":
        raise ValueError("请选择论文检索结果")
    for paper in search["result"].get("papers", []):
        if paper["paper_id"] == paper_id:
            return paper
    raise ValueError("论文不在所选检索结果中")


def public_asset(item: dict) -> dict:
    return {"id": item["id"], "session_id": item["session_id"], "doc_id": item["doc_id"],
            "paper": item["paper"], "status": item["status"], "expires_at": item["expires_at"],
            "identity_status": item["file_info"].get("identity_status"),
            "sha256": item["file_info"].get("sha256"), "version": item["file_info"].get("version"),
            "warnings": item["file_info"].get("warnings", []),
            "coverage": item["file_info"].get("coverage")}


class JobRunner:
    def __init__(self, workers: int = 3):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="refmind-web")
        self._lock = RLock()
        self._active: set[str] = set()
        self._imports = _IMPORT_LOCK

    def submit(self, task_id: str) -> None:
        with self._lock:
            if task_id in self._active:
                return
            self._active.add(task_id)
        self.pool.submit(self._run, task_id)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)

    def _check(self, task_id: str) -> dict:
        task = store.task(task_id)
        if task["cancel_requested"]:
            raise Cancelled()
        with connect() as conn:
            enabled = conn.execute("SELECT id FROM rm_users WHERE id=? AND disabled=0", (task["owner_id"],)).fetchone()
        if not enabled:
            raise PermissionError("账号已停用")
        return task

    def _progress(self, task_id: str, value: float, stage: str) -> None:
        self._check(task_id)
        store.update_task(task_id, progress=value, stage=stage)

    def _run(self, task_id: str) -> None:
        try:
            if store.task(task_id)["status"] != "queued":
                return
            task = self._check(task_id)
            # Claim once across accidental duplicate submissions/runner objects.
            with connect() as conn:
                claimed = conn.execute("UPDATE rm_tasks SET status='running',stage='正在准备',error='',updated_at=? WHERE id=? AND status='queued'", (time.time(), task_id)).rowcount
            if not claimed:
                return
            handler = getattr(self, "_" + task["kind"])
            result = handler(task)
            if store.task(task_id)["status"] == "running":
                self._check(task_id)
                store.update_task(task_id, status="completed", progress=1, stage="已完成", result=result)
        except Cancelled:
            # Completion and document registration are atomic; a late cancel
            # cannot turn a committed import into an apparently cancelled one.
            if store.task(task_id)["status"] == "completed":
                return
            try:
                self._cleanup_cancelled(store.task(task_id))
            except Exception:
                LOG.exception("Cancelled job cleanup failed: %s", task_id)
                store.update_task(task_id, status="failed", stage="cleanup_failed", error="取消后的清理未完成，可重试")
            else:
                store.update_task(task_id, status="cancelled", stage="已取消")
        except Exception as exc:
            LOG.exception("Research job %s failed", task_id)
            if store.task(task_id)["status"] == "completed":
                return
            if isinstance(exc, (PermissionError, ValueError)):
                try:
                    self._cleanup_cancelled(store.task(task_id))
                except Exception:
                    LOG.exception("Permission-revoked job cleanup failed: %s", task_id)
            # API keys and local paths must not be exposed through exception bodies.
            message = str(exc) if isinstance(exc, (ValueError, PermissionError)) else "任务未完成，请查看服务日志或重试"
            message = re.sub(r"[A-Za-z]:\\[^\s]+", "[local path]", re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", message))
            store.update_task(task_id, status="failed", error=message[:500], stage=type(exc).__name__)
        finally:
            with self._lock:
                self._active.discard(task_id)

    @staticmethod
    def _filename(task_id: str, sha256: str) -> str:
        # Persisted before any metadata registration: a ready document can be
        # recovered by this deterministic name after process death.
        return f"refmind-{task_id}-{sha256[:16]}.pdf"

    def _source(self, path: str, sha256: str, *, downloads: bool = False) -> Path:
        root = (workspace() / ("downloads" if downloads else "uploads")).resolve()
        source = Path(path)
        if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(root):
            raise ValueError("原文文件不存在或不在受控目录")
        if hashlib.sha256(source.read_bytes()).hexdigest() != sha256:
            raise ValueError("原文文件已变化，请重新获取")
        return source

    def _existing_document(self, task: dict, group_id: int, sha256: str):
        from ..services.ingestion import remove_document
        with connect() as conn:
            known = conn.execute("SELECT doc_id FROM rm_document_sources WHERE group_id=? AND sha256=?", (group_id, sha256)).fetchone()
        if known:
            row = storage.get_document(known["doc_id"])
            if row is None or row.status != "ready":
                raise ValueError("已有文档未就绪，需要先修复文献库")
            if not row.original_path or not Path(row.original_path).is_file() or hashlib.sha256(Path(row.original_path).read_bytes()).hexdigest() != sha256:
                raise ValueError("已有文档原文已变化，需要先修复文献库")
            return row, True
        marker = self._filename(task["id"], sha256)
        found = [d for d in storage.list_documents(group_id) if d.filename == marker]
        for row in found:
            if row.status == "ready" and row.original_path and row.parsed_json_path:
                source, parsed = Path(row.original_path), Path(row.parsed_json_path)
                if source.is_file() and parsed.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == sha256:
                    return row, False
            # Only this task's deterministic document is eligible for repair.
            remove_document(row.id)
        return None, False

    def _ingest_once(self, task: dict, group_id: int, source: Path, sha256: str):
        from ..services.ingestion import ingest_pdf
        self._check(task["id"])
        row, registered = self._existing_document(task, group_id, sha256)
        if row is None:
            row = ingest_pdf(group_id, source, self._filename(task["id"], sha256), make_summary=False,
                             progress=lambda value, stage: self._progress(task["id"], .2 + .75 * value, stage))
        return row, registered

    def _register(self, conn, row, group_id: int, sha256: str, metadata: dict, display_filename: str):
        conn.execute("INSERT INTO rm_document_sources(doc_id,group_id,sha256,metadata) VALUES(?,?,?,?) ON CONFLICT(group_id,sha256) DO NOTHING",
                     (row.id, group_id, sha256, json.dumps(metadata, ensure_ascii=False)))
        found = conn.execute("SELECT doc_id FROM rm_document_sources WHERE group_id=? AND sha256=?", (group_id, sha256)).fetchone()
        if found["doc_id"] != row.id:
            raise ValueError("文档登记冲突，请重试")
        conn.execute("UPDATE documents SET filename=? WHERE id=?", (Path(display_filename).name, row.id))

    def _complete(self, conn, task_id: str, result: dict):
        now = time.time()
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        # Check cancellation in the same write transaction as the commit point.
        cancelled = conn.execute("SELECT cancel_requested FROM rm_tasks WHERE id=?", (task_id,)).fetchone()
        if cancelled is None or cancelled[0]:
            raise Cancelled()
        conn.execute("UPDATE rm_tasks SET status='completed',progress=1,stage='已完成',result=?,updated_at=? WHERE id=?",
                     (json.dumps(result, ensure_ascii=False), now, task_id))
        conn.execute("INSERT INTO rm_events(task_id,data,created_at) VALUES(?,?,?)", (task_id, json.dumps({"status": "completed", "progress": 1}), now))

    def _cleanup_cancelled(self, task: dict):
        from ..services.ingestion import remove_document, remove_group
        p, tid = task["payload"], task["id"]
        with self._imports:
            if task["kind"] == "reading":
                with connect() as conn:
                    raw = conn.execute("SELECT * FROM rm_assets WHERE id=? AND owner_id=?", (tid, task["owner_id"])).fetchone()
                if raw:
                    gid = raw["group_id"]
                    if gid:
                        with connect() as conn:
                            owned = conn.execute("SELECT group_id FROM rm_libraries WHERE group_id=? AND owner_id=? AND kind='temporary'", (gid, task["owner_id"])).fetchone()
                        if not owned:
                            raise PermissionError("临时资料目录归属异常")
                        for row in storage.list_documents(gid):
                            remove_document(row.id)
                    with connect() as conn:
                        conn.execute("UPDATE rm_assets SET status='cancelled',doc_id=NULL,group_id=NULL WHERE id=?", (tid,))
                    if gid:
                        try:
                            remove_group(gid)
                        except Exception:
                            with connect() as conn:
                                conn.execute("UPDATE rm_assets SET group_id=? WHERE id=?", (gid, tid))
                            raise
                folder = (workspace() / "downloads" / tid).resolve()
                if folder.is_relative_to((workspace() / "downloads").resolve()) and folder.is_dir():
                    # Flat, service-created PDF files only; never recurse or follow links.
                    for path in folder.iterdir():
                        if path.is_file() and not path.is_symlink() and path.suffix in {".pdf", ".downloading"}:
                            path.unlink()
            elif task["kind"] in {"upload", "import"}:
                sha256 = p.get("sha256")
                if task["kind"] == "import":
                    asset = store.asset(p["asset_id"], task["owner_id"])
                    sha256 = asset["file_info"]["sha256"]
                if sha256:
                    for row in storage.list_documents(p["group_id"]):
                        if row.filename == self._filename(tid, sha256):
                            remove_document(row.id)

    def _search(self, task: dict) -> dict:
        from ..services.paper_discovery import SearchFilters, discover_papers
        p = task["payload"]
        self._progress(task["id"], .1, "正在查询学术索引")
        def partial(snapshot):
            self._check(task["id"])
            done = sum(value != "running" for value in snapshot.provider_status.values())
            store.update_task(task["id"], result={**snapshot.to_dict(), "query": p["query"], "provider": p.get("provider", "auto"), "filters": p.get("filters", {}), "limit": p.get("limit", 20)},
                progress=.1 + .8 * done / max(1, len(snapshot.provider_status)), stage="已收到部分来源，继续检索")
        result = discover_papers(p["query"], provider=p.get("provider", "auto"),
                                filters=SearchFilters(**p.get("filters", {})), limit=p.get("limit", 20),
                                cursor=p.get("cursor"), timeout_seconds=20, on_progress=partial)
        out = {**result.to_dict(), "query": p["query"], "provider": p.get("provider", "auto"), "filters": p.get("filters", {}), "limit": p.get("limit", 20)}
        if out.get("failed"):
            store.update_task(task["id"], status="failed", result=out, stage="检索源暂不可用")
        elif out.get("warnings"):
            store.update_task(task["id"], status="partial", progress=1, result=out, stage="已保留可用结果")
        return out

    def _reading(self, task: dict) -> dict:
        from ..services.paper_discovery import PaperCandidate
        from ..services.fulltext import fetch_public_pdf
        p, uid, tid = task["payload"], task["owner_id"], task["id"]
        store.session_access(uid, p["session_id"])
        paper = _candidate(uid, p["search_id"], p["paper_id"])
        with connect() as conn:
            found = conn.execute("SELECT id FROM rm_assets WHERE id=? AND owner_id=?", (tid, uid)).fetchone()
        if found:
            asset = store.asset(tid, uid)
            info = asset["file_info"]
            if asset["status"] == "ready":
                return public_asset(asset)
            if asset["status"] == "partial":
                result = public_asset(asset)
                store.update_task(tid, status="needs_review", stage="部分页面未解析，请选择是否研读可用部分", result=result)
                return result
            if asset["status"] == "expired":
                raise ValueError("会话原文已过期，请重新获取")
            if asset["status"] == "cancelled":
                # Cancellation removes task-owned files; retry explicitly fetches anew.
                with connect() as conn:
                    conn.execute("DELETE FROM rm_assets WHERE id=?", (tid,))
                found = None
        if not found:
            self._progress(tid, .1, "正在获取并核验公开 PDF")
            fetched = fetch_public_pdf(PaperCandidate.from_dict(paper), workspace() / "downloads" / tid,
                                       location_index=p.get("location_index", 0))
            info = asdict(fetched)
            info["path"] = str(info["path"])
            self._check(tid)
            now = time.time()
            with connect() as conn:
                conn.execute("INSERT INTO rm_assets(id,owner_id,session_id,paper,file_info,status,expires_at,created_at) VALUES(?,?,?,?,?,'downloaded',?,?)", (tid, uid, p["session_id"], json.dumps(paper, ensure_ascii=False), json.dumps(info, ensure_ascii=False), now + 7 * 86400, now))
        if info["identity_status"] not in {"matched", "user_confirmed"}:
            self._check(tid)
            result = {**public_asset(store.asset(tid, uid)), "status": "needs_review"}
            with connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                if conn.execute("SELECT cancel_requested FROM rm_tasks WHERE id=?", (tid,)).fetchone()[0]:
                    raise Cancelled()
                conn.execute("UPDATE rm_assets SET status='needs_review' WHERE id=?", (tid,))
                conn.execute("UPDATE rm_tasks SET status='needs_review',stage='请核对原文身份后继续',result=?,updated_at=? WHERE id=?", (json.dumps(result, ensure_ascii=False), time.time(), tid))
            return result
        source = self._source(info["path"], info["sha256"], downloads=True)
        with self._imports:
            self._check(tid)
            store.session_access(uid, p["session_id"])
            # Library creation and linking are one transaction, avoiding an
            # unreachable temporary library if the process dies between them.
            with connect() as conn:
                asset_row = conn.execute("SELECT group_id FROM rm_assets WHERE id=?", (tid,)).fetchone()
                gid = asset_row["group_id"]
                if not gid:
                    gid = conn.execute("INSERT INTO groups(name) VALUES(?)", ("web:" + secrets.token_hex(12),)).lastrowid
                    conn.execute("INSERT INTO rm_libraries VALUES(?,?,?,'temporary')", (gid, uid, paper["title"]))
                conn.execute("UPDATE rm_assets SET group_id=?,status='parsing' WHERE id=?", (gid, tid))
            row, registered = self._ingest_once(task, gid, source, info["sha256"])
            self._check(tid)
            store.session_access(uid, p["session_id"])
            parsed = json.loads(Path(row.parsed_json_path).read_text(encoding="utf-8"))
            pages = sorted({int(x.get("page", 0)) for x in parsed.get("pages", []) if x.get("page")})
            if not pages:
                pages = sorted({int(x.get("page", x.get("page_start", 0))) for x in parsed.get("blocks", []) if x.get("page", x.get("page_start"))})
            import fitz
            with fitz.open(source) as pdf:
                total = len(pdf)
            pages = [page for page in pages if 1 <= page <= total]
            info["coverage"] = {"parsed_pages": pages, "total_pages": total}
            partial = len(pages) < total
            # Register source, asset readiness and task completion together.
            with connect() as conn:
                if not registered:
                    self._register(conn, row, gid, info["sha256"], {"origin": "temporary", "asset_id": tid}, info["sha256"] + ".pdf")
                conn.execute("UPDATE rm_assets SET doc_id=?,file_info=?,status=? WHERE id=?", (row.id, json.dumps(info, ensure_ascii=False), "partial" if partial else "ready", tid))
                result = {**public_asset(store.asset(tid, uid)), "doc_id": row.id, "status": "partial" if partial else "ready", "coverage": info["coverage"]}
                if partial:
                    cancelled = conn.execute("SELECT cancel_requested FROM rm_tasks WHERE id=?", (tid,)).fetchone()
                    if cancelled[0]:
                        raise Cancelled()
                    conn.execute("UPDATE rm_tasks SET status='needs_review',stage='部分页面未解析，请选择是否研读可用部分',result=?,updated_at=? WHERE id=?", (json.dumps(result, ensure_ascii=False), time.time(), tid))
                else:
                    self._complete(conn, tid, result)
            return result

    def _upload(self, task: dict) -> dict:
        p, uid, tid = task["payload"], task["owner_id"], task["id"]
        if p.get("staging_expired"):
            raise ValueError("上传副本已超过7天保留期，请重新上传原PDF")
        store.library_access(uid, p["group_id"], write=True)
        with self._imports:
            row, registered = self._existing_document(task, p["group_id"], p["sha256"])
            if row is None:
                source = self._source(p["path"], p["sha256"])
                row, registered = self._ingest_once(task, p["group_id"], source, p["sha256"])
            self._check(tid)
            store.library_access(uid, p["group_id"], write=True)
            result = {"document_id": row.id, "reused": registered, "group_id": p["group_id"]}
            with connect() as conn:
                if not registered:
                    self._register(conn, row, p["group_id"], p["sha256"], {"origin": "upload", "task_id": tid}, p["filename"])
                self._complete(conn, tid, result)
            staging = Path(p["path"])
            if staging.is_file() and not staging.is_symlink() and staging.resolve().is_relative_to((workspace() / "uploads").resolve()):
                try:
                    staging.unlink(missing_ok=True)
                except OSError:
                    LOG.warning("Completed upload staging file retained for cleanup: %s", tid)
            return result

    def _import(self, task: dict) -> dict:
        p, uid, tid = task["payload"], task["owner_id"], task["id"]
        asset = store.asset(p["asset_id"], uid, require_ready=True)
        store.library_access(uid, p["group_id"], write=True)
        info = asset["file_info"]
        with self._imports:
            row, registered = self._existing_document(task, p["group_id"], info["sha256"])
            if row is None:
                source = self._source(info["path"], info["sha256"], downloads=True)
                row, registered = self._ingest_once(task, p["group_id"], source, info["sha256"])
            self._check(tid)
            store.asset(p["asset_id"], uid, require_ready=True)
            store.library_access(uid, p["group_id"], write=True)
            result = {"document_id": row.id, "reused": registered, "group_id": p["group_id"]}
            with connect() as conn:
                if not registered:
                    self._register(conn, row, p["group_id"], info["sha256"], {"paper": asset["paper"], "version": info.get("version"), "source_url": info.get("source_url"), "task_id": tid}, info["sha256"] + ".pdf")
                self._complete(conn, tid, result)
            return result

    def _answer(self, task: dict) -> dict:
        from langchain_core.documents import Document
        from langchain_core.messages import HumanMessage, AIMessage
        from ..rag.graph import answer_from_evidence
        from ..rag.scientific_retrieval import retrieve_for_task
        from ..rag.reranker import rerank
        from ..rag.research_pipeline import retrieve_research_candidates, run_research_rerank
        from ..rag.research_queries import filter_snapshot, snapshot_is_current
        from ..rag.evidence_bundle import describe_task, paired_history
        from ..contracts import ResearchScope
        p, uid, tid = task["payload"], task["owner_id"], task["id"]
        scope = ResearchScope(**p).model_dump()
        p = {**p, **scope}
        session = store.session_access(uid, p["session_id"])
        mode = p.get("mode", "library")
        if mode not in {"library", "external", "mixed"}:
            raise ValueError("无效研读模式")
        history = []
        for row in storage.list_messages(session["id"], limit=14):
            if row.role == "user":
                history.append(HumanMessage(content=row.content))
            elif row.role == "assistant":
                history.append(AIMessage(content=row.content))
        history = paired_history(history, 6)
        source_titles, source_ids, assets, abstracts = [], [], [], []
        gid = p.get("group_id") or session["group_id"]
        selected = p.get("selected_doc_ids")
        if selected is not None and any(type(v) is not int or v <= 0 for v in selected):
            raise ValueError("无效的文献选择")
        if mode in {"library", "mixed"}:
            store.library_access(uid, gid)
            if selected is not None:
                for doc_id in selected:
                    row = storage.get_document(doc_id)
                    if row is None or row.group_id != gid or row.status != "ready":
                        raise PermissionError("所选论文不属于可访问的就绪文献范围")
                    source_titles.append(row.paper_title or row.filename)
                    source_ids.append(f"library:{gid}:{doc_id}")
        if mode in {"external", "mixed"}:
            for asset_id in p.get("asset_ids", []):
                asset = store.asset(asset_id, uid, require_ready=True)
                if asset["session_id"] != session["id"]:
                    raise PermissionError("资料不属于当前会话")
                self._source(asset["file_info"]["path"], asset["file_info"]["sha256"], downloads=True)
                assets.append(asset)
                source_titles.append(asset["paper"]["title"])
                source_ids.append("asset:" + str(asset_id))
            for choice in p.get("abstracts", []):
                paper = _candidate(uid, choice["search_id"], choice["paper_id"])
                abstracts.append(paper)
                source_titles.append(paper["title"])
                source_ids.append("abstract:" + str(paper["paper_id"]))
        elif p.get("asset_ids") or p.get("abstracts"):
            raise ValueError("请先选择外部或混合研读模式")
        descriptor = describe_task(p["question"], history, source_titles, source_ids=source_ids)
        if descriptor.needs_clarification:
            output = {"answer": "请明确你指的是哪篇论文，或先在文献范围中选定需要研读的论文。", "status": "clarification",
                      "task_description": descriptor.to_dict(), "trace": {"stages": {}, "timings": {}},
                      "evidence": [], "session_id": session["id"], "mode": mode, "warnings": [], "research_scope": scope}
            self._check(tid)
            return self._save_answer(task, output)
        query = descriptor.resolved_query
        documents = []
        research_retrieval = None
        self._progress(tid, .1, "正在核对文献范围")
        if mode == "library":
            research_retrieval = retrieve_research_candidates(query, gid,
                selected_doc_ids=selected, task_type=descriptor.task_type)
            documents.extend(research_retrieval["documents"])
        elif mode == "mixed":
            documents.extend(retrieve_for_task(gid, query, selected_doc_ids=None if selected is None else tuple(selected), task_type=descriptor.task_type))
        if mode in {"external", "mixed"}:
            for asset in assets:
                asset_id = asset["id"]
                found = retrieve_for_task(asset["group_id"], query, selected_doc_ids=(asset["doc_id"],), task_type=descriptor.task_type)
                for d in found:
                    meta = {**d.metadata, "asset_id": asset_id, "resource_id": asset_id, "status": "ready", "identity_verified": True, "evidence_origin": "session_full_text", "evidence_level": "full_text", "source_url": asset["file_info"].get("source_url", ""), "paper_title": asset["paper"]["title"], "source_id": asset_id}
                    documents.append(Document(page_content=d.page_content, metadata=meta))
            for paper in abstracts:
                if not paper.get("abstract"):
                    continue
                doi = paper.get("doi", "")
                link = "https://doi.org/" + doi if doi else next((location.get("landing_url") for location in paper.get("locations", []) if location.get("landing_url")), "")
                documents.append(Document(page_content=paper["abstract"], metadata={"evidence_origin": "academic_search", "evidence_level": "abstract", "paper_title": paper["title"], "evidence_id": paper["paper_id"], "chunk_id": paper["paper_id"], "doi": doi, "external_url": link, "provider": ", ".join(paper.get("providers", []))}))
        self._progress(tid, .35, "正在组织证据并核对回答")
        rerank_started = time.perf_counter()
        documents = (run_research_rerank(query, documents) if mode == "library"
                     else rerank(query, documents, top_n=len(documents))) if documents else []
        if research_retrieval is not None:
            snapshot = research_retrieval.get("scope_snapshot")
            if snapshot is not None:
                documents = filter_snapshot(documents, snapshot)
                if not snapshot_is_current(snapshot):
                    documents = []
                    research_retrieval.update(retrieval_failed=True, degraded=True)
                    research_retrieval["warnings"].append("文献版本或可用范围在本轮处理中发生变化，请重新提问。")
            research_retrieval["trace"]["stages"]["reranked"] = {"documents": list(documents)}
            research_retrieval["trace"]["timings"]["rerank"] = time.perf_counter() - rerank_started
        def generation_guard(evidence):
            from ..rag.retrieval import filter_ready_documents
            self._check(tid)
            if mode in {"library", "mixed"}:
                store.library_access(uid, gid)
            for asset in assets:
                current = store.asset(asset["id"], uid, require_ready=True)
                self._source(current["file_info"]["path"], current["file_info"]["sha256"], downloads=True)
            guarded = []
            for item in evidence:
                if item.metadata.get("evidence_origin") == "academic_search":
                    guarded.append(item)
                elif item.metadata.get("evidence_origin") == "session_full_text":
                    if filter_ready_documents(int(item.metadata["group_id"]), [item], (int(item.metadata["doc_id"]),)):
                        guarded.append(item)
                else:
                    guarded.extend(filter_ready_documents(gid, [item], selected))
            if research_retrieval is not None and research_retrieval.get("scope_snapshot") is not None:
                guarded = filter_snapshot(guarded, research_retrieval["scope_snapshot"])
            return guarded
        if research_retrieval is not None and research_retrieval.get("retrieval_failed"):
            result = {"answer": "检索服务暂不可用，未据此判断文献是否能够回答。请稍后重试。",
                      "status": "system_error", "documents": [], "service_failed": True,
                      "degraded": True, "warnings": [], "trace": {}}
        else:
            enhancement_args = {"apply_research_enhancements": True} if mode == "library" else {}
            result = answer_from_evidence(p["question"], documents, history=history, user_id=str(uid), task_description=descriptor.to_dict(), evidence_guard=generation_guard, **enhancement_args)
        if research_retrieval is not None:
            trace = result.setdefault("trace", {})
            for field in ("stages", "timings"):
                for key, value in research_retrieval["trace"].get(field, {}).items():
                    trace.setdefault(field, {}).setdefault(key, value)
            for field in ("pipeline", "scope", "queries", "retrieval_channels", "retrieval_counts"):
                if field in research_retrieval["trace"]:
                    trace[field] = research_retrieval["trace"][field]
            result["warnings"] = [*research_retrieval["warnings"], *result.get("warnings", [])]
            result["degraded"] = bool(research_retrieval["degraded"] or result.get("degraded"))
        self._check(tid)
        # Permissions may have changed while models were running.
        if mode in {"library", "mixed"}:
            store.library_access(uid, gid)
        for asset in assets:
            current = store.asset(asset["id"], uid, require_ready=True)
            self._source(current["file_info"]["path"], current["file_info"]["sha256"], downloads=True)
        if research_retrieval is not None and research_retrieval.get("scope_snapshot") is not None and not snapshot_is_current(research_retrieval["scope_snapshot"]):
            # A late reindex/delete is an operational failure, not a verdict that
            # the papers lack evidence. Do not persist an answer to stale sources.
            result = {"answer": "文献版本或可用范围在本轮处理中发生变化，请重新提问。",
                      "status": "system_error", "documents": [], "service_failed": True,
                      "degraded": True, "review_status": "unreviewed", "rejection_stage": "scope_changed",
                      "warnings": ["已停止展示本轮旧版本证据和答案。"],
                      "trace": {"pipeline": "research", "status": "scope_changed", "stages": {}, "timings": {}}}
        allowed = {"paper_title", "doc_id", "group_id", "filename", "page", "page_start", "page_end", "bbox", "bbox_space", "bbox_coordinate_system", "source_locations", "content_type", "evidence_id", "evidence_origin", "evidence_level", "external_url", "source_url", "asset_id", "resource_id", "status", "file_sha256", "version", "library_index", "paragraph_index", "chunk_id", "section", "section_path", "provider", "evidence_view", "source_rows", "coverage_limit"}
        allowed.update({"view_id", "source_chunk_hash", "view_hash", "selected_spans", "omitted_spans",
                        "compression_status", "offset_unit", "interval", "score_type", "rerank_score", "rerank_backend",
                        "rerank_degraded", "rerank_reason", "rerank_capacity", "retrieval_channels",
                        "retrieval_fusion", "retrieval_runs"})
        evidence = [{"text": d.page_content, "metadata": {k: v for k, v in d.metadata.items() if k in allowed}} for d in result.get("documents", [])]
        def safe(value):
            if isinstance(value, Document):
                return {"text": value.page_content, "metadata": {k: safe(v) for k, v in value.metadata.items() if k in allowed}}
            if isinstance(value, dict):
                return {str(k): safe(v) for k, v in value.items() if str(k).casefold() not in {"path", "image_path", "original_path", "parsed_json_path", "api_key", "authorization", "source_text", "system_prompt", "user_prompt", "history", "rendered_evidence"}}
            if isinstance(value, (list, tuple)):
                return [safe(v) for v in value]
            if isinstance(value, str):
                return re.sub(r"[A-Za-z]:\\[^\s]+", "[local path]", re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", value))
            return value if value is None or isinstance(value, (int, float, bool)) else None
        output = {k: safe(result[k]) for k in ("answer", "status", "warnings", "service_failed", "rejection_stage", "degraded", "review_status", "evidence_review", "task_description", "trace") if k in result}
        from ..contracts import ResearchScope
        output.update({"evidence": safe(evidence), "session_id": session["id"], "mode": mode,
                       "research_scope": scope})
        output.setdefault("task_description", descriptor.to_dict())
        return self._save_answer(task, output)

    def _save_answer(self, task: dict, output: dict) -> dict:
        session_id = task["payload"]["session_id"]
        store.session_access(task["owner_id"], session_id)
        with connect() as conn:
            conn.execute("INSERT INTO messages(session_id,role,content) VALUES(?,'user',?)", (session_id, task["payload"]["question"]))
            mid = conn.execute("INSERT INTO messages(session_id,role,content) VALUES(?,'assistant',?)", (session_id, output["answer"])).lastrowid
            output["message_id"], output["task_id"] = mid, task["id"]
            conn.execute("INSERT INTO rm_answer_evidence VALUES(?,?)", (mid, json.dumps(output, ensure_ascii=False)))
            self._complete(conn, task["id"], output)
        return output


def expire_assets() -> dict:
    """Remove only expired session copies; imported documents are independent."""
    from ..services.ingestion import remove_document, remove_group
    with connect() as conn:
        rows = conn.execute("SELECT * FROM rm_assets WHERE expires_at<? AND status!='expired'", (time.time(),)).fetchall()
    expired, errors = [], []
    root = (workspace() / "downloads").resolve()
    for row in rows:
        try:
            with _IMPORT_LOCK:
                if row["group_id"]:
                    with connect() as conn:
                        owned = conn.execute("SELECT group_id FROM rm_libraries WHERE group_id=? AND owner_id=? AND kind='temporary'", (row["group_id"], row["owner_id"])).fetchone()
                    if not owned:
                        raise PermissionError("临时资料目录归属异常")
                if row["doc_id"]:
                    remove_document(row["doc_id"])
                info = json.loads(row["file_info"])
                path = Path(info["path"])
                if path.resolve().is_relative_to(root) and not path.is_symlink():
                    path.unlink(missing_ok=True)
                with connect() as conn:
                    conn.execute("UPDATE rm_assets SET status='expired',doc_id=NULL,group_id=NULL WHERE id=?", (row["id"],))
                if row["group_id"]:
                    try:
                        remove_group(row["group_id"])
                    except Exception:
                        with connect() as conn:
                            conn.execute("UPDATE rm_assets SET status='cleanup_failed',group_id=? WHERE id=?", (row["group_id"], row["id"]))
                        raise
            expired.append(row["id"])
        except Exception:
            errors.append(row["id"])
    upload_cleanup = expire_upload_staging()
    return {"expired": expired, "failed": errors, **upload_cleanup}


def expire_upload_staging(*, now: float | None = None) -> dict:
    """Expire only owner-scoped, inactive upload sources after seven days."""
    current_time = time.time() if now is None else now
    cutoff = current_time - 7 * 86400
    root = (workspace() / "uploads").resolve()
    expired, failed = [], []
    with _IMPORT_LOCK:
        with connect() as conn:
            rows = conn.execute("SELECT id FROM rm_tasks WHERE kind='upload' AND status IN ('completed','partial','failed','cancelled','interrupted') AND updated_at<?", (cutoff,)).fetchall()
        for row in rows:
            try:
                with connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    current = conn.execute("SELECT payload,status,updated_at FROM rm_tasks WHERE id=?", (row["id"],)).fetchone()
                    if current is None or current["status"] in {"queued", "running"} or current["updated_at"] >= cutoff:
                        continue
                    payload = json.loads(current["payload"])
                    if payload.get("staging_expired"):
                        continue
                    path = Path(payload["path"])
                    if path.is_symlink() or getattr(path, "is_junction", lambda: False)() or not path.resolve().is_relative_to(root):
                        raise ValueError("upload staging path escaped its controlled directory")
                    if path.exists() and not path.is_file():
                        raise ValueError("upload staging path is not a regular file")
                    path.unlink(missing_ok=True)
                    payload["staging_expired"] = True
                    payload["staging_expired_at"] = current_time
                    conn.execute("UPDATE rm_tasks SET payload=? WHERE id=?", (json.dumps(payload, ensure_ascii=False), row["id"]))
                expired.append(row["id"])
            except Exception:
                LOG.exception("Upload staging cleanup failed: %s", row["id"])
                failed.append(row["id"])
    return {"uploads_expired": expired, "upload_cleanup_failed": failed}
