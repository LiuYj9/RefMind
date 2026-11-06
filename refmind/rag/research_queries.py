"""Optional multi-query recall, bounded to one immutable library snapshot.

Each query uses the existing hybrid retriever. Across queries we merge identities,
not scores; all candidates subsequently face the original question in one rerank.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from contextvars import copy_context
from copy import deepcopy
import time

from langchain_core.documents import Document

from .. import storage
from ..config import settings

_AUTO_LLM = object()


def _limit() -> int:
    return max(1, min(3, int(settings.research_max_queries)))


def plan_research_queries(question: str, *, planner_llm=_AUTO_LLM) -> dict:
    """Reuse the established planner; explicit None supports a no-model plan.

The original question is always first. Numeric scientific conditions cannot be
silently replaced by a planner: branches with changed measurements are rejected.
This mechanical guard does not claim to prove semantic equivalence.
"""
    from ..agents.orchestration import MultiAgentConfig, PlanningAgent, _with_retrieval_anchor
    from .scientific_retrieval import scientific_aliases
    limit = _limit()
    fallback = {"queries": [question], "degraded": False, "warnings": [], "source": "original"}
    if not settings.research_multi_query_enabled or limit == 1:
        return fallback
    try:
        if planner_llm is _AUTO_LLM:
            from ..llm import get_llm
            planner_llm = get_llm(temperature=0)
        planned = PlanningAgent(planner_llm, MultiAgentConfig(max_subqueries=limit)).plan(question)
        queries = _with_retrieval_anchor(planned.queries, question, limit)
        # Preserve the exact original text as the first trace/query entry.
        queries[0] = question
        measurements = {x for x in scientific_aliases(question) if x.startswith("measurement:")}
        accepted, rejected = [question], 0
        for branch in queries[1:]:
            branch_measurements = {x for x in scientific_aliases(branch) if x.startswith("measurement:")}
            if branch_measurements != measurements:
                rejected += 1
            else:
                accepted.append(branch)
        warnings = ["查询规划不可用，已保留原问题检索。"] if planned.degraded else []
        if rejected:
            warnings.append("补充查询的数值条件不一致，已舍弃该分支并保留原问题。")
        return {"queries": accepted, "degraded": bool(planned.degraded or rejected),
                "warnings": warnings, "source": "planner", "rejected_branches": rejected}
    except Exception:
        return {**fallback, "degraded": True, "warnings": ["查询规划不可用，已保留原问题检索。"]}


def snapshot_from_rows(group_id: int, doc_ids: list[int], rows: dict) -> tuple:
    return tuple((group_id, doc_id, getattr(rows[doc_id], "index_version", None),
                  getattr(rows[doc_id], "file_sha256", None)) for doc_id in doc_ids)


def snapshot_is_current(snapshot: tuple) -> bool:
    for group_id, doc_id, version, digest in snapshot:
        row = storage.get_document(doc_id)
        if (row is None or row.status != "ready" or row.group_id != group_id
                or getattr(row, "index_version", None) != version
                or getattr(row, "file_sha256", None) != digest):
            return False
    return True


def filter_snapshot(documents: list[Document], snapshot: tuple) -> list[Document]:
    """Reject scope/version drift rather than mixing two ingestions in one answer."""
    expected = {entry[1]: entry for entry in snapshot}
    valid = {}
    for doc_id, entry in expected.items():
        row = storage.get_document(doc_id)
        if (row is not None and row.status == "ready" and row.group_id == entry[0]
                and getattr(row, "index_version", None) == entry[2]
                and getattr(row, "file_sha256", None) == entry[3]):
            valid[doc_id] = entry
    output = []
    for doc in documents:
        meta = doc.metadata
        try:
            entry = valid.get(int(meta.get("doc_id", -1)))
        except (TypeError, ValueError):
            continue
        if entry is None:
            continue
        try:
            if int(meta.get("group_id", entry[0])) != entry[0]:
                continue
        except (TypeError, ValueError):
            continue
        version, digest = entry[2:]
        if version is not None or digest is not None:
            if not version or not digest or meta.get("version") != version or meta.get("file_sha256") != digest:
                continue
        output.append(doc)
    return output


def _identity(document: Document) -> tuple:
    meta = document.metadata
    source = (meta.get("group_id"), meta.get("doc_id"), meta.get("version"), meta.get("file_sha256"))
    if meta.get("chunk_id") not in (None, ""):
        return (*source, "chunk", str(meta["chunk_id"]))
    return (*source, "legacy", meta.get("chunk_index"), meta.get("page"), " ".join(document.page_content.split()))


def _observation(document: Document, query_id: str, query: str) -> dict:
    channels = deepcopy(document.metadata.get("retrieval_channels", {}))
    fusion = deepcopy(document.metadata.get("retrieval_fusion", {}))
    observation = {"query_id": query_id, "text": query, "channels": channels, "fusion": fusion,
                   "raw_scores": None, "raw_score_status": "not_collected",
                   "rrf_score": None, "rrf_score_status": "not_collected",
                   "rank_kind": "per_query_weighted_rrf" if fusion.get("method") == "weighted_rrf" else None}
    # The existing observer records only first ranks. Repeated backend hits can
    # contribute again in EnsembleRetriever; reconstructing a total from first
    # ranks would silently invent an inaccurate score. Keep the observed rank.
    return observation


def retrieve_query_candidates(group_id: int, query: str, snapshot: tuple, *, query_plan: dict | None = None) -> dict:
    """Concurrent independent hybrid calls, deterministic merge in query order."""
    from ..agents.orchestration import _with_retrieval_anchor
    from .retrieval import get_retriever
    started = time.perf_counter()
    plan = (query_plan if query_plan is not None else plan_research_queries(query)) if snapshot else {"queries": [query]}
    queries = _with_retrieval_anchor(plan.get("queries", []), query, _limit())
    queries[0] = query
    warnings = list(plan.get("warnings", []))
    degraded = bool(plan.get("degraded"))
    ids = tuple(entry[1] for entry in snapshot)
    records = [{"query_id": f"q{i + 1}", "text": text, "status": "pending", "documents": []}
               for i, text in enumerate(queries)]
    try:
        retriever = get_retriever(group_id, selected_doc_ids=ids) if ids else None
    except Exception as exc:
        for record in records:
            record.update(status="error", error_type=type(exc).__name__)
        return {"documents": [], "queries": queries, "query_records": records, "degraded": True,
                "retrieval_failed": True, "warnings": [*warnings, "检索服务暂不可用。"],
                "elapsed": time.perf_counter() - started}
    if retriever is None:
        for record in records:
            record["status"] = "empty"
    else:
        executor = ThreadPoolExecutor(max_workers=min(len(queries), max(1, min(3, int(settings.research_query_workers)))))
        futures = {executor.submit(copy_context().run, retriever.invoke, text): i for i, text in enumerate(queries)}
        done, pending = wait(futures, timeout=max(.001, min(300., float(settings.research_query_timeout))))
        try:
            for future, index in futures.items():
                record = records[index]
                if future in pending:
                    record["status"] = "timeout"
                    future.cancel()
                    continue
                try:
                    hits = list(future.result() or [])
                    accepted = filter_snapshot(hits, snapshot)
                    record.update(status="ok" if accepted else "empty", documents=accepted,
                                  returned_count=len(hits), snapshot_rejected_count=len(hits) - len(accepted))
                except Exception as exc:
                    record.update(status="error", error_type=type(exc).__name__)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    failed = [r for r in records if r["status"] in {"error", "timeout"}]
    if failed:
        degraded = True
        warnings.append("部分查询失败或超时，已使用成功返回的候选。" if len(failed) < len(records) else "所有查询均调用失败，当前无法完成检索。")
    merged = {}
    for record in records:
        # Recheck at the merge boundary, including requests that finished earlier.
        record["documents"] = filter_snapshot(record["documents"], snapshot)
        for document in record["documents"]:
            key = _identity(document)
            if key not in merged:
                merged[key] = Document(page_content=document.page_content, metadata={**deepcopy(document.metadata), "retrieval_runs": []})
            runs = merged[key].metadata["retrieval_runs"]
            if not any(run["query_id"] == record["query_id"] for run in runs):
                runs.append(_observation(document, record["query_id"], record["text"]))
    scope_changed = not snapshot_is_current(snapshot)
    if scope_changed:
        warnings.append("文献版本或可用范围在检索期间发生变化，请重新发起请求。")
    return {"documents": [] if scope_changed else list(merged.values()), "queries": queries, "query_records": records,
            "degraded": bool(degraded or scope_changed), "retrieval_failed": len(failed) == len(records) or scope_changed,
            "scope_changed": scope_changed, "warnings": warnings,
            "elapsed": time.perf_counter() - started}
