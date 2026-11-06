"""Shared production research retrieval for the HTTP app and evaluation runner."""
from __future__ import annotations
import time
from langchain_core.documents import Document

from .. import storage
from ..contracts import EvaluationTrace
from ..config import settings
from .evidence_bundle import describe_task, paired_history
from .scientific_retrieval import retrieve_for_task
from .reranker import rerank
from .research_queries import filter_snapshot, plan_research_queries, snapshot_from_rows, retrieve_query_candidates
from .research_queries import snapshot_is_current


def _scope(group_id: int, selected_doc_ids, allowed_doc_ids):
    def checked_ids(values, name):
        if values is None:
            return None
        if not isinstance(values, (list, tuple)) or any(type(value) is not int or value <= 0 for value in values):
            raise ValueError(f"{name} must contain positive document IDs")
        return list(dict.fromkeys(values))
    selected = checked_ids(selected_doc_ids, "selected_doc_ids")
    allowed = checked_ids(allowed_doc_ids, "allowed_doc_ids")
    if selected is not None and allowed is not None and not set(selected) <= set(allowed):
        raise ValueError("selected_doc_ids must be a subset of allowed_doc_ids")
    # No explicit boundary retains the existing all-ready library behavior.
    if allowed is None:
        allowed = selected if selected is not None else [row.id for row in storage.list_documents(group_id) if row.status == "ready"]
    rows = {}
    for doc_id in allowed:
        row = storage.get_document(doc_id)
        if row is None or row.group_id != group_id or row.status != "ready":
            raise ValueError("文献范围包含未就绪或不属于当前文献库的记录")
        rows[doc_id] = row
    effective = selected if selected is not None else allowed
    required = selected if selected is not None else []
    titles = [rows[doc_id].paper_title or rows[doc_id].filename for doc_id in (selected if selected is not None else allowed)]
    return {"allowed": allowed, "selected": selected, "effective": effective, "required": required,
            "titles": titles, "target_resolution": "selected_sources" if required else "unresolved_allowed_scope",
            "snapshot": snapshot_from_rows(group_id, effective, rows)}


def _channel_observation(documents: list[Document]) -> tuple[dict, dict]:
    """Reconstruct observed direct candidates, never infer missing historical channels."""
    names = ("bm25", "vector")
    direct = [doc for doc in documents if not isinstance(doc.metadata.get("retrieval_origin"), dict)
              or doc.metadata["retrieval_origin"].get("direct_rank") is not None]
    info = {"available": False, "rank_scope": "authorized_inputs_to_fusion"}
    if not direct:
        return {**info, "reason": "no_observed_direct_hits"}, {}
    for document in direct:
        channels = document.metadata.get("retrieval_channels")
        fusion = document.metadata.get("retrieval_fusion")
        if (not isinstance(channels, dict) or not channels or not set(channels) <= set(names)
                or not isinstance(fusion, dict) or fusion.get("channels") != list(names)
                or any(not isinstance(value, dict) or type(value.get("rank")) is not int
                       or value["rank"] < 1 for value in channels.values())):
            return {**info, "reason": "incomplete_channel_metadata"}, {}
    stages = {}
    for name in names:
        hits = [doc for doc in direct if name in doc.metadata["retrieval_channels"]]
        hits.sort(key=lambda doc: doc.metadata["retrieval_channels"][name]["rank"])
        stages[name] = {"documents": hits, "rank_scope": info["rank_scope"], "scores_collected": False,
                        "document_representation": "fused_identity_representative"}
    return {**info, "available": True}, stages


def run_research_rerank(query: str, documents: list[Document]) -> list[Document]:
    """One original-query comparison across the complete candidate pool."""
    return rerank(query, documents, top_n=len(documents)) if documents else []


def _retrieve_candidates(query, group_id, scope, task_type, query_plan):
    trace: EvaluationTrace = {"pipeline": "research", "stages": {}, "timings": {},
        "scope": {"allowed_doc_ids": scope["allowed"], "selected_doc_ids": scope["selected"],
                  "retrieval_doc_ids": scope["effective"], "snapshot": list(scope["snapshot"])}}
    started = time.perf_counter()
    if settings.research_multi_query_enabled:
        from .scientific_retrieval import expand_research_candidates
        recalled = retrieve_query_candidates(group_id, query, scope["snapshot"], query_plan=query_plan)
        direct = recalled["documents"]
        raw = [] if recalled["retrieval_failed"] else expand_research_candidates(
            group_id, direct, selected_doc_ids=tuple(scope["effective"]), task_type=task_type,
            document_filter=lambda docs: filter_snapshot(docs, scope["snapshot"]))
        if not snapshot_is_current(scope["snapshot"]):
            raw = []
            recalled["retrieval_failed"] = recalled["degraded"] = True
            if not recalled.get("scope_changed"):
                recalled["warnings"].append("文献版本或可用范围在扩展期间发生变化，请重新发起请求。")
            recalled["scope_changed"] = True
        trace["queries"] = recalled["query_records"]
        trace["stages"]["direct_merged"] = {"documents": list(direct), "merge_key": "source_version_chunk_identity",
                                               "score_policy": "no_cross_query_score_addition"}
        trace["stages"]["expanded"] = {"documents": list(raw)}
        trace["retrieval_counts"] = {"queries": len(recalled["queries"]),
            "query_hits": sum(len(record["documents"]) for record in recalled["query_records"]),
            "direct_merged": len(direct), "expanded": len(raw)}
        # Channel ranks from different queries are never flattened into one leaderboard.
        channel_groups = {}
        for record in trace["queries"]:
            info, stages = _channel_observation(record["documents"])
            channel_groups[record["query_id"]] = {**info, "stages": stages}
        trace["retrieval_channels"] = {"available": any(v["available"] for v in channel_groups.values()),
            "rank_scope": "per_query_authorized_inputs_to_fusion", "queries": channel_groups}
    else:
        raw = retrieve_for_task(group_id, query, selected_doc_ids=tuple(scope["effective"]), task_type=task_type)
        recalled = {"queries": [query], "warnings": [], "degraded": False, "retrieval_failed": False}
        trace["queries"] = [{"query_id": "q1", "text": query, "status": "ok" if raw else "empty", "documents": list(raw)}]
        info, stages = _channel_observation(raw)
        trace["retrieval_channels"] = info
        trace["stages"].update(stages)
    trace["timings"]["retrieval"] = time.perf_counter() - started
    trace["stages"]["raw"] = {"documents": list(raw)}
    return {"documents": raw, "raw_documents": raw, "trace": trace,
            "queries": recalled["queries"], "warnings": recalled["warnings"], "degraded": recalled["degraded"],
            "retrieval_failed": recalled["retrieval_failed"], "retrieval_doc_ids": scope["effective"],
            "scope_changed": recalled.get("scope_changed", False),
            "scope_snapshot": scope["snapshot"] if settings.research_multi_query_enabled else None}


def retrieve_research_candidates(query: str, group_id: int, *, selected_doc_ids: list[int] | None = None,
                                 allowed_doc_ids: list[int] | None = None, task_type: str = "method",
                                 query_plan: dict | None = None) -> dict:
    """HTTP/evaluation shared recall only: scope, optional plan, hybrid, merge, expand.

The caller must rerank once against query and check retrieval_failed before generation.
An explicit empty selection stays empty. query_plan is an internal injection seam,
not a user-facing permission or source-selection mechanism.
"""
    return _retrieve_candidates(query, group_id, _scope(group_id, selected_doc_ids, allowed_doc_ids), task_type, query_plan)


def retrieve_research_evidence(question: str, group_id: int, *, selected_doc_ids: list[int] | None = None,
                               allowed_doc_ids: list[int] | None = None, history=None) -> dict:
    history = paired_history(history or [])
    scope = _scope(group_id, selected_doc_ids, allowed_doc_ids)
    description = describe_task(question, history, scope["titles"],
        source_ids=[f"library:{group_id}:{doc_id}" for doc_id in scope["allowed"]],
        required_source_ids=[f"library:{group_id}:{doc_id}" for doc_id in scope["required"]],
        target_resolution=scope["target_resolution"])
    trace: EvaluationTrace = {"stages": {}, "timings": {}, "pipeline": "research",
        "scope": {"allowed_doc_ids": scope["allowed"], "selected_doc_ids": scope["selected"],
                  "retrieval_doc_ids": scope["effective"], "required_source_ids": description.required_source_ids,
                  "target_resolution": description.target_resolution}}
    base = {"history": history, "task_description": description.to_dict(), "trace": trace,
            "retrieval_doc_ids": scope["effective"]}
    if description.needs_clarification:
        return {**base, "documents": [], "raw_documents": []}
    recalled = _retrieve_candidates(description.resolved_query, group_id, scope, description.task_type, None)
    raw = recalled["documents"]
    trace.update({key: value for key, value in recalled["trace"].items() if key != "scope"})
    trace["scope"]["snapshot"] = recalled["trace"]["scope"]["snapshot"]
    started = time.perf_counter()
    ranked = run_research_rerank(description.resolved_query, raw)
    if recalled["scope_snapshot"] is not None:
        ranked = filter_snapshot(ranked, recalled["scope_snapshot"])
        if not snapshot_is_current(recalled["scope_snapshot"]):
            ranked = []
            recalled.update(retrieval_failed=True, degraded=True, scope_changed=True)
            recalled["warnings"].append("文献版本或可用范围在重排期间发生变化，请重新发起请求。")
    trace["timings"]["rerank"] = time.perf_counter() - started
    trace["stages"]["reranked"] = {"documents": list(ranked)}
    return {**recalled, **base, "documents": ranked, "raw_documents": raw}


def answer_research_question(question: str, group_id: int, *, selected_doc_ids: list[int] | None = None,
                             allowed_doc_ids: list[int] | None = None, history=None, user_id=None) -> dict:
    from .graph import answer_from_evidence
    retrieved = retrieve_research_evidence(question, group_id, selected_doc_ids=selected_doc_ids,
                                         allowed_doc_ids=allowed_doc_ids, history=history)
    description = retrieved["task_description"]
    if description["needs_clarification"]:
        return {"answer": "请明确这次问题对应哪篇论文或哪个研究对象；当前指代不足以确定检索范围。",
                "status": "clarification", "documents": [], "queries": [question], "service_failed": False,
                "task_description": description, "trace": retrieved["trace"], "warnings": []}
    if retrieved.get("retrieval_failed"):
        return {"answer": ("检索期间文献版本或可用范围发生变化，未继续生成答案。请重新提问。" if retrieved.get("scope_changed")
                            else "检索服务暂不可用，未据此判断文献是否能够回答。请稍后重试。"),
                "status": "system_error", "documents": [], "queries": retrieved["queries"],
                "service_failed": True, "degraded": True, "task_description": description,
                "trace": retrieved["trace"], "warnings": retrieved["warnings"]}
    started = time.perf_counter()
    from .retrieval import filter_ready_documents
    def evidence_guard(documents):
        ready = filter_ready_documents(group_id, documents, retrieved["retrieval_doc_ids"])
        return filter_snapshot(ready, retrieved["scope_snapshot"]) if retrieved.get("scope_snapshot") is not None else ready
    result = answer_from_evidence(question, retrieved["documents"], history=retrieved["history"], user_id=user_id, task_description=description,
        evidence_guard=evidence_guard, apply_research_enhancements=True)
    trace = result.setdefault("trace", {})
    for key, value in retrieved["trace"]["stages"].items():
        trace.setdefault("stages", {}).setdefault(key, value)
    for key, value in retrieved["trace"]["timings"].items():
        trace.setdefault("timings", {}).setdefault(key, value)
    trace["timings"]["answer_total"] = time.perf_counter() - started
    trace["pipeline"] = "research"
    trace["scope"] = retrieved["trace"]["scope"]
    trace["retrieval_channels"] = retrieved["trace"]["retrieval_channels"]
    trace["queries"] = retrieved["trace"]["queries"]
    if "retrieval_counts" in retrieved["trace"]:
        trace["retrieval_counts"] = retrieved["trace"]["retrieval_counts"]
    result["queries"] = retrieved["queries"]
    result["warnings"] = [*retrieved["warnings"], *result.get("warnings", [])]
    result["degraded"] = bool(retrieved.get("degraded") or result.get("degraded"))
    return result
