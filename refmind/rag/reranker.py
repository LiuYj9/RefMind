"""候选分块精排。

混合召回拿到一批候选后，用重排模型对 (query, chunk) 做相关性打分并重新排序，
只保留最相关的前若干条。优先调用 DashScope 的 gte-rerank 系列；若未安装
dashscope 或调用失败，则回退到"嵌入余弦相似度"排序，保证流程始终可用。
"""

from __future__ import annotations

import importlib.util
import logging
import math
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document

from ..config import settings
from ..llm import get_embedding_model

LOGGER = logging.getLogger(__name__)


def rerank_capacity(query: str, docs: list[Document]) -> dict:
    """Conservative UTF-8 upper bound, explicitly not the model tokenizer.

    Unknown models retain server-side validation. A known over-budget request
    falls back as one batch; splitting would make score scales incomparable.
    """
    limits = {"gte-rerank-v2": (500, 4000, 30000),
              "qwen3-rerank": (500, 4000, 120000)}
    model = settings.rerank_model
    result = {"model": model, "estimator": "utf8_bytes_upper_bound", "documents": len(docs)}
    if model not in limits:
        return {**result, "status": "server_validation", "reason": None}
    max_docs, max_item, max_request = limits[model]
    query_cost = len(query.encode("utf-8"))
    doc_costs = [len(d.page_content.encode("utf-8")) for d in docs]
    total = query_cost * len(docs) + sum(doc_costs)
    reason = ("document_count_limit" if len(docs) > max_docs else
              "item_capacity_upper_bound" if max([query_cost, *doc_costs], default=0) > max_item else
              "request_capacity_upper_bound" if total > max_request else None)
    return {**result, "status": "fallback" if reason else "within_upper_bound",
            "estimated_input": total, "reason": reason}


def _stable_document_key(document: Document) -> tuple[str, ...]:
    metadata = document.metadata or {}
    return (
        str(metadata.get("doc_id") or ""),
        str(metadata.get("version") or ""),
        str(metadata.get("chunk_index") or ""),
        str(metadata.get("chunk_id") or ""),
        str(metadata.get("filename") or metadata.get("source") or ""),
        " ".join(document.page_content.split()),
    )


@lru_cache(maxsize=1)
def dashscope_sdk_available() -> bool:
    """不触发导入，仅检查当前解释器是否安装 DashScope SDK。"""
    try:
        return importlib.util.find_spec("dashscope") is not None
    except (ImportError, ValueError):
        return False


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _dashscope_rerank(
    query: str, docs: list[Document]
) -> list[tuple[float, Document]]:
    import dashscope
    from http import HTTPStatus

    resp = dashscope.TextReRank.call(
        model=settings.rerank_model,
        query=query,
        documents=[d.page_content for d in docs],
        top_n=len(docs),
        return_documents=False,
        api_key=settings.dashscope_api_key,
    )
    if getattr(resp, "status_code", HTTPStatus.OK) != HTTPStatus.OK:
        raise RuntimeError(getattr(resp, "message", "rerank 调用失败"))

    results = resp.output["results"]
    scored: list[tuple[float, Document]] = []
    seen = set()
    for item in results:
        idx = item["index"] if isinstance(item, dict) else item.index
        score = (
            item["relevance_score"]
            if isinstance(item, dict)
            else item.relevance_score
        )
        if not isinstance(idx, int) or isinstance(idx, bool) or idx in seen or not 0 <= idx < len(docs) or not math.isfinite(float(score)):
            raise ValueError("invalid_rerank_result")
        seen.add(idx)
        scored.append((float(score), docs[idx]))
    if len(seen) != len(docs):
        raise ValueError("incomplete_rerank_results")
    scored.sort(key=lambda item: (-item[0], _stable_document_key(item[1])))
    return scored


def _embedding_rerank(
    query: str, docs: list[Document]
) -> list[tuple[float, Document]]:
    embeddings = get_embedding_model()
    query_vec = np.array(embeddings.embed_query(query))
    doc_vecs = embeddings.embed_documents([d.page_content for d in docs])
    if len(doc_vecs) != len(docs):
        raise ValueError("incomplete_embedding_results")
    scored = [
        (_cosine(query_vec, np.array(v)), d) for v, d in zip(doc_vecs, docs)
    ]
    scored.sort(key=lambda item: (-item[0], _stable_document_key(item[1])))
    return scored


def rerank(
    query: str, docs: list[Document], top_n: int | None = None
) -> list[Document]:
    """对候选分块重排并返回前 top_n 条，重排分记入 metadata["rerank_score"]。"""
    if not docs:
        return []
    top_n = top_n or settings.rerank_top_n
    capacity = rerank_capacity(query, docs)

    def labelled(
        document: Document,
        backend: str,
        reason: str | None = None,
        *,
        score: float | None = None,
    ) -> Document:
        metadata = {
            key: value
            for key, value in document.metadata.items()
            if key not in {"rerank_score", "rerank_backend", "rerank_degraded", "rerank_reason"}
        }
        metadata["rerank_backend"] = backend
        metadata["rerank_degraded"] = backend in {"embedding_similarity", "recall_order"}
        metadata["rerank_capacity"] = capacity
        if reason:
            metadata["rerank_reason"] = reason
        if score is not None:
            metadata["rerank_score"] = round(score, 4)
        return Document(page_content=document.page_content, metadata=metadata)

    if not settings.rerank_enabled:
        return [labelled(document, "disabled") for document in docs[:top_n]]

    scored: list[tuple[float, Document]] | None = None
    backend, reason = "cross_encoder", "cross_encoder_unavailable"
    if (
        settings.dashscope_api_key
        and settings.rerank_model
        and dashscope_sdk_available()
    ):
        try:
            if capacity.get("reason"):
                raise ValueError(str(capacity["reason"]))
            scored = _dashscope_rerank(query, docs)
        except Exception as exc:  # noqa: BLE001
            reason = str(capacity.get("reason") or type(exc).__name__)
            LOGGER.warning(
                "DashScope 重排调用失败，回退嵌入相似度重排：%s", exc
            )

    if scored is None:
        try:
            scored = _embedding_rerank(query, docs)
            backend = "embedding_similarity"
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("嵌入重排失败，按召回顺序返回：%s", exc)
            return [
                labelled(document, "recall_order", type(exc).__name__)
                for document in docs[:top_n]
            ]

    return [
        labelled(document, backend, reason if backend == "embedding_similarity" else None, score=score)
        for score, document in scored[:top_n]
    ]
