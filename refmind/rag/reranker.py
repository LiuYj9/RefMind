"""候选分块精排。

混合召回拿到一批候选后，用重排模型对 (query, chunk) 做相关性打分并重新排序，
只保留最相关的前若干条。优先调用 DashScope 的 gte-rerank 系列；若未安装
dashscope 或调用失败，则回退到"嵌入余弦相似度"排序，保证流程始终可用。
"""

from __future__ import annotations

import numpy as np
from langchain_core.documents import Document

from ..config import settings
from ..llm import get_embedding_model


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
    for item in results:
        idx = item["index"] if isinstance(item, dict) else item.index
        score = (
            item["relevance_score"]
            if isinstance(item, dict)
            else item.relevance_score
        )
        scored.append((float(score), docs[idx]))
    return scored


def _embedding_rerank(
    query: str, docs: list[Document]
) -> list[tuple[float, Document]]:
    embeddings = get_embedding_model()
    query_vec = np.array(embeddings.embed_query(query))
    doc_vecs = embeddings.embed_documents([d.page_content for d in docs])
    scored = [
        (_cosine(query_vec, np.array(v)), d) for v, d in zip(doc_vecs, docs)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def rerank(
    query: str, docs: list[Document], top_n: int | None = None
) -> list[Document]:
    """对候选分块重排并返回前 top_n 条，重排分记入 metadata["rerank_score"]。"""
    if not docs:
        return []
    top_n = top_n or settings.rerank_top_n

    if not settings.rerank_enabled:
        return docs[:top_n]

    scored: list[tuple[float, Document]] | None = None
    if settings.dashscope_api_key and settings.rerank_model:
        try:
            scored = _dashscope_rerank(query, docs)
        except Exception as exc:  # noqa: BLE001
            print(f"[reranker] DashScope 重排不可用（{exc}），回退嵌入相似度重排。")

    if scored is None:
        try:
            scored = _embedding_rerank(query, docs)
        except Exception as exc:  # noqa: BLE001
            print(f"[reranker] 重排失败（{exc}），按召回顺序返回。")
            return docs[:top_n]

    out: list[Document] = []
    for score, doc in scored[:top_n]:
        meta = dict(doc.metadata or {})
        meta["rerank_score"] = round(score, 4)
        out.append(Document(page_content=doc.page_content, metadata=meta))
    return out
