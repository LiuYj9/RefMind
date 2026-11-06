"""混合检索：BM25 关键词 + Chroma 向量，等权融合。

每个文献库一个 EnsembleRetriever。BM25 是内存索引，文档增删后需重建，
所以检索器带缓存，并在入库/删除时显式失效。中文用 jieba 分词。
"""

from __future__ import annotations

import re
import hashlib
from typing import Any
import unicodedata
from threading import RLock

import jieba
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..config import settings
from .. import storage
from .document_processor import get_vectorstore, load_group_documents

_RETRIEVER_CACHE: dict[tuple[int, tuple[int, ...]], Any] = {}
_RETRIEVER_CACHE_LOCK = RLock()

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


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


def allowed_ready_doc_ids(group_id: int, selected_doc_ids: tuple[int, ...] | list[int] | None = None) -> tuple[int, ...]:
    """Return committed IDs in this group; [] is an empty scope, never all."""
    allowed = {row.id for row in storage.list_documents(group_id) if row.status == "ready"}
    if selected_doc_ids is not None:
        allowed.intersection_update(selected_doc_ids)
    return tuple(sorted(allowed))


def filter_ready_documents(group_id: int, documents: list[Document], selected_doc_ids: tuple[int, ...] | list[int] | None = None) -> list[Document]:
    if not documents:
        return []
    selected = None if selected_doc_ids is None else set(selected_doc_ids)
    ready = {row.id: row for row in storage.list_documents(group_id)
             if row.status == "ready" and (selected is None or row.id in selected)}
    output = []
    for document in documents:
        metadata = document.metadata or {}
        try:
            row = ready.get(int(metadata.get("doc_id", -1)))
            if row is None or int(metadata.get("group_id", group_id)) != group_id:
                continue
        except (ValueError, TypeError):
            continue
        # Legacy rows have no committed pair. For all other rows require both fields.
        if row.index_version is not None or row.file_sha256 is not None:
            if (not row.index_version or not row.file_sha256
                    or metadata.get("version") != row.index_version
                    or metadata.get("file_sha256") != row.file_sha256):
                continue
        output.append(document)
    return output


class _ScopedRetriever(BaseRetriever):
    source: Any
    group_id: int
    allowed_ids: tuple[int, ...]

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        documents = filter_ready_documents(self.group_id, list(self.source.invoke(query) or []), self.allowed_ids)
        output = []
        for document in documents:
            metadata = dict(document.metadata or {})
            metadata["_retrieval_identity"] = hashlib.sha256(repr(_stable_document_key(document)).encode()).hexdigest()
            output.append(Document(page_content=document.page_content, metadata=metadata))
        return output


class _ObservedEnsembleRetriever(EnsembleRetriever):
    """Observe authorized input ranks without replacing LangChain's RRF ordering."""

    channel_names: tuple[str, ...] = ("bm25", "vector")

    def weighted_reciprocal_rank(self, doc_lists: list[list[Document]]) -> list[Document]:
        if len(doc_lists) != len(self.channel_names):
            raise ValueError("channel names must match rank lists")
        ranked = super().weighted_reciprocal_rank(doc_lists)

        def key(document):
            return document.page_content if self.id_key is None else document.metadata[self.id_key]

        observed = {}
        for channel, documents in zip(self.channel_names, doc_lists):
            for rank, document in enumerate(documents, 1):
                # The first occurrence is the observable rank, even if a backend repeats a hit.
                observed.setdefault(key(document), {}).setdefault(channel, {"rank": rank})
        output = []
        for rank, document in enumerate(ranked, 1):
            metadata = {**document.metadata,
                "retrieval_channels": {name: dict(value) for name, value in observed[key(document)].items()},
                "retrieval_fusion": {"method": "weighted_rrf", "rank": rank, "c": self.c,
                    "weights": dict(zip(self.channel_names, self.weights)),
                    "channels": list(self.channel_names)}}
            output.append(Document(page_content=document.page_content, metadata=metadata))
        return output


def _preprocess(text: str) -> list[str]:
    """中英文混合分词：jieba 切中文，英文/数字整体保留并转小写。"""
    tokens: list[str] = []
    for token in jieba.cut(unicodedata.normalize("NFKC", text)):
        token = token.strip()
        if not token:
            continue
        if all(unicodedata.category(char).startswith("P") for char in token):
            continue
        if _TOKEN_RE.fullmatch(token):
            tokens.append(token.lower())
        elif not token.isspace():
            tokens.append(token)
    from .scientific_retrieval import scientific_aliases
    return [*tokens, *scientific_aliases(text)]


def build_retriever(
    group_id: int, documents: list[Document] | None = None, k: int | None = None,
    *, selected_doc_ids: tuple[int, ...] | None = None,
):
    """构建某库的混合检索器；库内无内容时返回 None。

    k 为召回候选数（默认 recall_top_k），召回后一般再经重排精排。
    """
    allowed = allowed_ready_doc_ids(group_id, selected_doc_ids)
    if not allowed:
        return None
    if documents is None:
        documents = load_group_documents(group_id, selected_doc_ids=allowed)
    else:
        documents = filter_ready_documents(group_id, documents, allowed)

    if not documents:
        return None

    documents = sorted(documents, key=_stable_document_key)
    k = k or settings.recall_top_k

    bm25 = BM25Retriever.from_documents(documents, preprocess_func=_preprocess)
    bm25.k = k

    vector_retriever = get_vectorstore(group_id).as_retriever(
        search_kwargs={"k": k, "filter": {"doc_id": {"$in": list(allowed)}}}
    )

    ensemble = _ObservedEnsembleRetriever(
        retrievers=[
            _ScopedRetriever(source=bm25, group_id=group_id, allowed_ids=allowed),
            _ScopedRetriever(source=vector_retriever, group_id=group_id, allowed_ids=allowed),
        ],
        weights=[0.5, 0.5],
        id_key="_retrieval_identity",
    )
    return _ScopedRetriever(source=ensemble, group_id=group_id, allowed_ids=allowed)


def get_retriever(group_id: int, *, selected_doc_ids: tuple[int, ...] | None = None):
    # Ready IDs belong to the cache key: a commit cannot leave a stale snapshot.
    with _RETRIEVER_CACHE_LOCK:
        allowed = allowed_ready_doc_ids(group_id, selected_doc_ids)
        key = (group_id, allowed)
        if key not in _RETRIEVER_CACHE:
            retriever = build_retriever(group_id) if selected_doc_ids is None else build_retriever(group_id, selected_doc_ids=allowed)
            if retriever is None:
                return None
            if len(_RETRIEVER_CACHE) >= 128:
                _RETRIEVER_CACHE.pop(next(iter(_RETRIEVER_CACHE)))
            _RETRIEVER_CACHE[key] = retriever
        return _RETRIEVER_CACHE[key]


def invalidate_retriever(group_id: int) -> None:
    """失效某库缓存，下次使用时重建 BM25。"""
    with _RETRIEVER_CACHE_LOCK:
        for key in list(_RETRIEVER_CACHE):
            if key[0] == group_id:
                _RETRIEVER_CACHE.pop(key, None)


def reset_retrievers() -> None:
    with _RETRIEVER_CACHE_LOCK:
        _RETRIEVER_CACHE.clear()
