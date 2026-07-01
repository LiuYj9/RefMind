"""混合检索：BM25 关键词 + Chroma 向量，等权融合。

每个文献库一个 EnsembleRetriever。BM25 是内存索引，文档增删后需重建，
所以检索器带缓存，并在入库/删除时显式失效。中文用 jieba 分词。
"""

from __future__ import annotations

import re

import jieba
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ..config import settings
from .document_processor import get_vectorstore, load_group_documents

_RETRIEVER_CACHE: dict[int, EnsembleRetriever] = {}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _preprocess(text: str) -> list[str]:
    """中英文混合分词：jieba 切中文，英文/数字整体保留并转小写。"""
    tokens: list[str] = []
    for token in jieba.cut(text):
        token = token.strip()
        if not token:
            continue
        if _TOKEN_RE.fullmatch(token):
            tokens.append(token.lower())
        elif not token.isspace():
            tokens.append(token)
    return tokens


def build_retriever(
    group_id: int, documents: list[Document] | None = None, k: int | None = None
):
    """构建某库的混合检索器；库内无内容时返回 None。

    k 为召回候选数（默认 recall_top_k），召回后一般再经重排精排。
    """
    if documents is None:
        documents = load_group_documents(group_id)

    if not documents:
        return None

    k = k or settings.recall_top_k

    bm25 = BM25Retriever.from_documents(documents, preprocess_func=_preprocess)
    bm25.k = k

    vector_retriever = get_vectorstore(group_id).as_retriever(
        search_kwargs={"k": k}
    )

    return EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.5, 0.5],
    )


def get_retriever(group_id: int):
    if group_id not in _RETRIEVER_CACHE:
        retriever = build_retriever(group_id)
        if retriever is None:
            return None
        _RETRIEVER_CACHE[group_id] = retriever
    return _RETRIEVER_CACHE[group_id]


def invalidate_retriever(group_id: int) -> None:
    """失效某库缓存，下次使用时重建 BM25。"""
    _RETRIEVER_CACHE.pop(group_id, None)


def reset_retrievers() -> None:
    _RETRIEVER_CACHE.clear()
