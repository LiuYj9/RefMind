"""混合检索（BM25 关键词 + 稠密向量），两者等权融合。

每个用户组拥有独立的 ``EnsembleRetriever``，组合：
  * 基于该组内存分块列表构建的 BM25 检索器；
  * Chroma 向量检索器。

BM25 索引常驻内存，组内文档变更时必须重建，因此检索器会被缓存，
并在文档入库后显式失效。使用基于 jieba 的分词器以支持中文关键词检索。

LangChain 1.0 中 ``EnsembleRetriever`` 已迁移至 ``langchain_classic.retrievers``。
"""

from __future__ import annotations

import re

import jieba
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ..config import settings
from .document_processor import get_vectorstore, load_group_documents

# group_id -> EnsembleRetriever 缓存
_RETRIEVER_CACHE: dict[int, EnsembleRetriever] = {}

# 纯英文 / 数字 token 的匹配规则
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _preprocess(text: str) -> list[str]:
    """对中英文混合文本分词，供 BM25 使用。

    jieba 负责中文分词；英文单词 / 数字整体保留并转小写。
    """
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


def build_retriever(group_id: int, documents: list[Document] | None = None):
    """为某组构建全新的混合检索器。

    未提供 ``documents`` 时从 Chroma 加载全部分块。
    当该组尚无索引内容时返回 ``None``。
    """
    if documents is None:
        documents = load_group_documents(group_id)

    if not documents:
        return None

    k = settings.retrieval_top_k

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
    """返回某组缓存的混合检索器，必要时构建。"""
    if group_id not in _RETRIEVER_CACHE:
        retriever = build_retriever(group_id)
        if retriever is None:
            return None
        _RETRIEVER_CACHE[group_id] = retriever
    return _RETRIEVER_CACHE[group_id]


def invalidate_retriever(group_id: int) -> None:
    """使某组的缓存检索器失效，以便下次使用时重建 BM25 索引。"""
    _RETRIEVER_CACHE.pop(group_id, None)


def reset_retrievers() -> None:
    """清空全部检索器缓存（如嵌入模型变更后调用）。"""
    _RETRIEVER_CACHE.clear()
