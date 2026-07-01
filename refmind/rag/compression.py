"""上下文压缩。

精排之后，把要送进 Prompt 的内容进一步瘦身：先去掉近似重复的分块，再在句子
粒度上剔除与问题无关的句子，最后按字数预算截断。目的是在保留关键证据的前提下
降低冗余与 token 消耗。嵌入不可用时退化为仅按字数预算截断。
"""

from __future__ import annotations

import re

import numpy as np
from langchain_core.documents import Document

from ..config import settings
from ..llm import get_embedding_model

# 中文按标点断句；英文按 "句末标点 + 空格 + 大写" 断句，避免误切小数如 3.2
_SENT_SPLIT = re.compile(r"(?<=[。！？!?])\s*|(?<=[.?!])\s+(?=[A-Z])|\n+")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def _apply_char_budget(docs: list[Document]) -> list[Document]:
    """按字数预算保留靠前的分块，超出预算处截断。"""
    budget = settings.context_max_chars
    if budget <= 0:
        return docs
    out: list[Document] = []
    used = 0
    for doc in docs:
        if used >= budget:
            break
        content = doc.page_content
        if used + len(content) > budget:
            content = content[: budget - used]
        if content == doc.page_content:
            out.append(doc)
        else:
            out.append(Document(page_content=content, metadata=doc.metadata))
        used += len(content)
    return out


def _dedup(
    docs: list[Document], vecs: list[np.ndarray]
) -> tuple[list[Document], list[np.ndarray]]:
    """丢弃与已保留分块高度相似的重复块（保留排名靠前者）。"""
    kept_docs: list[Document] = []
    kept_vecs: list[np.ndarray] = []
    for doc, vec in zip(docs, vecs):
        if any(
            _cosine(vec, kv) >= settings.redundancy_threshold for kv in kept_vecs
        ):
            continue
        kept_docs.append(doc)
        kept_vecs.append(vec)
    return kept_docs, kept_vecs


def _compress_one(
    doc: Document, query_vec: np.ndarray, embeddings
) -> Document:
    """句级压缩单个分块：保留与问题相关的句子（至少保留最相关的一句）。"""
    sentences = _split_sentences(doc.page_content)
    if len(sentences) <= 1:
        return doc
    try:
        sent_vecs = embeddings.embed_documents(sentences)
    except Exception:  # noqa: BLE001
        return doc

    scores = [_cosine(query_vec, np.array(v)) for v in sent_vecs]
    threshold = settings.sentence_relevance_threshold
    keep_idx = [i for i, s in enumerate(scores) if s >= threshold]
    if not keep_idx:
        keep_idx = [int(np.argmax(scores))]

    kept = [sentences[i] for i in sorted(keep_idx)]
    if len(kept) == len(sentences):
        return doc

    meta = dict(doc.metadata or {})
    meta["compressed"] = True
    meta["char_count"] = sum(len(s) for s in kept)
    return Document(page_content=" ".join(kept), metadata=meta)


def compress_context(query: str, docs: list[Document]) -> list[Document]:
    """去重 + 句级过滤 + 字数预算，返回压缩后的上下文分块。"""
    if not docs:
        return []
    if not settings.context_compression_enabled:
        return _apply_char_budget(docs)

    try:
        embeddings = get_embedding_model()
        query_vec = np.array(embeddings.embed_query(query))
        chunk_vecs = [
            np.array(v)
            for v in embeddings.embed_documents([d.page_content for d in docs])
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"[compression] 嵌入不可用（{exc}），仅按字数预算截断。")
        return _apply_char_budget(docs)

    kept_docs, _ = _dedup(docs, chunk_vecs)
    compressed = [_compress_one(d, query_vec, embeddings) for d in kept_docs]
    return _apply_char_budget(compressed)
