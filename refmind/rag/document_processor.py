"""文档分块、向量化并写入按库隔离的 Chroma 集合。

每个分块都带上溯源与治理所需的 metadata：来源文件、文档 id、页码、章节、
版本、权限（按文献库隔离）、分块序号与字数，方便检索时过滤与引用。
"""

from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, RLock
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from ..llm import get_embedding_model
from .layout_chunker import layout_chunks, scalar_metadata

# markdown 标题，或 "1"、"2.3" 这类编号标题
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_NUM_HEADING = re.compile(r"^\d+(?:\.\d+){0,3}\.?\s+\S.{0,60}$")

_EMBEDDING_CONDITION = Condition(RLock())
_ACTIVE_EMBEDDING_BATCHES = 0
_EMBEDDING_EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="refmind-embedding",
)


@dataclass(frozen=True)
class PreparedVectorBatch:
    """已完成远程 Embedding、等待写入 Chroma 的预计算批次。"""

    ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, Any]]
    embeddings: list[list[float]]


@contextmanager
def _embedding_batch_slot():
    """跨文档限制远程 Embedding 批次并发，避免突发请求击穿供应商限流。"""
    global _ACTIVE_EMBEDDING_BATCHES
    limit = min(8, max(1, int(settings.embedding_max_parallel_batches)))
    with _EMBEDDING_CONDITION:
        while _ACTIVE_EMBEDDING_BATCHES >= limit:
            _EMBEDDING_CONDITION.wait()
        _ACTIVE_EMBEDDING_BATCHES += 1
    try:
        yield
    finally:
        with _EMBEDDING_CONDITION:
            _ACTIVE_EMBEDDING_BATCHES -= 1
            _EMBEDDING_CONDITION.notify_all()


def _build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )


def _detect_section(text: str, fallback: str) -> str:
    """从一段文本里找出章节标题；找不到则沿用上一节标题。"""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        md = _MD_HEADING.match(line)
        if md:
            return md.group(1).strip()
        if _NUM_HEADING.match(line) and not line.endswith(("。", "，", ",")):
            return line
        # 首个短行且不像正文句子，视为标题
        if len(line) <= 40 and not line.endswith(("。", ".", "，", ",", "；", ";")):
            return line
        break
    return fallback


def parsed_to_documents(
    parsed: dict[str, Any],
    group_id: int,
    filename: str,
    doc_id: int | None = None,
    version: str | None = None,
) -> list[Document]:
    """把解析结果切成带完整 metadata 的分块，按页切分以保留页码与章节。"""
    splitter = _build_splitter()
    parser = parsed.get("parser", "")
    documents: list[Document] = []
    section = "正文"
    chunk_index = 0

    def _make(
        chunk: str,
        page_no: int,
        sec: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> Document:
        nonlocal chunk_index
        meta = {
            "group_id": group_id,
            "permission": f"group:{group_id}",
            "filename": filename,
            "source": filename,
            "doc_id": doc_id if doc_id is not None else -1,
            "version": version or "",
            "parser": parser,
            "page": page_no,
            "section": sec,
            "chunk_index": chunk_index,
            "char_count": len(chunk),
            "chunk_id": str(uuid.uuid4()),
        }
        if extra_meta:
            meta.update({k: v for k, v in extra_meta.items() if v is not None})
        chunk_index += 1
        return Document(page_content=chunk, metadata=meta)

    # MinerU 等版面解析器提供 blocks 时，优先按论文结构切分；pages 仅作为旧数据回退，
    # 避免正文/表格在两条路径中被重复向量化。
    blocks = parsed.get("blocks") or []
    if blocks:
        max_chars = max(settings.chunk_size, settings.layout_chunk_max_chars)
        for item in layout_chunks(
            blocks,
            target_chars=settings.chunk_size,
            max_chars=max_chars,
            overlap=settings.chunk_overlap,
        ):
            extra = {
                "content_type": item.content_type,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "section_path": item.section,
                "block_ids": scalar_metadata(item.block_ids),
                "bbox": scalar_metadata(item.bbox),
                "layout_parser": parser,
                "layout_confidence": parsed.get("layout_confidence", "high"),
                # 路径只指向本地 docstore；生成阶段会再次校验其必须位于该目录。
                "image_id": item.image_id or None,
                "image_path": item.image_path or None,
                "image_mime_type": item.image_mime_type or None,
            }
            documents.append(_make(item.text, item.page_start, item.section, extra))
        return documents

    pages = parsed.get("pages") or []
    if pages:
        for page in pages:
            page_no = page.get("page", 1)
            page_text = page.get("text", "")
            section = _detect_section(page_text, section)
            for chunk in splitter.split_text(page_text):
                if chunk.strip():
                    documents.append(_make(chunk, page_no, section))
    else:
        full_text = parsed.get("markdown", "")
        for i, chunk in enumerate(splitter.split_text(full_text), start=1):
            if not chunk.strip():
                continue
            section = _detect_section(chunk, section)
            documents.append(_make(chunk, i, section))

    for table in parsed.get("tables") or []:
        table_text = table.get("normalized_text") or table.get("raw_text") or ""
        if not table_text.strip():
            continue
        page_start = int(table.get("page_start") or table.get("page") or 1)
        page_end = int(table.get("page_end") or page_start)
        caption = str(table.get("caption") or "")
        table_id = str(table.get("table_id") or "")
        section_name = caption or "Table"
        prefix = (
            f"Table {table_id}\n"
            f"Pages: {page_start}-{page_end}\n"
            f"Caption: {caption}"
        ).strip()
        meta = {
            "content_type": "table",
            "table_id": table_id,
            "page_start": page_start,
            "page_end": page_end,
            "table_caption": caption,
            "continued_table": bool(table.get("continued")),
        }
        for chunk in splitter.split_text(table_text):
            if not chunk.strip():
                continue
            content = chunk if chunk.startswith("Table:") else f"{prefix}\n{chunk}"
            documents.append(_make(content, page_start, section_name, meta))

    return documents


def get_vectorstore(group_id: int):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=f"group_{group_id}",
        embedding_function=get_embedding_model(),
        persist_directory=str(settings.group_chroma_dir(group_id)),
    )


def prepare_ingestion_batch(documents: list[Document]) -> PreparedVectorBatch:
    """并行生成 Embedding；此阶段不接触 Chroma，可跨文档安全并发。"""
    if not documents:
        return PreparedVectorBatch([], [], [], [])

    texts = [document.page_content for document in documents]
    metadatas = [dict(document.metadata or {}) for document in documents]
    # 保持 Chroma.add_documents 的旧语义：record id 独立生成。chunk_id 由插件或
    # 上游 metadata 提供，可能被复制，不能用作 upsert 主键。
    ids = [str(uuid.uuid4()) for _ in metadatas]
    batch_size = max(1, int(settings.embedding_batch_size))
    batches = [
        texts[index : index + batch_size]
        for index in range(0, len(texts), batch_size)
    ]
    embedding_model = get_embedding_model()

    def _embed(batch: list[str]) -> list[list[float]]:
        with _embedding_batch_slot():
            vectors = [
                [float(value) for value in vector]
                for vector in embedding_model.embed_documents(batch)
            ]
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedding 批次返回数量异常：期望 {len(batch)}，实际 {len(vectors)}"
            )
        return vectors

    worker_count = min(
        len(batches),
        8,
        max(1, int(settings.embedding_max_parallel_batches)),
    )
    if worker_count == 1:
        embedded_batches = [_embed(batch) for batch in batches]
    else:
        futures = [
            _EMBEDDING_EXECUTOR.submit(_embed, batch)
            for batch in batches
        ]
        try:
            # 按提交顺序取结果，保证向量与原始 Document 的位置严格一致。
            embedded_batches = [future.result() for future in futures]
        except Exception:
            for future in futures:
                future.cancel()
            raise
    embeddings = [vector for batch in embedded_batches for vector in batch]
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding 返回数量异常：期望 {len(texts)}，实际 {len(embeddings)}"
        )
    return PreparedVectorBatch(ids, texts, metadatas, embeddings)


def ingest_documents(
    group_id: int,
    documents: list[Document],
    *,
    prepared: PreparedVectorBatch | None = None,
) -> int:
    """写入 Chroma；可传入锁外预计算的 Embedding，避免在写锁中等待网络。"""
    if not documents:
        return 0
    batch = prepared or prepare_ingestion_batch(documents)
    expected = len(documents)
    batch_lengths = {
        len(batch.ids),
        len(batch.texts),
        len(batch.metadatas),
        len(batch.embeddings),
    }
    if batch_lengths != {expected}:
        raise ValueError("预计算向量批次与文档数量不一致。")
    if len(set(batch.ids)) != expected:
        raise ValueError("预计算向量批次包含重复的 Chroma record id。")
    if any(
        text != document.page_content
        or metadata != dict(document.metadata or {})
        for document, text, metadata in zip(
            documents, batch.texts, batch.metadatas
        )
    ):
        raise ValueError("预计算向量批次与原始文档内容或 metadata 不一致。")
    dimensions = {len(vector) for vector in batch.embeddings}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("预计算向量维度为空或不一致。")
    # langchain-chroma 没有公开的 add_embeddings；直接通过 chromadb 的公开
    # PersistentClient/Collection API 提交预计算向量，避免依赖 LangChain 私有属性。
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.group_chroma_dir(group_id)))
    collection = client.get_or_create_collection(
        name=f"group_{group_id}", embedding_function=None
    )
    # Chroma 对单次提交数量有上限；超大文档按客户端公开上限分批提交，整个提交过程
    # 仍由 ingestion 层的同 group 互斥锁保护。
    get_limit = getattr(client, "get_max_batch_size", None)
    max_batch_size = (
        max(1, int(get_limit())) if callable(get_limit) else expected
    )
    for start in range(0, expected, max_batch_size):
        end = min(expected, start + max_batch_size)
        collection.upsert(
            ids=batch.ids[start:end],
            documents=batch.texts[start:end],
            metadatas=batch.metadatas[start:end],
            embeddings=batch.embeddings[start:end],
        )
    return len(batch.texts)


def load_group_documents(group_id: int) -> list[Document]:
    """从 Chroma 读回某库全部分块，用于重建 BM25 索引。"""
    vectorstore = get_vectorstore(group_id)
    raw = vectorstore.get(include=["documents", "metadatas"])
    contents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    documents: list[Document] = []
    for content, metadata in zip(contents, metadatas):
        documents.append(Document(page_content=content, metadata=metadata or {}))
    return documents


def delete_document_chunks(group_id: int, filename: str) -> None:
    vectorstore = get_vectorstore(group_id)
    vectorstore.delete(where={"filename": filename})
