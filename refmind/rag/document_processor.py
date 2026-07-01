"""文档分块、向量化并写入按库隔离的 Chroma 集合。

每个分块都带上溯源与治理所需的 metadata：来源文件、文档 id、页码、章节、
版本、权限（按文献库隔离）、分块序号与字数，方便检索时过滤与引用。
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from ..llm import get_embedding_model

# markdown 标题，或 "1"、"2.3" 这类编号标题
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_NUM_HEADING = re.compile(r"^\d+(?:\.\d+){0,3}\.?\s+\S.{0,60}$")


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


def ingest_documents(group_id: int, documents: list[Document]) -> int:
    """向量化并写入该库的 Chroma 集合，返回写入分块数。"""
    if not documents:
        return 0
    vectorstore = get_vectorstore(group_id)
    vectorstore.add_documents(documents)
    return len(documents)


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
