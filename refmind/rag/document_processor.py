"""文档处理：将解析后的文档分块、向量化并写入按组隔离的 Chroma 集合。

分块时保留页码元数据，便于回答时溯源引用。
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings
from ..llm import get_embedding_model


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """构造文本分割器。"""
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )


def parsed_to_documents(
    parsed: dict[str, Any], group_id: int, filename: str
) -> list[Document]:
    """将解析后的 PDF 切分为带元数据的 ``Document`` 分块。

    按页切分以便每个分块保留准确页码；若无页码信息则对全文整体切分。
    """
    splitter = _build_splitter()
    pages = parsed.get("pages") or []
    documents: list[Document] = []

    if pages:
        for page in pages:
            page_no = page.get("page", 1)
            for chunk in splitter.split_text(page.get("text", "")):
                if not chunk.strip():
                    continue
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "group_id": group_id,
                            "filename": filename,
                            "page": page_no,
                            "chunk_id": str(uuid.uuid4()),
                        },
                    )
                )
    else:
        full_text = parsed.get("markdown", "")
        for i, chunk in enumerate(splitter.split_text(full_text), start=1):
            if not chunk.strip():
                continue
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "group_id": group_id,
                        "filename": filename,
                        "page": i,
                        "chunk_id": str(uuid.uuid4()),
                    },
                )
            )

    return documents


def get_vectorstore(group_id: int):
    """返回某组的 Chroma 向量库（持久化、按组隔离）。"""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=f"group_{group_id}",
        embedding_function=get_embedding_model(),
        persist_directory=str(settings.group_chroma_dir(group_id)),
    )


def ingest_documents(group_id: int, documents: list[Document]) -> int:
    """将分块向量化并持久化到该组的 Chroma 集合，返回写入的分块数。

    langchain-chroma 会自动持久化，无需显式调用 persist()。
    """
    if not documents:
        return 0
    vectorstore = get_vectorstore(group_id)
    vectorstore.add_documents(documents)
    return len(documents)


def load_group_documents(group_id: int) -> list[Document]:
    """从 Chroma 重新加载某组的全部分块。

    用于（重）构建内存中的 BM25 关键词索引：组内文档变更后需重建。
    """
    vectorstore = get_vectorstore(group_id)
    raw = vectorstore.get(include=["documents", "metadatas"])
    contents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    documents: list[Document] = []
    for content, metadata in zip(contents, metadatas):
        documents.append(Document(page_content=content, metadata=metadata or {}))
    return documents


def delete_document_chunks(group_id: int, filename: str) -> None:
    """从某组中删除指定文件名对应的全部分块。"""
    vectorstore = get_vectorstore(group_id)
    vectorstore.delete(where={"filename": filename})
