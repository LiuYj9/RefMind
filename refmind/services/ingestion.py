"""上传入库编排：串联解析、处理、存储与检索重建。

Streamlit 前端只调用本模块中的函数。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from .. import storage
from ..config import settings
from ..llm import generate_summary
from ..parsing import parse_pdf, save_parsed
from ..rag import (
    delete_document_chunks,
    ingest_documents,
    invalidate_retriever,
    parsed_to_documents,
)

# 进度回调签名：(进度比例 0~1, 提示信息)
ProgressCb = Callable[[float, str], None]


def _noop(_p: float, _m: str) -> None:
    """默认空进度回调。"""


def ingest_pdf(
    group_id: int,
    source_path: str | Path,
    filename: str,
    progress: ProgressCb = _noop,
    make_summary: bool = True,
) -> storage.DocumentRow:
    """对单个 PDF 执行完整入库流程。

    解析 -> 分块 -> 向量化入库 ->（可选）摘要 -> 写入元数据，
    最后重建该组的混合检索器。
    """
    settings.ensure_dirs()
    source_path = Path(source_path)

    # 保留原始上传文件副本
    stored_pdf = settings.upload_dir / f"{group_id}_{filename}"
    if Path(source_path).resolve() != stored_pdf.resolve():
        shutil.copy(source_path, stored_pdf)

    doc_id = storage.create_document(
        group_id, filename, original_path=str(stored_pdf), status="parsing"
    )

    progress(0.1, f"正在解析 {filename} ...")
    parsed = parse_pdf(stored_pdf)

    parsed_path = settings.parsed_dir / f"{group_id}_{Path(filename).stem}.json"
    save_parsed(parsed, parsed_path)

    progress(0.4, "正在分块与向量化 ...")
    documents = parsed_to_documents(parsed, group_id, filename)
    num_chunks = ingest_documents(group_id, documents)
    storage.update_document(
        doc_id,
        parsed_json_path=str(parsed_path),
        num_chunks=num_chunks,
        status="indexing",
    )

    summary = ""
    if make_summary and documents:
        progress(0.75, "正在生成摘要 ...")
        try:
            summary = generate_summary(documents)
        except Exception as exc:  # noqa: BLE001
            summary = f"(摘要生成失败: {exc})"

    storage.update_document(doc_id, summary=summary, status="ready")

    progress(0.95, "正在重建混合检索索引 ...")
    invalidate_retriever(group_id)

    progress(1.0, f"{filename} 处理完成。")
    return storage.get_document(doc_id)


def remove_document(doc_id: int) -> None:
    """删除文档的分块、文件与元数据，并重建检索器。"""
    doc = storage.get_document(doc_id)
    if doc is None:
        return
    try:
        delete_document_chunks(doc.group_id, doc.filename)
    except Exception:  # noqa: BLE001 - 向量清理尽力而为
        pass
    for path in (doc.original_path, doc.parsed_json_path):
        if path:
            Path(path).unlink(missing_ok=True)
    storage.delete_document(doc_id)
    invalidate_retriever(doc.group_id)


def remove_group(group_id: int) -> None:
    """删除用户组及其文档、会话与向量库目录。"""
    for doc in storage.list_documents(group_id):
        for path in (doc.original_path, doc.parsed_json_path):
            if path:
                Path(path).unlink(missing_ok=True)
    chroma_dir = settings.chroma_persist_dir / f"group_{group_id}"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
    invalidate_retriever(group_id)
    storage.delete_group(group_id)
