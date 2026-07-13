"""上传入库编排：解析、分块入库、摘要、写元数据、重建检索器。"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from .. import storage
from ..config import settings
from ..llm import generate_summary
from ..parsing import parse_pdf, save_parsed
from ..plugins import CoreHook, get_plugin_manager
from ..rag import (
    get_vectorstore,
    ingest_documents,
    invalidate_retriever,
    parsed_to_documents,
)

ProgressCb = Callable[[float, str], None]


def _noop(_p: float, _m: str) -> None:
    pass


def _validate_filename(filename: str) -> str:
    """只接受单个 PDF 文件名，避免路径穿越或意外写到子目录。"""
    safe_name = Path(filename).name
    if (
        not safe_name
        or safe_name != filename
        or Path(safe_name).suffix.lower() != ".pdf"
    ):
        raise ValueError(f"无效的 PDF 文件名：{filename!r}")
    return safe_name


def _install_source_pdf(source: Path, destination: Path) -> tuple[bool, Path | None]:
    """原子安装上传文件，返回（是否复制，旧文件备份）。"""
    if not source.is_file():
        raise FileNotFoundError(f"未找到待入库 PDF：{source}")
    if source.resolve() == destination.resolve():
        return False, None

    destination.parent.mkdir(parents=True, exist_ok=True)
    staged_path: Path | None = None
    backup_path: Path | None = None
    try:
        # 先复制到同分区临时文件，os.replace 才能保证目标不会半写入。
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".uploading",
            delete=False,
        ) as handle:
            staged_path = Path(handle.name)
        shutil.copy2(source, staged_path)

        if destination.exists():
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".backup",
                delete=False,
            ) as handle:
                backup_path = Path(handle.name)
            backup_path.unlink(missing_ok=True)
            os.replace(destination, backup_path)

        os.replace(staged_path, destination)
        staged_path = None
        return True, backup_path
    except Exception:
        if backup_path is not None and backup_path.exists():
            os.replace(backup_path, destination)
            backup_path = None
        raise
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)


def _delete_chunks_by_doc_id(group_id: int, doc_id: int) -> None:
    """仅删除本次入库的块，不会误删同名文档的旧向量。"""
    get_vectorstore(group_id).delete(where={"doc_id": doc_id})


def _safe_unlink(path: Path | None) -> None:
    """回滚阶段的文件清理不应覆盖最初的入库异常。"""
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _run_parse_hooks(
    stored_pdf: Path,
    *,
    group_id: int,
    doc_id: int,
    filename: str,
) -> dict:
    """执行解析前后 hook，并在插件返回非法类型时保留核心解析结果。"""
    manager = get_plugin_manager()
    metadata = {"group_id": group_id, "doc_id": doc_id, "filename": filename}
    source_candidate = manager.run_hook(
        CoreHook.BEFORE_PARSE, stored_pdf, metadata=metadata
    ).value
    source = Path(source_candidate) if isinstance(source_candidate, (str, Path)) else stored_pdf
    if not source.is_file():
        source = stored_pdf

    parsed = parse_pdf(source)
    parsed_candidate = manager.run_hook(
        CoreHook.AFTER_PARSE, parsed, metadata=metadata
    ).value
    return dict(parsed_candidate) if isinstance(parsed_candidate, Mapping) else parsed


def _run_before_ingest_hook(
    documents: list[Document],
    *,
    group_id: int,
    doc_id: int,
    filename: str,
    version: str,
) -> list[Document]:
    """允许插件变换分块，同时重申删除/权限/溯源所依赖的 metadata。"""
    candidate = get_plugin_manager().run_hook(
        CoreHook.BEFORE_INGEST,
        documents,
        metadata={"group_id": group_id, "doc_id": doc_id, "filename": filename},
    ).value
    if isinstance(candidate, (list, tuple)) and all(
        isinstance(item, Document) for item in candidate
    ):
        selected = list(candidate)
    else:
        selected = documents

    normalized: list[Document] = []
    for index, document in enumerate(selected):
        metadata = dict(document.metadata or {})
        # 这些字段是权限隔离和失败补偿的硬约束，插件不能删除或改写。
        metadata.update(
            {
                "group_id": group_id,
                "permission": f"group:{group_id}",
                "filename": filename,
                "source": filename,
                "doc_id": doc_id,
                "char_count": len(document.page_content),
            }
        )
        metadata.setdefault("version", version)
        metadata.setdefault("chunk_index", index)
        metadata.setdefault("chunk_id", str(uuid.uuid4()))
        normalized.append(
            Document(page_content=document.page_content, metadata=metadata)
        )
    return normalized


def _rollback_ingestion(
    *,
    group_id: int,
    doc_id: int,
    stored_pdf: Path,
    copied_source: bool,
    source_backup: Path | None,
    parsed_path: Path | None,
    indexing_attempted: bool,
) -> None:
    """对跨 SQLite、文件系统和 Chroma 的写入执行最佳努力补偿。"""
    if indexing_attempted:
        try:
            _delete_chunks_by_doc_id(group_id, doc_id)
        except Exception:  # noqa: BLE001
            # 向量尚未清掉时保留记录和源文件，之后可按 doc_id 再次清理。
            try:
                storage.update_document(doc_id, status="cleanup_failed")
            except Exception:  # noqa: BLE001
                pass
            try:
                invalidate_retriever(group_id)
            except Exception:  # noqa: BLE001
                pass
            return

    _safe_unlink(parsed_path)

    if copied_source:
        _safe_unlink(stored_pdf)
        if source_backup is not None and source_backup.exists():
            try:
                os.replace(source_backup, stored_pdf)
            except OSError:
                pass

    try:
        storage.delete_document(doc_id)
    except Exception:  # noqa: BLE001
        try:
            # 若数据库删除失败，至少避免记录长期停留在 parsing。
            storage.update_document(doc_id, status="failed")
        except Exception:  # noqa: BLE001
            pass
    try:
        invalidate_retriever(group_id)
    except Exception:  # noqa: BLE001
        pass


def ingest_pdf(
    group_id: int,
    source_path: str | Path,
    filename: str,
    progress: ProgressCb = _noop,
    make_summary: bool = True,
) -> storage.DocumentRow:
    """处理单个 PDF：解析→分块入库→摘要→写元数据，最后重建检索器。"""
    settings.ensure_dirs()
    source_path = Path(source_path)
    filename = _validate_filename(filename)

    doc_id = storage.create_document(
        group_id, filename, original_path=None, status="parsing"
    )
    # doc_id 进入文件名后，同名重传也不会让两条记录共享并互删源 PDF。
    stored_pdf = settings.upload_dir / f"{group_id}_{doc_id}_{filename}"

    copied_source = False
    source_backup: Path | None = None
    parsed_path: Path | None = None
    indexing_attempted = False
    try:
        storage.update_document(doc_id, original_path=str(stored_pdf))
        copied_source, source_backup = _install_source_pdf(source_path, stored_pdf)

        progress(0.1, f"正在解析 {filename} ...")
        parsed = _run_parse_hooks(
            stored_pdf,
            group_id=group_id,
            doc_id=doc_id,
            filename=filename,
        )

        # 将 doc_id 纳入文件名，避免同名重传覆盖其他记录的解析结果。
        parsed_path = (
            settings.parsed_dir / f"{group_id}_{doc_id}_{Path(filename).stem}.json"
        )
        save_parsed(parsed, parsed_path)

        progress(0.4, "正在分块与向量化 ...")
        version = time.strftime("%Y%m%d%H%M%S")
        documents = parsed_to_documents(
            parsed, group_id, filename, doc_id=doc_id, version=version
        )
        documents = _run_before_ingest_hook(
            documents,
            group_id=group_id,
            doc_id=doc_id,
            filename=filename,
            version=version,
        )
        if not documents:
            raise ValueError("解析结果未生成任何可入库分块。")

        # add_documents 在批次中途失败时仍可能已写入部分块，
        # 因此必须在调用前记录状态，异常时按 doc_id 精确回滚。
        indexing_attempted = True
        num_chunks = ingest_documents(group_id, documents)
        if num_chunks <= 0:
            raise RuntimeError("向量库未写入任何分块。")
        storage.update_document(
            doc_id,
            parsed_json_path=str(parsed_path),
            num_chunks=num_chunks,
            status="indexing",
        )

        summary = ""
        if make_summary:
            progress(0.75, "正在生成摘要 ...")
            try:
                summary = generate_summary(documents)
            except Exception as exc:  # noqa: BLE001
                # 摘要是可选增强，失败不应撤销已验证的正文索引。
                summary = f"(摘要生成失败: {exc})"

        progress(0.95, "正在重建混合检索索引 ...")
        invalidate_retriever(group_id)
        # ready 是最后的提交点：在此之前的任何异常都会触发补偿回滚。
        storage.update_document(doc_id, summary=summary, status="ready")

        result = storage.get_document(doc_id)
        if result is None:
            raise RuntimeError("入库完成后无法读回文档记录。")

        # 成功提交后旧文件备份已无需保留。
        _safe_unlink(source_backup)
        try:
            progress(1.0, f"{filename} 处理完成。")
        except Exception:  # noqa: BLE001
            # 进度 UI 失效不能撤销已完成提交的文档。
            pass
        try:
            plugin_result = get_plugin_manager().run_hook(
                CoreHook.AFTER_INGEST,
                result,
                metadata={
                    "group_id": group_id,
                    "doc_id": doc_id,
                    "filename": filename,
                },
            ).value
        except Exception:  # noqa: BLE001
            # ready 提交之后，插件框架自身的意外错误也不能撤销已完成文档。
            return result
        return plugin_result if isinstance(plugin_result, type(result)) else result
    except Exception:
        _rollback_ingestion(
            group_id=group_id,
            doc_id=doc_id,
            stored_pdf=stored_pdf,
            copied_source=copied_source,
            source_backup=source_backup,
            parsed_path=parsed_path,
            indexing_attempted=indexing_attempted,
        )
        raise


def remove_document(doc_id: int) -> None:
    """删除文档的分块、文件与元数据，并重建检索器。"""
    doc = storage.get_document(doc_id)
    if doc is None:
        return
    storage.update_document(doc_id, status="deleting")
    try:
        _delete_chunks_by_doc_id(doc.group_id, doc.id)
    except Exception as exc:  # noqa: BLE001
        # 不丢弃数据库线索，否则残留向量将无法再按 doc_id 清理。
        storage.update_document(doc_id, status="cleanup_failed")
        invalidate_retriever(doc.group_id)
        raise RuntimeError("向量清理失败，文档记录已保留以便重试。") from exc
    try:
        for path in (doc.original_path, doc.parsed_json_path):
            if path:
                Path(path).unlink(missing_ok=True)
        storage.delete_document(doc_id)
    except Exception as exc:  # noqa: BLE001
        # 删除是可重试状态机；保留 doc_id 让下次启动/人工操作继续收敛。
        storage.update_document(doc_id, status="cleanup_failed")
        invalidate_retriever(doc.group_id)
        raise RuntimeError("文档删除未完成，记录已保留以便重试。") from exc
    invalidate_retriever(doc.group_id)


def recover_incomplete_ingestions() -> dict[str, list[int]]:
    """启动时清理未提交/未删净记录，覆盖进程强杀留下的跨存储中间态。"""
    recovered: list[int] = []
    failed: list[int] = []
    for group in storage.list_groups():
        for document in storage.list_documents(group.id):
            if document.status == "ready":
                continue
            try:
                remove_document(document.id)
                recovered.append(document.id)
            except Exception:  # noqa: BLE001
                failed.append(document.id)
    return {"recovered": recovered, "failed": failed}


def remove_group(group_id: int) -> None:
    """删除用户组及其文档、会话与向量库目录。"""
    # 逐文档走可恢复删除状态机；任何失败都会保留组和剩余记录。
    for doc in storage.list_documents(group_id):
        remove_document(doc.id)

    invalidate_retriever(group_id)
    chroma_dir = settings.chroma_persist_dir / f"group_{group_id}"
    if chroma_dir.exists():
        try:
            shutil.rmtree(chroma_dir)
        except OSError as exc:
            raise RuntimeError("向量库目录删除失败，文献库记录已保留。") from exc

    storage.delete_group(group_id)
