"""上传入库编排：解析、分块入库、摘要、写元数据、重建检索器。"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Condition, RLock, Semaphore
from typing import Callable, Iterator, Sequence

from langchain_core.documents import Document

from .. import storage
from ..config import settings
from ..llm import generate_summary
from ..llm.image_understanding import summarize_image
from ..parsing.image_store import extract_pdf_figures
from ..parsing import parse_pdf, save_parsed
from ..plugins import CoreHook, get_plugin_manager
from ..rag import (
    get_vectorstore,
    ingest_documents,
    invalidate_retriever,
    parsed_to_documents,
    prepare_ingestion_batch,
)

ProgressCb = Callable[[float, str], None]
BatchProgressCb = Callable[[int, int, float, str], None]


@dataclass(frozen=True)
class PdfIngestionTask:
    """一篇待入库 PDF；旧版本只会在新版本成功后清理。"""

    source_path: Path
    filename: str
    previous_document_id: int | None = None


@dataclass
class PdfIngestionResult:
    """批量入库的单文档结果，结果顺序与输入任务一致。"""

    filename: str
    document: storage.DocumentRow | None = None
    error: str = ""
    cleanup_warning: str = ""
    stage_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.document is not None and not self.error


_VECTOR_MUTATION_LOCKS: dict[int, RLock] = {}
_VECTOR_MUTATION_LOCKS_GUARD = RLock()
_PDF_TASK_CONDITION = Condition(RLock())
_ACTIVE_PDF_TASKS = 0
_IMAGE_SUMMARY_CONDITION = Condition(RLock())
_ACTIVE_IMAGE_SUMMARIES = 0
_DOCUMENT_SUMMARY_CONDITION = Condition(RLock())
_ACTIVE_DOCUMENT_SUMMARIES = 0
_IMAGE_SUMMARY_EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="refmind-image",
)
_DOCUMENT_SUMMARY_EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="refmind-document-summary",
)
_GROUP_LIFECYCLE_CONDITION = Condition(RLock())
_ACTIVE_GROUP_OPERATIONS: dict[int, int] = {}
_DELETING_GROUPS: set[int] = set()


def _vector_mutation_lock(group_id: int) -> RLock:
    """同一文献库的 Chroma 变更串行化，不阻塞不同文献库。"""
    with _VECTOR_MUTATION_LOCKS_GUARD:
        return _VECTOR_MUTATION_LOCKS.setdefault(group_id, RLock())


@contextmanager
def _pdf_task_slot() -> Iterator[None]:
    """进程级解析闸门，确保多个 Streamlit 会话共享 MinerU 并发上限。"""
    global _ACTIVE_PDF_TASKS
    with _PDF_TASK_CONDITION:
        while _ACTIVE_PDF_TASKS >= min(
            8, max(1, int(settings.pdf_max_parallel_documents))
        ):
            _PDF_TASK_CONDITION.wait()
        _ACTIVE_PDF_TASKS += 1
    try:
        yield
    finally:
        with _PDF_TASK_CONDITION:
            _ACTIVE_PDF_TASKS -= 1
            _PDF_TASK_CONDITION.notify_all()


@contextmanager
def _image_summary_slot() -> Iterator[None]:
    """跨文档/会话限制多模态图片摘要并发请求。"""
    global _ACTIVE_IMAGE_SUMMARIES
    with _IMAGE_SUMMARY_CONDITION:
        while _ACTIVE_IMAGE_SUMMARIES >= min(
            8, max(1, int(settings.image_summary_max_workers))
        ):
            _IMAGE_SUMMARY_CONDITION.wait()
        _ACTIVE_IMAGE_SUMMARIES += 1
    try:
        yield
    finally:
        with _IMAGE_SUMMARY_CONDITION:
            _ACTIVE_IMAGE_SUMMARIES -= 1
            _IMAGE_SUMMARY_CONDITION.notify_all()


@contextmanager
def _document_summary_slot() -> Iterator[None]:
    """限制跨文档文本摘要并发，同时允许它与 Embedding 重叠。"""
    global _ACTIVE_DOCUMENT_SUMMARIES
    with _DOCUMENT_SUMMARY_CONDITION:
        while _ACTIVE_DOCUMENT_SUMMARIES >= min(
            8, max(1, int(settings.document_summary_max_workers))
        ):
            _DOCUMENT_SUMMARY_CONDITION.wait()
        _ACTIVE_DOCUMENT_SUMMARIES += 1
    try:
        yield
    finally:
        with _DOCUMENT_SUMMARY_CONDITION:
            _ACTIVE_DOCUMENT_SUMMARIES -= 1
            _DOCUMENT_SUMMARY_CONDITION.notify_all()


@contextmanager
def _group_operation_slot(group_id: int) -> Iterator[None]:
    """登记同库活跃操作，防止向量目录在任务运行时被并发删除。"""
    with _GROUP_LIFECYCLE_CONDITION:
        if group_id in _DELETING_GROUPS:
            raise RuntimeError("文献库正在删除，无法开始新的文档操作。")
        _ACTIVE_GROUP_OPERATIONS[group_id] = (
            _ACTIVE_GROUP_OPERATIONS.get(group_id, 0) + 1
        )
    try:
        yield
    finally:
        with _GROUP_LIFECYCLE_CONDITION:
            remaining = _ACTIVE_GROUP_OPERATIONS.get(group_id, 1) - 1
            if remaining > 0:
                _ACTIVE_GROUP_OPERATIONS[group_id] = remaining
            else:
                _ACTIVE_GROUP_OPERATIONS.pop(group_id, None)
            _GROUP_LIFECYCLE_CONDITION.notify_all()


@contextmanager
def _group_delete_slot(group_id: int) -> Iterator[None]:
    """等待同库入库收敛，并阻止删除期间再启动新任务。"""
    with _GROUP_LIFECYCLE_CONDITION:
        while (
            group_id in _DELETING_GROUPS
            or _ACTIVE_GROUP_OPERATIONS.get(group_id, 0) > 0
        ):
            _GROUP_LIFECYCLE_CONDITION.wait()
        _DELETING_GROUPS.add(group_id)
    try:
        yield
    finally:
        with _GROUP_LIFECYCLE_CONDITION:
            _DELETING_GROUPS.discard(group_id)
            _GROUP_LIFECYCLE_CONDITION.notify_all()


def _noop(_p: float, _m: str) -> None:
    pass


def _batch_noop(_index: int, _total: int, _p: float, _m: str) -> None:
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
    with _vector_mutation_lock(group_id):
        get_vectorstore(group_id).delete(where={"doc_id": doc_id})


def _safe_unlink(path: Path | None) -> None:
    """回滚阶段的文件清理不应覆盖最初的入库异常。"""
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _docstore_document_dir(doc_id: int) -> Path:
    """返回单篇文档的资产目录；目录名只来自数据库整数，避免路径穿越。"""
    root = getattr(settings, "docstore_dir", settings.parsed_dir.parent / "docstore")
    return Path(root) / f"doc_{doc_id}"


def _safe_remove_docstore(doc_id: int) -> None:
    """仅删除当前文档自己的图片资产，清理失败不覆盖原始业务异常。"""
    target = _docstore_document_dir(doc_id)
    try:
        root = Path(getattr(settings, "docstore_dir", settings.parsed_dir.parent / "docstore"))
        target.resolve().relative_to(root.resolve())
        shutil.rmtree(target, ignore_errors=True)
    except (OSError, ValueError):
        pass


def _enrich_figure_blocks(
    parsed: dict,
    stored_pdf: Path,
    *,
    doc_id: int,
) -> int:
    """提取原图、生成摘要，并把二者关联回 MinerU 的 figure layout block。"""
    blocks = parsed.setdefault("blocks", [])
    if not isinstance(blocks, list):
        return 0
    figures = [block for block in blocks if isinstance(block, dict) and block.get("type") == "figure"]
    assets = extract_pdf_figures(
        stored_pdf,
        docstore_dir=Path(getattr(settings, "docstore_dir", settings.parsed_dir.parent / "docstore")),
        doc_id=doc_id,
        figure_blocks=figures,
    )
    if not assets:
        return 0
    blocks_by_id = {str(block.get("block_id")): block for block in figures}
    jobs: list[tuple[dict, dict | None, str]] = []
    for asset in assets:
        block = blocks_by_id.get(str(asset.get("block_id")))
        caption = str(block.get("text") or "").strip() if block is not None else ""
        jobs.append((asset, block, caption))

    def _summarize(job: tuple[dict, dict | None, str]) -> str:
        asset, _block, caption = job
        try:
            with _image_summary_slot():
                return summarize_image(asset["image_path"], caption=caption)
        except Exception:
            # 单张图失败不影响其他图片，也不撤销正文入库。
            return ""

    # 同一提取图片可能被多个 figure block 引用；视觉内容只分析一次，各块图注仍分别保留。
    unique_jobs: dict[str, tuple[dict, dict | None, str]] = {}
    for job in jobs:
        unique_jobs.setdefault(str(job[0].get("image_path", "")), job)
    summary_jobs = list(unique_jobs.values())
    summary_by_path: dict[str, str] = {}
    summary_enabled = bool(
        getattr(settings, "image_summary_enabled", True)
        and getattr(settings, "has_api_key", False)
    )
    if summary_enabled and summary_jobs:
        futures = [
            _IMAGE_SUMMARY_EXECUTOR.submit(_summarize, job)
            for job in summary_jobs
        ]
        # 按提交顺序取结果；worker 不修改 parsed，避免完成顺序造成错配。
        summary_values = [future.result() for future in futures]
        summary_by_path = {
            str(job[0].get("image_path", "")): summary
            for job, summary in zip(summary_jobs, summary_values)
        }

    for asset, block, caption in jobs:
        summary = summary_by_path.get(str(asset.get("image_path", "")), "")
        if block is not None:
            block.update(asset)
            block["text"] = _figure_index_text(summary, caption)
            continue
        # PyMuPDF 回退没有 figure block 时，把 synthetic 资产追加为可索引块。
        blocks.append(
            {
                "type": "figure",
                "reading_order": 10_000 + len(blocks),
                **asset,
                "text": _figure_index_text(summary, ""),
            }
        )
    return len(assets)


def _figure_index_text(summary: str, caption: str) -> str:
    """向量化的仅是摘要/图注，绝不把 base64 或图片二进制混入文本索引。"""
    parts = ["Figure"]
    if caption:
        parts.append(f"Caption: {caption}")
    if summary:
        parts.append(f"Visual summary: {summary}")
    if len(parts) == 1:
        parts.append("Visual summary unavailable; original figure is stored for multimodal answering.")
    return "\n".join(parts)


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
    _safe_remove_docstore(doc_id)

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
    stage_timings: dict[str, float] | None = None,
    _parse_semaphore: Semaphore | None = None,
) -> storage.DocumentRow:
    """处理单个 PDF；同一文献库删除会等待该任务完整收敛。"""
    started = time.perf_counter()
    try:
        with _group_operation_slot(group_id):
            return _ingest_pdf_impl(
                group_id,
                source_path,
                filename,
                progress=progress,
                make_summary=make_summary,
                stage_timings=stage_timings,
                _parse_semaphore=_parse_semaphore,
            )
    finally:
        if stage_timings is not None:
            stage_timings["total"] = round(time.perf_counter() - started, 3)


def _ingest_pdf_impl(
    group_id: int,
    source_path: str | Path,
    filename: str,
    progress: ProgressCb = _noop,
    make_summary: bool = True,
    stage_timings: dict[str, float] | None = None,
    _parse_semaphore: Semaphore | None = None,
) -> storage.DocumentRow:
    """处理单个 PDF，并让独立的远程增强阶段尽可能重叠执行。"""
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
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    timings_lock = RLock()

    def _timed(stage: str, callback):
        started = time.perf_counter()
        try:
            return callback()
        finally:
            with timings_lock:
                timings[stage] = round(time.perf_counter() - started, 3)

    try:
        def _prepare_source():
            storage.update_document(doc_id, original_path=str(stored_pdf))
            return _install_source_pdf(source_path, stored_pdf)

        copied_source, source_backup = _timed("prepare", _prepare_source)

        progress(0.1, f"正在解析 {filename} ...")
        # 解析名额只覆盖 MinerU/PyMuPDF，不再被图片、Embedding、摘要长期占用。
        parse_wait_started = time.perf_counter()
        local_parse_slot = _parse_semaphore or Semaphore(1)
        with local_parse_slot:
            with _pdf_task_slot():
                with timings_lock:
                    timings["parse_wait"] = round(
                        time.perf_counter() - parse_wait_started, 3
                    )
                parsed = _timed(
                    "parse",
                    lambda: _run_parse_hooks(
                        stored_pdf,
                        group_id=group_id,
                        doc_id=doc_id,
                        filename=filename,
                    ),
                )

        progress(0.28, "正在提取图片并并行生成可检索摘要 ...")
        image_count = _timed(
            "image_enrichment",
            lambda: _enrich_figure_blocks(parsed, stored_pdf, doc_id=doc_id),
        )

        # 将 doc_id 纳入文件名，避免同名重传覆盖其他记录的解析结果。
        parsed_path = (
            settings.parsed_dir / f"{group_id}_{doc_id}_{Path(filename).stem}.json"
        )
        _timed("save_parsed", lambda: save_parsed(parsed, parsed_path))

        progress(0.4, "正在切分文档 ...")
        version = time.strftime("%Y%m%d%H%M%S")

        def _chunk_documents() -> list[Document]:
            chunked = parsed_to_documents(
                parsed, group_id, filename, doc_id=doc_id, version=version
            )
            return _run_before_ingest_hook(
                chunked,
                group_id=group_id,
                doc_id=doc_id,
                filename=filename,
                version=version,
            )

        documents = _timed("chunking", _chunk_documents)
        if not documents:
            raise ValueError("解析结果未生成任何可入库分块。")

        progress(0.5, "正在并行执行 Embedding 与文档摘要 ...")

        def _prepare_vectors():
            return _timed(
                "embedding", lambda: prepare_ingestion_batch(documents)
            )

        def _generate_document_summary() -> str:
            try:
                wait_started = time.perf_counter()
                with _document_summary_slot():
                    with timings_lock:
                        timings["document_summary_wait"] = round(
                            time.perf_counter() - wait_started, 3
                        )
                    return _timed(
                        "document_summary", lambda: generate_summary(documents)
                    )
            except Exception as exc:  # noqa: BLE001
                return f"(摘要生成失败: {exc})"

        summary = ""
        if make_summary:
            summary_future = _DOCUMENT_SUMMARY_EXECUTOR.submit(
                _generate_document_summary
            )
            try:
                prepared = _prepare_vectors()
            except Exception:
                summary_future.cancel()
                raise
            summary = summary_future.result()
        else:
            prepared = _prepare_vectors()

        progress(0.85, "Embedding 完成，正在提交 Chroma ...")
        # 网络 Embedding 已在锁外完成；仅最终 upsert/失败补偿需要同 group 锁。
        indexing_attempted = True
        commit_wait_started = time.perf_counter()
        with _vector_mutation_lock(group_id):
            with timings_lock:
                timings["vector_commit_wait"] = round(
                    time.perf_counter() - commit_wait_started, 3
                )
            num_chunks = _timed(
                "vector_commit",
                lambda: ingest_documents(
                    group_id, documents, prepared=prepared
                ),
            )
        if num_chunks <= 0:
            raise RuntimeError("向量库未写入任何分块。")
        storage.update_document(
            doc_id,
            parsed_json_path=str(parsed_path),
            num_chunks=num_chunks,
            status="indexing",
        )

        progress(0.95, "正在提交元数据并刷新检索缓存 ...")
        invalidate_retriever(group_id)
        # ready 是最后的提交点：在此之前的任何异常都会触发补偿回滚。
        storage.update_document(doc_id, summary=summary, status="ready")

        result = storage.get_document(doc_id)
        if result is None:
            raise RuntimeError("入库完成后无法读回文档记录。")

        # 成功提交后旧文件备份已无需保留。
        _safe_unlink(source_backup)
        try:
            progress(
                1.0,
                (
                    f"{filename} 完成：解析 {timings.get('parse', 0):.1f}s · "
                    f"图片 {timings.get('image_enrichment', 0):.1f}s · "
                    f"Embedding {timings.get('embedding', 0):.1f}s · "
                    f"摘要 {timings.get('document_summary', 0):.1f}s · "
                    f"{image_count} 图/{num_chunks} 块"
                ),
            )
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
    finally:
        timings["total"] = round(time.perf_counter() - total_started, 3)
        if stage_timings is not None:
            stage_timings.clear()
            stage_timings.update(timings)


def ingest_pdfs_parallel(
    group_id: int,
    tasks: Sequence[PdfIngestionTask],
    *,
    max_workers: int | None = None,
    progress: BatchProgressCb = _batch_noop,
    make_summary: bool = True,
) -> list[PdfIngestionResult]:
    """在同库生命周期租约内执行完整批次，防止排队任务期间文献库被删除。"""
    if not tasks:
        return []
    with _group_operation_slot(group_id):
        return _ingest_pdfs_parallel_impl(
            group_id,
            tasks,
            max_workers=max_workers,
            progress=progress,
            make_summary=make_summary,
        )


def _ingest_pdfs_parallel_impl(
    group_id: int,
    tasks: Sequence[PdfIngestionTask],
    *,
    max_workers: int | None = None,
    progress: BatchProgressCb = _batch_noop,
    make_summary: bool = True,
) -> list[PdfIngestionResult]:
    """以有界线程池并行处理多篇 PDF，并隔离单文档失败。

    每个 worker 只发送进度事件；``progress`` 始终由调用线程执行，因此可安全更新
    Streamlit。旧版本清理在所有新版本任务结束后按输入顺序执行，避免删除与并行写入交叉。
    """
    normalized = tuple(tasks)
    if not normalized:
        return []
    filenames = [task.filename for task in normalized]
    if len(set(filenames)) != len(filenames):
        raise ValueError("同一批次不能包含重名 PDF，请重命名后重试。")

    configured = min(
        8,
        max(
            1,
            int(
                settings.pdf_max_parallel_documents
                if max_workers is None
                else max_workers
            ),
        ),
    )
    # 外层 worker 只负责推进文档状态机；重资源阶段由共享 executor/闸门限流。
    # 让最多 32 篇文档同时在流水线中，可避免前几篇等待图片/API 时饿死解析槽。
    worker_count = min(len(normalized), 32)
    local_parse_semaphore = Semaphore(configured)
    updates: SimpleQueue[tuple[int, float, str]] = SimpleQueue()
    results: list[PdfIngestionResult | None] = [None] * len(normalized)

    def _report(index: int, value: float, message: str) -> None:
        updates.put((index, min(1.0, max(0.0, float(value))), str(message)))

    def _run(index: int, task: PdfIngestionTask) -> PdfIngestionResult:
        timings: dict[str, float] = {}
        try:
            document = ingest_pdf(
                group_id,
                task.source_path,
                task.filename,
                progress=lambda value, message: _report(index, value, message),
                make_summary=make_summary,
                stage_timings=timings,
                _parse_semaphore=local_parse_semaphore,
            )
            return PdfIngestionResult(
                task.filename,
                document=document,
                stage_seconds=timings,
            )
        except Exception as exc:  # noqa: BLE001 - 单篇失败不能取消整个批次
            return PdfIngestionResult(
                task.filename,
                error=f"{type(exc).__name__}: {exc}"[:500],
                stage_seconds=timings,
            )

    def _drain_progress() -> None:
        while True:
            try:
                index, value, message = updates.get_nowait()
            except Empty:
                return
            try:
                progress(index, len(normalized), value, message)
            except Exception:  # noqa: BLE001 - UI 消失不应中断后台入库
                continue

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix=f"refmind-pdf-g{group_id}",
    ) as executor:
        future_indexes = {
            executor.submit(_run, index, task): index
            for index, task in enumerate(normalized)
        }
        pending = set(future_indexes)
        while pending:
            done, pending = wait(
                pending, timeout=0.1, return_when=FIRST_COMPLETED
            )
            _drain_progress()
            for future in done:
                index = future_indexes[future]
                # _run 已隔离普通异常；此处仍为线程框架级异常保留防御结果。
                try:
                    results[index] = future.result()
                except Exception as exc:  # pragma: no cover - 防御边界
                    results[index] = PdfIngestionResult(
                        normalized[index].filename,
                        error=f"{type(exc).__name__}: {exc}"[:500],
                    )
                try:
                    progress(index, len(normalized), 1.0, "处理结束")
                except Exception:  # noqa: BLE001
                    pass
        _drain_progress()

    final_results = [
        result
        if result is not None
        else PdfIngestionResult(normalized[index].filename, error="任务未返回结果")
        for index, result in enumerate(results)
    ]
    # 新版本 ready 后才清理旧版本；清理失败只产生告警，不撤销已经提交的新版本。
    for task, result in zip(normalized, final_results):
        if not result.succeeded or task.previous_document_id is None:
            continue
        try:
            remove_document(task.previous_document_id)
        except Exception as exc:  # noqa: BLE001
            result.cleanup_warning = f"旧版本待重试清理：{exc}"
    return final_results


def remove_document(doc_id: int) -> None:
    """删除文档的分块、文件与元数据，并重建检索器。"""
    doc = storage.get_document(doc_id)
    if doc is None:
        return
    with _group_operation_slot(doc.group_id):
        _remove_document_impl(doc_id)


def _remove_document_impl(doc_id: int) -> None:
    """已持有 group 生命周期租约时执行可恢复的单文档删除。"""
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
        _safe_remove_docstore(doc_id)
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
    with _group_delete_slot(group_id):
        # 逐文档走可恢复删除状态机；任何失败都会保留组和剩余记录。
        for doc in storage.list_documents(group_id):
            _remove_document_impl(doc.id)

        invalidate_retriever(group_id)
        chroma_dir = settings.chroma_persist_dir / f"group_{group_id}"
        if chroma_dir.exists():
            try:
                shutil.rmtree(chroma_dir)
            except OSError as exc:
                raise RuntimeError("向量库目录删除失败，文献库记录已保留。") from exc

        storage.delete_group(group_id)
