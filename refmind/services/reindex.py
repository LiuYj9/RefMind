"""Explicit offline reconstruction of legacy indexes from a verified original.

This creates a new index generation. It never guesses the historic generation
from untrusted chunks and never changes the document ID or original file.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from .. import storage
from ..config import settings
from ..parsing import parse_pdf, save_parsed
from ..rag.provenance import file_fingerprint, processing_versions
from ..storage.runtime_lock import acquire_runtime_lock, release_runtime_lock
from . import ingestion


REINDEX_STATES = {"reindexing", "reindex_failed"}


def _verify_source(row, expected: str) -> Path:
    """Require independent persisted provenance, never a digest from the index."""
    source = Path(row.original_path or "")
    if not source.is_file() or not source.resolve().is_relative_to(settings.upload_dir.resolve()):
        raise ValueError("历史原文不存在或不在原文目录中")
    recorded = []
    embedding = None
    if row.file_sha256:
        recorded.append(row.file_sha256)
    if row.parsed_json_path:
        parsed = json.loads(Path(row.parsed_json_path).read_text(encoding="utf-8"))
        if parsed.get("file_sha256"):
            recorded.append(parsed["file_sha256"])
        embedding = parsed.get("processing_versions")
    with storage.connect() as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='rm_document_sources'").fetchone():
            registered = conn.execute("SELECT sha256,group_id FROM rm_document_sources WHERE doc_id=?", (row.id,)).fetchone()
            if registered:
                if registered["group_id"] != row.group_id:
                    raise ValueError("历史来源登记的文献库不一致")
                recorded.append(registered["sha256"])
    if not recorded or any(value != expected for value in recorded):
        raise ValueError("缺少一致的历史文件指纹，须先核对独立入库记录")
    if file_fingerprint(source) != expected:
        raise ValueError("历史原文与确认的文件指纹不一致")
    current = processing_versions("verification")
    if not isinstance(embedding, dict) or any(
        embedding.get(key) != current[key] for key in ("embedding_provider", "embedding_model")
    ):
        raise ValueError("历史嵌入配置缺失或与当前配置不同，不能在原库中混用向量")
    return source


def rebuild_legacy_index(doc_id: int, *, expected_sha256: str) -> storage.DocumentRow:
    """Stop writers first. Failures retain files/references and require retry.

The caller supplies a SHA256 confirmed against independent ingestion records.
An already protected ready document is returned without repeating paid work.
"""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("须提供已核对的64位小写SHA256")
    # A unique mode also rejects concurrent/reentrant calls in this process.
    acquire_runtime_lock(settings.database_path, f"legacy-reindex-{uuid4().hex}")
    try:
        return _rebuild_legacy_index_locked(doc_id, expected_sha256=expected_sha256)
    finally:
        release_runtime_lock(settings.database_path)


def _rebuild_legacy_index_locked(doc_id: int, *, expected_sha256: str) -> storage.DocumentRow:
    """Internal entry for an offline maintenance process already holding the writer lock."""
    previous_openai_library = settings.allow_openai_library
    try:
        storage.init_db()
        row = storage.get_document(doc_id)
        if row is None or row.status not in {"ready", *REINDEX_STATES}:
            raise ValueError("文献不存在或当前状态不允许历史索引重建")
        source = _verify_source(row, expected_sha256)
        if row.status == "ready" and row.index_version and row.file_sha256:
            return row
        if row.index_version or row.file_sha256:
            raise ValueError("索引身份不完整或不是待迁移的历史记录，须人工核对")
        settings.allow_openai_library = True
        with ingestion._group_operation_slot(row.group_id):
            return _rebuild(row, source, expected_sha256)
    finally:
        settings.allow_openai_library = previous_openai_library


def _rebuild(row, source: Path, expected: str) -> storage.DocumentRow:
    version = "reindex-" + uuid4().hex
    parsed_path = settings.parsed_dir / f"{row.group_id}_{row.id}_{version}.json"
    # Keep original parsed data and original figure files for saved citations.
    asset_root = settings.docstore_dir / f"doc_{row.id}" / "reindexes" / version
    with storage.connect() as conn:
        changed = conn.execute(
            "UPDATE documents SET status='reindexing' WHERE id=? AND status=? "
            "AND original_path=? AND COALESCE(index_version,'')='' AND COALESCE(file_sha256,'')=''",
            (row.id, row.status, row.original_path),
        ).rowcount
        if changed != 1:
            raise RuntimeError("文献已变化，未开始重建")
    try:
        ingestion.invalidate_retriever(row.group_id)
        # Parse a verified byte copy so a changed live file cannot be relabelled.
        with tempfile.TemporaryDirectory(prefix="reindex-", dir=settings.parsed_dir) as folder:
            snapshot = Path(folder) / "source.pdf"
            shutil.copyfile(source, snapshot)
            if file_fingerprint(snapshot) != expected:
                raise ValueError("复制期间原文发生变化")
            parsed = parse_pdf(snapshot)
            parsed.update(file_sha256=expected, index_version=version,
                          paper_title=row.paper_title or Path(row.filename).stem,
                          library_index=row.library_index,
                          processing_versions=processing_versions(str(parsed.get("parser") or "unknown")))
            ingestion._enrich_figure_blocks(parsed, snapshot, doc_id=row.id, asset_root=asset_root)
            if file_fingerprint(snapshot) != expected:
                raise ValueError("解析期间原文副本发生变化")
            save_parsed(parsed, parsed_path)
        # Reconstruction deliberately uses core parsing/chunking, not stale plugin caches.
        documents = ingestion.parsed_to_documents(parsed, row.group_id, row.filename,
                                                   doc_id=row.id, version=version)
        if not documents:
            raise ValueError("原文未生成可入库分块")
        for document in documents:
            meta = document.metadata
            if (meta.get("doc_id"), meta.get("version"), meta.get("file_sha256")) != (row.id, version, expected):
                raise ValueError("重建分块的文献身份不一致")
        prepared = ingestion.prepare_ingestion_batch(documents)
        _verify_source(row, expected)
        with ingestion._vector_mutation_lock(row.group_id):
            # Old/partial vectors are not trusted and must not starve retrieval.
            # State was withdrawn before this destructive index-only operation.
            ingestion._delete_chunks_by_doc_id(row.group_id, row.id)
            count = ingestion.ingest_documents(row.group_id, documents, prepared=prepared)
        if count != len(documents):
            raise RuntimeError("重建索引未完整写入")
        _verify_source(row, expected)
        ingestion.invalidate_retriever(row.group_id)
        with storage.connect() as conn:
            changed = conn.execute(
                "UPDATE documents SET status='ready',index_version=?,file_sha256=?,parsed_json_path=?,num_chunks=? "
                "WHERE id=? AND status='reindexing' AND original_path=? "
                "AND COALESCE(index_version,'')='' AND COALESCE(file_sha256,'')=''",
                (version, expected, str(parsed_path), count, row.id, row.original_path),
            ).rowcount
            if changed != 1:
                raise RuntimeError("提交前文献状态改变，未发布新索引")
    except BaseException:
        # Also safe for KeyboardInterrupt. A hard kill leaves 'reindexing'.
        # Neither state is startup garbage; do not restore the unsafe legacy path.
        with storage.connect() as conn:
            conn.execute("UPDATE documents SET status='reindex_failed' WHERE id=? AND status='reindexing'",
                         (row.id,))
        raise
    result = storage.get_document(row.id)
    if result is None:
        raise RuntimeError("无法读回已提交的重建记录")
    return result
