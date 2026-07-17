"""服务子包：跨模块的高层业务编排。"""

from .academic_search import (
    AcademicSearchError,
    AcademicSearchResult,
    provider_label,
    sanitize_external_url,
    search_academic_papers,
)
from .ingestion import (
    backfill_document_titles,
    PdfIngestionResult,
    PdfIngestionTask,
    ingest_pdf,
    ingest_pdfs_parallel,
    recover_incomplete_ingestions,
    remove_document,
    remove_documents,
    remove_group,
)

__all__ = [
    "AcademicSearchError",
    "AcademicSearchResult",
    "provider_label",
    "sanitize_external_url",
    "search_academic_papers",
    "ingest_pdf",
    "ingest_pdfs_parallel",
    "PdfIngestionTask",
    "PdfIngestionResult",
    "backfill_document_titles",
    "recover_incomplete_ingestions",
    "remove_document",
    "remove_documents",
    "remove_group",
]
