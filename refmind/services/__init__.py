"""服务子包：跨模块的高层业务编排。"""

from .ingestion import (
    PdfIngestionResult,
    PdfIngestionTask,
    ingest_pdf,
    ingest_pdfs_parallel,
    recover_incomplete_ingestions,
    remove_document,
    remove_group,
)

__all__ = [
    "ingest_pdf",
    "ingest_pdfs_parallel",
    "PdfIngestionTask",
    "PdfIngestionResult",
    "recover_incomplete_ingestions",
    "remove_document",
    "remove_group",
]
