"""服务子包：跨模块的高层业务编排。"""

from .ingestion import ingest_pdf, remove_document, remove_group

__all__ = ["ingest_pdf", "remove_document", "remove_group"]
