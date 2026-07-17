"""RAG 子包：文档处理、混合检索、长对话记忆与 LangGraph 对话流水线。"""

from .document_processor import (
    PreparedVectorBatch,
    delete_document_chunks,
    get_vectorstore,
    ingest_documents,
    load_group_documents,
    parsed_to_documents,
    prepare_ingestion_batch,
)
from .graph import (
    ACADEMIC_NO_CONTEXT_REPLY,
    NO_CONTEXT_REPLY,
    RETRIEVAL_ERROR_REPLY,
    SERVICE_ERROR_REPLY,
    answer_question,
    build_graph,
    format_documents,
    get_system_prompt,
)
from .compression import compress_context
from .memory import (
    LongTermMemoryService,
    MemoryCandidate,
    MemoryUpdateResult,
    RelevantMemory,
    RetrievedMemory,
    format_long_term_memories,
)
from .reranker import dashscope_sdk_available, rerank
from .retrieval import build_retriever, get_retriever, invalidate_retriever

__all__ = [
    # 文档处理
    "parsed_to_documents",
    "PreparedVectorBatch",
    "prepare_ingestion_batch",
    "get_vectorstore",
    "ingest_documents",
    "load_group_documents",
    "delete_document_chunks",
    # 检索
    "build_retriever",
    "get_retriever",
    "invalidate_retriever",
    # 重排与上下文压缩
    "rerank",
    "dashscope_sdk_available",
    "compress_context",
    # 记忆
    "RelevantMemory",
    "LongTermMemoryService",
    "MemoryCandidate",
    "RetrievedMemory",
    "MemoryUpdateResult",
    "format_long_term_memories",
    # 对话图
    "answer_question",
    "build_graph",
    "get_system_prompt",
    "format_documents",
    "NO_CONTEXT_REPLY",
    "ACADEMIC_NO_CONTEXT_REPLY",
    "RETRIEVAL_ERROR_REPLY",
    "SERVICE_ERROR_REPLY",
]
