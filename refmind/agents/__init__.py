"""RefMind 的受控多智能体编排组件。

这里的“智能体”都有明确边界：规划、并行检索、证据筛选和答案审校。
编排器本身不依赖具体模型供应商，只要求注入对象实现 ``invoke``。
"""

from .orchestration import (
    AgentRunResult,
    AnswerDraft,
    canonicalize_retrieval_query,
    INSUFFICIENT_EVIDENCE_REPLY,
    MultiAgentConfig,
    MultiAgentOrchestrator,
)

__all__ = [
    "AgentRunResult",
    "AnswerDraft",
    "canonicalize_retrieval_query",
    "INSUFFICIENT_EVIDENCE_REPLY",
    "MultiAgentConfig",
    "MultiAgentOrchestrator",
]
