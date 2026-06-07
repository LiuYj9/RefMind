"""带相关性过滤的长对话记忆。

消息持久化在 SQLite（见 :mod:`refmind.storage`）。每次新提问时，加载最近窗口
（最近 ``MEMORY_MAX_TURNS`` 轮），仅保留与当前问题嵌入相似度足够高的历史消息，
从而在长对话中聚焦上下文、剔除无关闲聊。
"""

from __future__ import annotations

import numpy as np
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .. import storage
from ..config import settings
from ..llm import get_embedding_model


def _to_messages(rows: list[storage.Message]) -> list[BaseMessage]:
    """将数据库消息行转换为 LangChain 消息对象。"""
    messages: list[BaseMessage] = []
    for row in rows:
        if row.role == "user":
            messages.append(HumanMessage(content=row.content))
        else:
            messages.append(AIMessage(content=row.content))
    return messages


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度。"""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class RelevantMemory:
    """基于语义相关性过滤的滑动窗口对话记忆。

    每个实例绑定单个对话会话。
    """

    def __init__(
        self,
        session_id: int,
        max_turns: int | None = None,
        threshold: float | None = None,
    ) -> None:
        self.session_id = session_id
        self.max_turns = max_turns or settings.memory_max_turns
        self.threshold = (
            threshold
            if threshold is not None
            else settings.memory_relevance_threshold
        )

    def load_window(self) -> list[BaseMessage]:
        """返回最近的消息（每轮 2 条）。"""
        rows = storage.list_messages(self.session_id, limit=self.max_turns * 2)
        return _to_messages(rows)

    def get_relevant_history(self, query: str) -> list[BaseMessage]:
        """返回最近窗口中与 ``query`` 相关的消息。

        当嵌入不可用（如未配置 API 密钥）时，回退为返回未过滤的窗口，
        保证应用仍可运行。
        """
        messages = self.load_window()
        if not messages:
            return []

        try:
            embeddings = get_embedding_model()
            query_vec = np.array(embeddings.embed_query(query))
            contents = [m.content for m in messages]
            doc_vecs = [np.array(v) for v in embeddings.embed_documents(contents)]
        except Exception as exc:  # noqa: BLE001
            print(f"[memory] 跳过相关性过滤（{exc}），使用完整窗口。")
            return messages

        relevant: list[BaseMessage] = []
        for message, vec in zip(messages, doc_vecs):
            if _cosine(query_vec, vec) >= self.threshold:
                relevant.append(message)

        # 保持原有顺序，并限制在配置的轮数预算内
        return relevant[-(self.max_turns * 2):]

    def add_user_message(self, content: str) -> None:
        """持久化一条用户消息。"""
        storage.add_message(self.session_id, "user", content)

    def add_ai_message(self, content: str) -> None:
        """持久化一条助手消息。"""
        storage.add_message(self.session_id, "assistant", content)
