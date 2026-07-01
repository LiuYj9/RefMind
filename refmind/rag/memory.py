"""按相关性过滤的长对话记忆。

消息存在 SQLite。每次提问时取最近若干轮，只保留与当前问题嵌入相似度
足够高的历史，避免长对话里无关内容干扰上下文。
"""

from __future__ import annotations

import numpy as np
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .. import storage
from ..config import settings
from ..llm import get_embedding_model


def _to_messages(rows: list[storage.Message]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for row in rows:
        if row.role == "user":
            messages.append(HumanMessage(content=row.content))
        else:
            messages.append(AIMessage(content=row.content))
    return messages


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class RelevantMemory:
    """滑动窗口 + 相关性过滤的会话记忆，一个实例对应一个会话。"""

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
        rows = storage.list_messages(self.session_id, limit=self.max_turns * 2)
        return _to_messages(rows)

    def get_relevant_history(self, query: str) -> list[BaseMessage]:
        """返回窗口内与 query 相关的历史；嵌入不可用时退回完整窗口。"""
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

        return relevant[-(self.max_turns * 2):]

    def add_user_message(self, content: str) -> None:
        storage.add_message(self.session_id, "user", content)

    def add_ai_message(self, content: str) -> None:
        storage.add_message(self.session_id, "assistant", content)
