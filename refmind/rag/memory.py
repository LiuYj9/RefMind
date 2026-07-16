"""会话历史与跨会话用户长期记忆。

``RelevantMemory`` 保留原有的单会话滑动窗口；``LongTermMemoryService`` 则负责
从用户话语提取原子化语义/情景记忆，并在独立 SQLite 表中完成检索、合并、冲突
失效与时间衰减。长期记忆只描述“用户是谁、正在做什么”，绝不保存论文事实。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any

import numpy as np
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from .. import storage
from ..config import settings
from ..llm import get_embedding_model, get_llm


MEMORY_EXTRACTION_PROMPT = """你是 RefMind 的用户长期记忆筛选器。只分析本轮“用户消息”，
不得把助手回答或论文检索证据当成用户记忆。

值得保存：稳定偏好、研究方向、用户背景、术语习惯、持续任务/阅读进度，以及对后续交流
确有帮助的重要会话事件。不值得保存：寒暄、一次性命令、无后续价值的临时问题、敏感凭证、
论文中的结论/数值/公式/表格事实、模型自己推断而用户未表达的事实。

memory_type 只能是 semantic 或 episodic。semantic 是相对稳定的用户事实；episodic 是有时间
和会话边界的用户经历/任务事件。content 必须是独立、简短、以“用户”开头的原子事实。
semantic 的 memory_key 要使用稳定主题键（如 research.topic、preference.answer_style、
terminology.quench、task.paper_a_progress），同一主题的新事实必须复用同一个键；episodic 通常
令 memory_key 为 null。importance/confidence 取 0~1。只有 should_store=true 的项才会写入。

严格返回 JSON，不要 Markdown：
{"memories":[{"content":"...","memory_type":"semantic|episodic","subtype":"preference|research|task|terminology|background|interaction","memory_key":"...或null","importance":0.0,"confidence":0.0,"should_store":true,"reason":"简短理由"}]}
若没有值得保存的信息，返回 {"memories":[]}。"""

_ALLOWED_TYPES = {"semantic", "episodic"}
_ALLOWED_SUBTYPES = {
    "preference",
    "research",
    "task",
    "terminology",
    "background",
    "interaction",
}
_ALWAYS_RELEVANT_SUBTYPES = {"preference", "terminology"}


@dataclass(frozen=True)
class MemoryCandidate:
    """已经过结构校验、等待持久化决策的原子记忆。"""

    content: str
    memory_type: str
    subtype: str
    memory_key: str | None
    importance: float
    confidence: float


@dataclass(frozen=True)
class RetrievedMemory:
    """带本轮相关性与衰减后权重的召回记忆。"""

    memory: storage.LongTermMemory
    relevance: float
    effective_importance: float
    score: float


@dataclass(frozen=True)
class MemoryUpdateResult:
    inserted: int = 0
    merged: int = 0
    superseded: int = 0
    skipped: int = 0


def _clamp(value: object, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _normalise_content(content: str) -> str:
    return re.sub(r"[\W_]+", "", content, flags=re.UNICODE).lower()


def _content_hash(content: str) -> str:
    return hashlib.sha256(_normalise_content(content).encode("utf-8")).hexdigest()


def _serialise_embedding(vector: list[float] | None) -> str | None:
    if vector is None:
        return None
    return json.dumps([float(value) for value in vector], separators=(",", ":"))


def _deserialise_embedding(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    try:
        vector = np.asarray(json.loads(value), dtype=float)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return vector if vector.ndim == 1 and vector.size else None


def _parse_db_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _lexical_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalise_content(left), _normalise_content(right)).ratio()


def _has_negation(text: str) -> bool:
    return any(token in text.lower() for token in ("不再", "不是", "停止", "取消", "don't", "not "))


def _contains_sensitive_data(text: str) -> bool:
    lowered = text.lower()
    labels = (
        "密码",
        "口令",
        "身份证",
        "银行卡",
        "api key",
        "apikey",
        "access token",
        "refresh token",
        "private key",
        "secret key",
    )
    return any(label in lowered for label in labels)


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
        group_id: int | None = None,
        user_id: str | None = None,
        max_turns: int | None = None,
        threshold: float | None = None,
    ) -> None:
        self.session_id = session_id
        try:
            session = storage.get_session(session_id) if group_id is None else None
        except Exception:  # noqa: BLE001 - 兼容尚未初始化数据库的库调用方
            session = None
        self.group_id = group_id if group_id is not None else (session.group_id if session else None)
        self.user_id = user_id or settings.user_id
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


class LongTermMemoryService:
    """SQLite 长期记忆服务；节点边界可在后续替换为 Mem0 add/search。"""

    def __init__(self, *, embedding_model: Any | None = None, llm: Any | None = None) -> None:
        self._embedding_model = embedding_model
        self._llm = llm

    def _embed_query(self, text: str) -> list[float] | None:
        try:
            model = self._embedding_model or get_embedding_model()
            return [float(value) for value in model.embed_query(text)]
        except Exception:  # noqa: BLE001 - 记忆增强失败不能阻断 RAG 主链
            return None

    def extract(
        self,
        user_message: str,
        *,
        recalled_memories: list[RetrievedMemory] | None = None,
    ) -> list[MemoryCandidate]:
        """从单条用户消息提取值得跨会话保留的候选记忆。"""
        if not settings.long_term_memory_enabled or not user_message.strip():
            return []
        existing = "\n".join(
            f"- {item.memory.memory_key or '-'}: {item.memory.content}"
            for item in (recalled_memories or [])
        )
        prompt = (
            f"本轮用户消息：\n{user_message.strip()}\n\n"
            f"本轮已召回的旧记忆（仅用于保持主题键一致，不得照抄）：\n{existing or '无'}"
        )
        try:
            llm = self._llm or get_llm(temperature=0.0)
            response = llm.invoke(
                [
                    SystemMessage(content=MEMORY_EXTRACTION_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception:  # noqa: BLE001 - 提取是非关键增强
            return []

        raw_items = _parse_json_object(_message_text(response)).get("memories", [])
        if not isinstance(raw_items, list):
            return []
        candidates: list[MemoryCandidate] = []
        for raw in raw_items[: settings.long_term_memory_max_candidates]:
            if not isinstance(raw, dict) or raw.get("should_store") is not True:
                continue
            content = " ".join(str(raw.get("content", "")).split())
            memory_type = str(raw.get("memory_type", "")).lower()
            subtype = str(raw.get("subtype", "interaction")).lower()
            importance = _clamp(raw.get("importance"), 0.5)
            confidence = _clamp(raw.get("confidence"), 0.5)
            key_value = raw.get("memory_key")
            memory_key = str(key_value).strip().lower() if key_value else None
            if memory_key and not re.fullmatch(r"[a-z0-9_.:-]{1,120}", memory_key):
                memory_key = None
            if (
                memory_type not in _ALLOWED_TYPES
                or subtype not in _ALLOWED_SUBTYPES
                or not content.startswith("用户")
                or not 6 <= len(content) <= 300
                or _contains_sensitive_data(content)
                or importance < settings.long_term_memory_min_importance
                or confidence < settings.long_term_memory_min_confidence
            ):
                continue
            if memory_type == "semantic" and not memory_key:
                memory_key = f"{subtype}.{_content_hash(content)[:12]}"
            if memory_type == "episodic":
                memory_key = None
            candidates.append(
                MemoryCandidate(
                    content=content,
                    memory_type=memory_type,
                    subtype=subtype,
                    memory_key=memory_key,
                    importance=importance,
                    confidence=confidence,
                )
            )
        return candidates

    def _effective_importance(
        self, memory: storage.LongTermMemory, *, now: datetime
    ) -> float:
        anchor = _parse_db_time(memory.last_accessed_at) or _parse_db_time(memory.updated_at)
        age_days = max(0.0, (now - (anchor or now)).total_seconds() / 86400.0)
        half_life = (
            settings.semantic_memory_half_life_days
            if memory.memory_type == "semantic"
            else settings.episodic_memory_half_life_days
        )
        decay = math.pow(0.5, age_days / max(1.0, half_life))
        usage_boost = min(0.15, math.log1p(memory.access_count) * 0.03)
        return min(1.0, float(memory.importance) * decay + usage_boost)

    def _archive_stale(
        self, memories: list[storage.LongTermMemory], *, now: datetime
    ) -> list[storage.LongTermMemory]:
        active: list[storage.LongTermMemory] = []
        for memory in memories:
            expiry = _parse_db_time(memory.expires_at)
            anchor = _parse_db_time(memory.last_accessed_at) or _parse_db_time(memory.updated_at)
            inactive_days = max(0.0, (now - (anchor or now)).total_seconds() / 86400.0)
            max_inactive = (
                settings.semantic_memory_max_inactive_days
                if memory.memory_type == "semantic"
                else settings.episodic_memory_max_inactive_days
            )
            should_archive = bool(expiry and expiry <= now) or (
                inactive_days >= max_inactive
                and self._effective_importance(memory, now=now)
                < settings.long_term_memory_archive_threshold
            )
            if should_archive:
                storage.deactivate_long_term_memory(memory.id)
            else:
                active.append(memory)
        return active

    def search(
        self, query: str, *, user_id: str, group_id: int
    ) -> list[RetrievedMemory]:
        """仅在给定 user/group/active 边界内召回长期记忆。"""
        if not settings.long_term_memory_enabled:
            return []
        rows = storage.list_long_term_memories(
            user_id,
            group_id,
            active_only=True,
            limit=settings.long_term_memory_scan_limit,
        )
        if not rows:
            return []
        now = datetime.now(UTC).replace(tzinfo=None)
        rows = self._archive_stale(rows, now=now)
        query_vector = self._embed_query(query)
        scored: list[RetrievedMemory] = []
        for memory in rows:
            vector = _deserialise_embedding(memory.embedding)
            if query_vector is not None and vector is not None:
                relevance = _cosine(np.asarray(query_vector), vector)
            else:
                relevance = _lexical_similarity(query, memory.content)
            if memory.subtype in _ALWAYS_RELEVANT_SUBTYPES:
                relevance = max(relevance, 0.45)
            effective = self._effective_importance(memory, now=now)
            score = 0.65 * max(0.0, relevance) + 0.25 * effective + 0.10 * memory.confidence
            if relevance >= settings.long_term_memory_relevance_threshold:
                scored.append(RetrievedMemory(memory, relevance, effective, score))
        scored.sort(key=lambda item: (item.score, item.memory.updated_at), reverse=True)
        selected = scored[: settings.long_term_memory_top_k]
        storage.touch_long_term_memories([item.memory.id for item in selected])
        return selected

    def update(
        self,
        candidates: list[MemoryCandidate],
        *,
        user_id: str,
        group_id: int,
        session_id: int | None,
    ) -> MemoryUpdateResult:
        """写入候选；精确/语义重复合并，同主题冲突则软失效旧事实。"""
        inserted = merged = superseded = skipped = 0
        existing = storage.list_long_term_memories(user_id, group_id, active_only=True)
        for candidate in candidates:
            candidate_hash = _content_hash(candidate.content)
            vector = self._embed_query(candidate.content)
            exact = next(
                (
                    item
                    for item in existing
                    if item.content_hash == candidate_hash
                    and item.memory_type == candidate.memory_type
                ),
                None,
            )
            best: storage.LongTermMemory | None = exact
            best_similarity = 1.0 if exact else 0.0
            if best is None:
                candidate_vector = np.asarray(vector) if vector is not None else None
                for item in existing:
                    if item.memory_type != candidate.memory_type:
                        continue
                    stored_vector = _deserialise_embedding(item.embedding)
                    similarity = (
                        _cosine(candidate_vector, stored_vector)
                        if candidate_vector is not None and stored_vector is not None
                        else _lexical_similarity(candidate.content, item.content)
                    )
                    if similarity > best_similarity:
                        best, best_similarity = item, similarity

            same_key = next(
                (
                    item
                    for item in existing
                    if candidate.memory_key
                    and item.memory_key == candidate.memory_key
                    and item.memory_type == candidate.memory_type
                ),
                None,
            )
            polarity_changed = bool(
                best and _has_negation(best.content) != _has_negation(candidate.content)
            )
            duplicate = (
                best
                if best
                and not polarity_changed
                and best_similarity >= settings.long_term_memory_duplicate_threshold
                else None
            )
            if duplicate is not None and (same_key is None or duplicate.id == same_key.id):
                refreshed_expiry = duplicate.expires_at
                if duplicate.memory_type == "episodic":
                    refreshed_expiry = (
                        datetime.now(UTC).replace(tzinfo=None)
                        + timedelta(days=settings.episodic_memory_max_inactive_days)
                    ).isoformat(timespec="seconds")
                storage.update_long_term_memory(
                    duplicate.id,
                    session_id=session_id,
                    importance=min(1.0, max(duplicate.importance, candidate.importance) + 0.03),
                    confidence=max(duplicate.confidence, candidate.confidence),
                    embedding=duplicate.embedding or _serialise_embedding(vector),
                    expires_at=refreshed_expiry,
                )
                refreshed = storage.get_long_term_memory(duplicate.id)
                existing = [refreshed if item.id == duplicate.id else item for item in existing]
                merged += 1
                continue

            expires_at = None
            if candidate.memory_type == "episodic":
                expires_at = (
                    datetime.now(UTC).replace(tzinfo=None)
                    + timedelta(days=settings.episodic_memory_max_inactive_days)
                ).isoformat(timespec="seconds")
            created = storage.create_long_term_memory(
                user_id=user_id,
                group_id=group_id,
                session_id=session_id,
                content=candidate.content,
                memory_type=candidate.memory_type,
                subtype=candidate.subtype,
                memory_key=candidate.memory_key,
                content_hash=candidate_hash,
                importance=candidate.importance,
                confidence=candidate.confidence,
                embedding=_serialise_embedding(vector),
                expires_at=expires_at,
            )
            inserted += 1
            if same_key is not None:
                storage.deactivate_long_term_memory(same_key.id, superseded_by=created.id)
                existing = [item for item in existing if item.id != same_key.id]
                superseded += 1
            existing.append(created)
        return MemoryUpdateResult(inserted, merged, superseded, skipped)


def format_long_term_memories(memories: list[RetrievedMemory]) -> str:
    """渲染用户上下文，不生成可被误认为论文来源的引用标签。"""
    if not memories:
        return "（无相关用户长期记忆）"
    return "\n".join(
        f"- [{item.memory.memory_type}/{item.memory.subtype}] {item.memory.content}"
        for item in memories
    )
