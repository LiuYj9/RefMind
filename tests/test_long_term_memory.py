"""跨会话语义/情景记忆的存储、治理与 LangGraph 集成测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from refmind import storage
from refmind.config import settings
from refmind.rag import graph
from refmind.rag.memory import (
    LongTermMemoryService,
    MemoryCandidate,
    MemoryUpdateResult,
)


class _FakeEmbedding:
    def embed_query(self, text: str) -> list[float]:
        if "GraphRAG" in text:
            return [0.0, 1.0]
        return [1.0, 0.0]


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def invoke(self, _messages):
        return SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


class LongTermMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_db = settings.database_path
        settings.database_path = Path(self.tempdir.name) / "memory.db"
        storage.init_db()
        self.group = storage.create_group("研究项目")
        self.session = storage.create_session(self.group.id)

    def tearDown(self) -> None:
        settings.database_path = self.original_db
        self.tempdir.cleanup()

    def test_extract_keeps_only_worthy_atomic_user_memories(self) -> None:
        service = LongTermMemoryService(
            embedding_model=_FakeEmbedding(),
            llm=_FakeLLM(
                {
                    "memories": [
                        {
                            "content": "用户主要研究高温超导发电机",
                            "memory_type": "semantic",
                            "subtype": "research",
                            "memory_key": "research.topic",
                            "importance": 0.85,
                            "confidence": 0.93,
                            "should_store": True,
                        },
                        {
                            "content": "论文A准确率为92.6%",
                            "memory_type": "semantic",
                            "subtype": "research",
                            "memory_key": "paper.accuracy",
                            "importance": 0.9,
                            "confidence": 0.99,
                            "should_store": True,
                        },
                        {
                            "content": "用户说了你好",
                            "memory_type": "episodic",
                            "subtype": "interaction",
                            "memory_key": None,
                            "importance": 0.1,
                            "confidence": 0.99,
                            "should_store": True,
                        },
                        {
                            "content": "用户的API Key是sk-sensitive-value",
                            "memory_type": "semantic",
                            "subtype": "background",
                            "memory_key": "account.api_key",
                            "importance": 0.99,
                            "confidence": 0.99,
                            "should_store": True,
                        },
                    ]
                }
            ),
        )

        candidates = service.extract("我主要研究高温超导发电机")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].memory_type, "semantic")
        self.assertEqual(candidates[0].memory_key, "research.topic")

    def test_one_off_question_is_not_promoted_to_research_memory(self) -> None:
        service = LongTermMemoryService(
            embedding_model=_FakeEmbedding(),
            llm=_FakeLLM(
                {
                    "memories": [
                        {
                            "content": "用户正在研究降低线圈交流损耗的方法",
                            "memory_type": "semantic",
                            "subtype": "research",
                            "memory_key": "research.ac_loss",
                            "importance": 0.85,
                            "confidence": 0.95,
                            "should_store": True,
                        }
                    ]
                }
            ),
        )

        candidates = service.extract("降低线圈交流损耗的方法有哪些")

        self.assertEqual(candidates, [])

    def test_duplicate_is_merged_and_conflict_supersedes_old_fact(self) -> None:
        service = LongTermMemoryService(embedding_model=_FakeEmbedding())
        common = {
            "user_id": "user_001",
            "group_id": self.group.id,
            "session_id": self.session.id,
        }
        first = MemoryCandidate(
            "用户主要研究高温超导发电机",
            "semantic",
            "research",
            "research.topic",
            0.85,
            0.92,
        )
        duplicate = MemoryCandidate(
            "用户主要研究高温超导发电机",
            "semantic",
            "research",
            "research.topic",
            0.80,
            0.90,
        )
        conflict = MemoryCandidate(
            "用户不再研究高温超导发电机",
            "semantic",
            "research",
            "research.topic",
            0.90,
            0.95,
        )

        self.assertEqual(service.update([first], **common).inserted, 1)
        self.assertEqual(service.update([duplicate], **common).merged, 1)
        result = service.update([conflict], **common)

        self.assertEqual(result.inserted, 1)
        self.assertEqual(result.superseded, 1)
        active = storage.list_long_term_memories("user_001", self.group.id)
        all_rows = storage.list_long_term_memories(
            "user_001", self.group.id, active_only=False
        )
        self.assertEqual([row.content for row in active], [conflict.content])
        old = next(row for row in all_rows if row.content == first.content)
        self.assertEqual(old.is_active, 0)
        self.assertEqual(old.superseded_by, active[0].id)

    def test_search_is_scope_isolated_and_archives_expired_episode(self) -> None:
        embedding = json.dumps([1.0, 0.0])
        storage.create_long_term_memory(
            user_id="user_001",
            group_id=self.group.id,
            session_id=self.session.id,
            content="用户关注YBCO线圈失超检测",
            memory_type="semantic",
            subtype="research",
            memory_key="research.ybco",
            content_hash="semantic",
            importance=0.9,
            confidence=0.9,
            embedding=embedding,
        )
        expired = storage.create_long_term_memory(
            user_id="user_001",
            group_id=self.group.id,
            session_id=self.session.id,
            content="用户在一次旧会话中询问了YBCO",
            memory_type="episodic",
            subtype="interaction",
            memory_key=None,
            content_hash="expired",
            importance=0.6,
            confidence=0.9,
            embedding=embedding,
            expires_at="2020-01-01T00:00:00",
        )
        storage.create_long_term_memory(
            user_id="another_user",
            group_id=self.group.id,
            session_id=self.session.id,
            content="另一用户关注YBCO",
            memory_type="semantic",
            subtype="research",
            memory_key="research.ybco",
            content_hash="other-user",
            importance=0.9,
            confidence=0.9,
            embedding=embedding,
        )

        hits = LongTermMemoryService(embedding_model=_FakeEmbedding()).search(
            "YBCO失超", user_id="user_001", group_id=self.group.id
        )

        self.assertEqual([hit.memory.content for hit in hits], ["用户关注YBCO线圈失超检测"])
        self.assertEqual(storage.get_long_term_memory(expired.id).is_active, 0)

    def test_graph_contains_and_executes_three_memory_nodes(self) -> None:
        events: list[str] = []

        class _FakeMemoryService:
            def search(self, _query, **_scope):
                events.append("memory_retrieve")
                return []

            def extract(self, _message, **_context):
                events.append("memory_extract")
                return []

            def update(self, _candidates, **_scope):
                events.append("memory_update")
                return MemoryUpdateResult()

        document = Document(page_content="证据", metadata={"filename": "p.pdf"})
        with (
            patch.object(graph, "_get_long_term_memory_service", return_value=_FakeMemoryService()),
            patch.object(graph, "get_retriever", return_value=object()),
            patch.object(
                graph,
                "_invoke_retriever",
                side_effect=lambda *_args, **_kwargs: (
                    events.append("retrieve") or [document]
                ),
            ),
            patch.object(
                graph,
                "_postprocess_documents",
                side_effect=lambda _question, docs: docs,
            ),
            patch.object(graph, "_prepare_documents_for_generation", side_effect=lambda _q, docs, **_kw: docs),
            patch.object(graph, "_generate_answer", side_effect=lambda *_args, **_kw: (events.append("generate") or "回答")),
        ):
            result = graph.build_graph().invoke(
                {
                    "question": "问题",
                    "group_id": self.group.id,
                    "user_id": "user_001",
                    "session_id": self.session.id,
                    "history": [],
                }
            )

        self.assertEqual(
            events,
            ["memory_retrieve", "retrieve", "generate", "memory_extract", "memory_update"],
        )
        self.assertEqual(result["answer"], "回答")


if __name__ == "__main__":
    unittest.main()
