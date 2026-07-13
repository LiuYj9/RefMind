"""multi-agent 与现有 RAG 图的集成回归测试。"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from refmind.config import settings
from refmind.rag import graph


class _FakeLLM:
    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)

    def invoke(self, _prompt, config=None, **_kwargs):
        return SimpleNamespace(content=self._responses.pop(0))


class _FakeRetriever:
    def invoke(self, query: str):
        return [
            Document(
                page_content=f"evidence:{query}",
                metadata={"chunk_id": query, "filename": "paper.pdf", "page": 1},
            )
        ]


class AgentGraphIntegrationTests(unittest.TestCase):
    def test_system_prompt_uses_compact_fragment_citations(self) -> None:
        prompt = graph.get_system_prompt()

        self.assertIn("[片段1]", prompt)
        self.assertIn("不要在正文重复完整长文件名", prompt)

    def test_multi_query_result_keeps_postprocessed_evidence_and_diagnostics(self) -> None:
        llm = _FakeLLM(
            json.dumps({"subqueries": ["方法", "结果"]}, ensure_ascii=False),
            json.dumps({"answer": "审校答案"}, ensure_ascii=False),
        )

        def postprocess(_question: str, documents: list[Document]):
            return list(reversed(documents))

        with (
            patch.object(settings, "multi_agent_max_subqueries", 3),
            patch.object(settings, "multi_agent_max_workers", 2),
            patch.object(settings, "evidence_review_enabled", False),
            patch.object(settings, "answer_review_enabled", True),
            patch.object(graph, "get_retriever", return_value=_FakeRetriever()),
            patch.object(graph, "get_llm", return_value=llm),
            patch.object(graph, "_postprocess_documents", side_effect=postprocess),
            patch.object(graph, "_generate_answer", return_value="草稿答案"),
        ):
            result = graph._answer_with_agents("比较方法和结果", 7, [])

        self.assertEqual(result["answer"], "审校答案")
        self.assertEqual(result["queries"], ["方法", "结果"])
        self.assertTrue(result["used_multi_agent"])
        self.assertEqual(
            [doc.metadata["chunk_id"] for doc in result["documents"]],
            ["结果", "方法"],
        )

    def test_empty_library_skips_planner_and_returns_grounded_refusal(self) -> None:
        with (
            patch.object(graph, "get_retriever", return_value=None),
            patch.object(graph, "get_llm") as llm_factory,
        ):
            result = graph._answer_with_agents("任意问题", 99, [])

        self.assertEqual(result["answer"], graph.NO_CONTEXT_REPLY)
        self.assertEqual(result["documents"], [])
        llm_factory.assert_not_called()

    def test_unexpected_pipeline_error_returns_actionable_result(self) -> None:
        with (
            patch.object(settings, "multi_agent_enabled", True),
            patch.object(
                graph,
                "_answer_with_agents",
                side_effect=RuntimeError("provider unavailable"),
            ),
        ):
            result = graph.answer_question("问题", 1)

        self.assertTrue(result["service_failed"])
        self.assertTrue(result["degraded"])
        self.assertEqual(result["answer"], graph.SERVICE_ERROR_REPLY)
        self.assertIn("provider unavailable", result["warnings"][0])


if __name__ == "__main__":
    unittest.main()
