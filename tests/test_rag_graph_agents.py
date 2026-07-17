"""multi-agent 与现有 RAG 图的集成回归测试。"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from refmind.config import settings
from refmind.rag import graph
from refmind.services.academic_search import AcademicSearchResult


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
    def test_baseline_retrieval_keeps_canonical_question_as_rerank_anchor(self) -> None:
        calls: list[str] = []
        rerank_questions: list[str] = []

        def invoke(_retriever, query: str, _group_id: int, **_kwargs):
            calls.append(query)
            chunk_id = "direct" if "上下文" not in query else "expanded"
            return [
                Document(
                    page_content=chunk_id,
                    metadata={"chunk_id": chunk_id},
                )
            ]

        def postprocess(question: str, documents: list[Document]):
            rerank_questions.append(question)
            return documents

        with (
            patch.object(graph, "get_retriever", return_value=object()),
            patch.object(graph, "_invoke_retriever", side_effect=invoke),
            patch.object(graph, "_postprocess_documents", side_effect=postprocess),
        ):
            result = graph._retrieve_node(
                {
                    "question": "如何降低损耗？",
                    "retrieval_question": "如何降低损耗\n用户上下文：偏好实验论文",
                    "group_id": 7,
                }
            )

        self.assertEqual(calls[0], "如何降低损耗")
        self.assertEqual(rerank_questions, ["如何降低损耗"])
        self.assertEqual(
            [item.metadata["chunk_id"] for item in result["documents"]],
            ["direct", "expanded"],
        )

    def test_system_prompt_requires_complete_paper_citations(self) -> None:
        prompt = graph.get_system_prompt()

        self.assertIn("[文献2《论文题名》，第5页", prompt)
        self.assertIn("自动转换成论文式 [1][2]", prompt)
        self.assertIn("不要自行缩写为 [片段1]", prompt)
        self.assertIn("证据必须支持同一关系", prompt)
        self.assertIn("数值问题需要数值证据", prompt)
        self.assertIn("比较/排序问题需要覆盖各比较对象和同一指标", prompt)
        self.assertNotIn("HTS", prompt)
        self.assertNotIn("铜盘", prompt)

    def test_academic_prompt_limits_evidence_to_temporary_abstracts(self) -> None:
        prompt = graph.get_system_prompt("academic")

        self.assertIn("外部学术检索临时证据", prompt)
        self.assertIn("不等同于已读取论文全文", prompt)
        self.assertIn("伪造页码", prompt)
        self.assertIn("[GS文献2《论文题名》", prompt)
        self.assertIn("外部不可信数据", prompt)
        self.assertNotIn("仅基于用户上传的论文PDF", prompt)

    def test_academic_search_reranks_candidates_and_uses_external_mode(self) -> None:
        candidate = Document(
            page_content="论文题名：Paper\n摘要：direct evidence",
            metadata={
                "evidence_origin": "academic_search",
                "evidence_level": "abstract",
                "paper_title": "Paper",
                "provider": "Semantic Scholar",
                "external_url": "https://example.org/paper",
            },
        )
        search_result = AcademicSearchResult(
            query="HTS motor AC loss reduction",
            provider="semantic_scholar",
            documents=(candidate,),
        )
        with (
            patch.object(
                graph,
                "_plan_academic_search_query",
                return_value=("HTS motor AC loss reduction", None),
            ),
            patch(
                "refmind.services.academic_search.search_academic_papers",
                return_value=search_result,
            ),
            patch.object(graph, "rerank", return_value=[candidate]) as reranker,
            patch.object(graph, "_generate_answer", return_value="外部证据回答") as generate,
        ):
            result = graph._answer_with_academic_search("如何降低交流损耗？", [])

        self.assertTrue(result["used_academic_search"])
        self.assertEqual(result["evidence_source"], "academic")
        self.assertEqual(result["documents"], [candidate])
        reranker.assert_called_once()
        self.assertEqual(generate.call_args.kwargs["evidence_mode"], "academic")

    def test_academic_mode_falls_back_to_local_only_after_empty_external_search(self) -> None:
        local_document = Document(page_content="本地证据", metadata={"chunk_id": "local"})
        academic_result = {
            "answer": graph.ACADEMIC_NO_CONTEXT_REPLY,
            "documents": [],
            "academic_documents": [],
            "academic_query": "academic query",
            "academic_provider": "semantic_scholar",
            "academic_providers": ["Semantic Scholar"],
            "academic_search_attempted": True,
            "academic_search_failed": False,
            "used_academic_search": False,
            "evidence_source": "none",
            "warnings": ("没有可用摘要",),
        }
        local_result = {
            "answer": "本地回答",
            "documents": [local_document],
            "queries": ["问题"],
            "used_multi_agent": False,
            "degraded": False,
            "retrieval_failed": False,
            "rejection_stage": None,
            "warnings": (),
        }
        with (
            patch.object(graph, "_memory_retrieve_node", return_value={"long_term_memories": []}),
            patch.object(graph, "_memory_extract_node", return_value={"memory_candidates": []}),
            patch.object(graph, "_memory_update_node", return_value={}),
            patch.object(graph, "_answer_with_academic_search", return_value=academic_result),
            patch.object(graph, "_answer_from_local_library", return_value=local_result) as local,
        ):
            result = graph.answer_question(
                "问题",
                1,
                retrieval_mode="academic",
            )

        local.assert_called_once()
        self.assertEqual(result["answer"], "本地回答")
        self.assertEqual(result["evidence_source"], "local")
        self.assertTrue(result["academic_search_attempted"])
        self.assertFalse(result["used_academic_search"])
        self.assertTrue(result["degraded"])

    def test_multi_query_result_keeps_postprocessed_evidence_and_diagnostics(self) -> None:
        llm = _FakeLLM(
            json.dumps({"subqueries": ["方法", "结果"]}, ensure_ascii=False),
            json.dumps(
                {
                    "coverage": "direct",
                    "answerable": True,
                    "answer": "审校答案",
                    "reason": "与问题直接对齐",
                },
                ensure_ascii=False,
            ),
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

        self.assertTrue(result["answer"].startswith("审校答案"))
        self.assertIn("### 参考来源", result["answer"])
        self.assertEqual(result["queries"], ["比较方法和结果", "方法", "结果"])
        self.assertTrue(result["used_multi_agent"])
        self.assertEqual(
            [doc.metadata["chunk_id"] for doc in result["documents"]],
            ["结果", "方法", "比较方法和结果"],
        )

    def test_repeated_question_drops_previous_turn_from_generation_history(self) -> None:
        history = [
            HumanMessage(content="降低线圈损耗的方法有哪些"),
            AIMessage(content=graph.NO_CONTEXT_REPLY),
            HumanMessage(content="请使用中文"),
            AIMessage(content="好的"),
        ]

        filtered = graph._remove_repeated_question_turns(
            history, "降低线圈损耗的方法有哪些？"
        )

        self.assertEqual(
            [message.content for message in filtered],
            ["请使用中文", "好的"],
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
