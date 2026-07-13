"""多智能体编排的纯离线单元测试。"""

from __future__ import annotations

import json
import threading
import time
import unittest
from types import SimpleNamespace

from langchain_core.documents import Document

from refmind.agents import (
    AnswerDraft,
    MultiAgentConfig,
    MultiAgentOrchestrator,
)


class FakeLLM:
    """按顺序返回预设响应，不访问网络。"""

    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str, config=None, **kwargs):  # noqa: ANN001
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("没有更多 fake 响应")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response)


def _doc(chunk_id: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "filename": "paper.pdf", "page": 1},
    )


class MultiAgentOrchestrationTests(unittest.TestCase):
    def test_plans_parallel_queries_and_deduplicates_in_query_order(self) -> None:
        planner = FakeLLM(
            json.dumps({"subqueries": ["定义", "方法", "结果"]}, ensure_ascii=False)
        )
        config = MultiAgentConfig(max_workers=2)
        orchestrator = MultiAgentOrchestrator(config, planner_llm=planner)
        active = 0
        max_active = 0
        lock = threading.Lock()

        def retrieve(query: str):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            mapping = {
                "定义": [_doc("a", "A")],
                "方法": [_doc("b", "B"), _doc("a", "重复 A")],
                "结果": [_doc("c", "C")],
            }
            return mapping[query]

        result = orchestrator.run(
            "比较定义、方法和结果",
            retrieve=retrieve,
            answer=lambda _question, docs: "".join(d.page_content for d in docs),
        )

        self.assertEqual(result.queries, ["定义", "方法", "结果"])
        self.assertEqual(
            [d.metadata["chunk_id"] for d in result.documents], ["a", "b", "c"]
        )
        self.assertEqual(result.answer, "ABC")
        self.assertTrue(result.used_multi_agent)
        self.assertLessEqual(max_active, 2)
        self.assertGreaterEqual(max_active, 2)

    def test_planner_failure_degrades_to_original_single_query(self) -> None:
        orchestrator = MultiAgentOrchestrator(
            planner_llm=FakeLLM(RuntimeError("planner down"))
        )
        seen_queries: list[str] = []

        result = orchestrator.run(
            "原问题",
            retrieve=lambda query: seen_queries.append(query) or [_doc("a", "证据")],
            answer=lambda _question, _docs: "原答案",
        )

        self.assertEqual(seen_queries, ["原问题"])
        self.assertEqual(result.answer, "原答案")
        self.assertTrue(result.degraded)
        self.assertFalse(result.used_multi_agent)

    def test_retrieval_exception_is_distinct_from_normal_empty_result(self) -> None:
        orchestrator = MultiAgentOrchestrator(
            planner_llm=FakeLLM('{"subqueries": ["原问题"]}')
        )

        result = orchestrator.run(
            "原问题",
            retrieve=lambda _query: (_ for _ in ()).throw(
                RuntimeError("vector store down")
            ),
            answer=lambda _question, _docs: "无上下文",
        )

        self.assertTrue(result.retrieval_failed)
        self.assertTrue(result.degraded)

    def test_parallel_retrieval_deadline_returns_degraded_result(self) -> None:
        release = threading.Event()
        orchestrator = MultiAgentOrchestrator(
            MultiAgentConfig(retrieval_timeout_seconds=0.01),
            planner_llm=FakeLLM('{"subqueries": ["问题一", "问题二"]}'),
        )

        def blocked_retrieve(_query: str):
            release.wait(timeout=1)
            return [_doc("late", "迟到证据")]

        try:
            result = orchestrator.run(
                "原问题",
                retrieve=blocked_retrieve,
                answer=lambda _question, _docs: "检索失败提示",
            )
        finally:
            release.set()

        self.assertTrue(result.retrieval_failed)
        self.assertTrue(result.degraded)
        self.assertTrue(any("停止等待" in warning for warning in result.warnings))

    def test_all_subquery_failures_retry_original_question(self) -> None:
        planner = FakeLLM('{"subqueries": ["子问题一", "子问题二"]}')
        orchestrator = MultiAgentOrchestrator(planner_llm=planner)
        calls: list[str] = []

        def retrieve(query: str):
            calls.append(query)
            if query != "原问题":
                raise RuntimeError("temporary failure")
            return [_doc("fallback", "回退证据")]

        result = orchestrator.run(
            "原问题", retrieve=retrieve, answer=lambda _q, _d: "回退答案"
        )

        self.assertIn("原问题", calls)
        self.assertEqual(result.answer, "回退答案")
        self.assertEqual(result.documents[0].metadata["chunk_id"], "fallback")
        self.assertTrue(result.degraded)

    def test_single_rewritten_query_with_no_hits_retries_original(self) -> None:
        orchestrator = MultiAgentOrchestrator(
            planner_llm=FakeLLM('{"subqueries": ["改写问题"]}')
        )
        calls: list[str] = []

        def retrieve(query: str):
            calls.append(query)
            return [] if query == "改写问题" else [_doc("base", "原问题证据")]

        result = orchestrator.run(
            "原问题",
            retrieve=retrieve,
            answer=lambda _question, docs: docs[0].page_content,
        )

        self.assertEqual(calls, ["改写问题", "原问题"])
        self.assertEqual(result.answer, "原问题证据")
        self.assertTrue(result.degraded)

    def test_optional_reviewers_have_real_roles_and_safe_fallback(self) -> None:
        planner = FakeLLM('{"subqueries": ["原问题"]}')
        evidence_reviewer = FakeLLM('{"keep": [2]}')
        answer_reviewer = FakeLLM(RuntimeError("reviewer down"))
        orchestrator = MultiAgentOrchestrator(
            MultiAgentConfig(
                enable_evidence_review=True,
                enable_answer_review=True,
            ),
            planner_llm=planner,
            evidence_reviewer_llm=evidence_reviewer,
            answer_reviewer_llm=answer_reviewer,
        )

        result = orchestrator.run(
            "原问题",
            retrieve=lambda _query: [_doc("noise", "噪声"), _doc("proof", "证据")],
            answer=lambda _question, docs: f"草稿：{docs[0].page_content}",
        )

        self.assertEqual(
            [d.metadata["chunk_id"] for d in result.documents], ["proof"]
        )
        self.assertEqual(result.answer, "草稿：证据")
        self.assertTrue(result.degraded)
        self.assertTrue(any("保留原答案" in warning for warning in result.warnings))

    def test_existing_rerank_compression_pipeline_can_run_after_merge(self) -> None:
        planner = FakeLLM('{"subqueries": ["子问题一", "子问题二"]}')
        orchestrator = MultiAgentOrchestrator(planner_llm=planner)
        postprocess_inputs: list[list[str]] = []

        def postprocess(_question: str, docs: list[Document]):
            postprocess_inputs.append([d.metadata["chunk_id"] for d in docs])
            return docs[-1:]

        result = orchestrator.run(
            "原问题",
            retrieve=lambda query: (
                [_doc("a", "A")] if query == "子问题一" else [_doc("b", "B")]
            ),
            postprocess=postprocess,
            answer=lambda _question, docs: docs[0].page_content,
        )

        self.assertEqual(postprocess_inputs, [["a", "b"]])
        self.assertEqual(result.answer, "B")
        self.assertEqual([d.metadata["chunk_id"] for d in result.documents], ["b"])

    def test_prepare_runs_even_when_postprocess_fails(self) -> None:
        orchestrator = MultiAgentOrchestrator(
            planner_llm=FakeLLM('{"subqueries": ["原问题"]}')
        )

        result = orchestrator.run(
            "原问题",
            retrieve=lambda _query: [_doc("raw", "原始证据")],
            postprocess=lambda _question, _docs: (_ for _ in ()).throw(
                RuntimeError("rerank down")
            ),
            prepare=lambda _question, docs: [
                _doc("prepared", docs[0].page_content)
            ],
            answer=lambda _question, docs: docs[0].metadata["chunk_id"],
        )

        self.assertEqual(result.answer, "prepared")
        self.assertEqual(result.documents[0].metadata["chunk_id"], "prepared")
        self.assertTrue(result.degraded)

    def test_unexpected_generation_failure_uses_injected_baseline(self) -> None:
        planner = FakeLLM('{"subqueries": ["子问题一", "子问题二"]}')
        orchestrator = MultiAgentOrchestrator(planner_llm=planner)
        baseline_calls: list[str] = []

        result = orchestrator.run(
            "原问题",
            retrieve=lambda _query: [_doc("a", "证据")],
            answer=lambda _question, _docs: (_ for _ in ()).throw(
                RuntimeError("generation failed")
            ),
            baseline=lambda question: (
                baseline_calls.append(question)
                or AnswerDraft("原流程答案", [_doc("base", "原流程证据")])
            ),
        )

        self.assertEqual(baseline_calls, ["原问题"])
        self.assertEqual(result.answer, "原流程答案")
        self.assertEqual(result.queries, ["原问题"])
        self.assertTrue(result.degraded)


if __name__ == "__main__":
    unittest.main()
