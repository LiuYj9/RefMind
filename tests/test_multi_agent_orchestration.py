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
    canonicalize_retrieval_query,
    INSUFFICIENT_EVIDENCE_REPLY,
    MultiAgentConfig,
    MultiAgentOrchestrator,
)
from refmind.agents.orchestration import PlanningAgent


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
    def test_planner_reuses_result_for_canonical_equivalent_question(self) -> None:
        llm = FakeLLM('{"subqueries": ["稳定改写"]}')
        planner = PlanningAgent(llm, MultiAgentConfig())

        first = planner.plan("同一个问题？")
        second = planner.plan("同一个问题?")

        self.assertEqual(first.queries, ["稳定改写"])
        self.assertEqual(second.queries, first.queries)
        self.assertEqual(len(llm.prompts), 1)

    def test_canonical_anchor_is_always_searched_before_planner_expansions(self) -> None:
        planner = FakeLLM('{"subqueries": ["相关但不精确的改写"]}')
        orchestrator = MultiAgentOrchestrator(
            MultiAgentConfig(max_subqueries=3), planner_llm=planner
        )
        calls: list[str] = []

        def retrieve(query: str) -> list[Document]:
            calls.append(query)
            return (
                [_doc("proof", "直接证据")]
                if query == "降低线圈交流损耗的方法有哪些"
                else [_doc("noise", "相邻主题")]
            )

        result = orchestrator.run(
            "降低线圈交流损耗的方法有哪些？",
            retrieve=retrieve,
            answer=lambda _question, documents: documents[0].page_content,
            review_question="降低线圈交流损耗的方法有哪些？",
            retrieval_anchor="降低线圈交流损耗的方法有哪些？",
            retrieval_expansions=["用户背景扩展"],
        )

        self.assertEqual(calls[0], "降低线圈交流损耗的方法有哪些")
        self.assertEqual(result.documents[0].metadata["chunk_id"], "proof")
        self.assertEqual(
            result.queries,
            ["降低线圈交流损耗的方法有哪些", "相关但不精确的改写", "用户背景扩展"],
        )
        for suffix in ("", "?", "？", "。", " ？  "):
            self.assertEqual(
                canonicalize_retrieval_query("降低线圈交流损耗的方法有哪些" + suffix),
                "降低线圈交流损耗的方法有哪些",
            )

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
        evidence_reviewer = FakeLLM(
            '{"coverage": "direct", "answerable": true, "keep": [2], '
            '"reason": "直接证据"}'
        )
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
        review_prompt = evidence_reviewer.prompts[0]
        self.assertIn("问题类型（定义、存在性、方法、原因/机理、数值、比较/排序", review_prompt)
        self.assertIn("direct 表示直接且范围相符", review_prompt)
        self.assertNotIn("HTS", review_prompt)
        self.assertNotIn("铜盘", review_prompt)

    def test_adjacent_eddy_loss_evidence_cannot_answer_quench_question(self) -> None:
        question = "如何解决 HTS 电机所使用的超导带材容易失超的问题？"
        planner = FakeLLM(
            json.dumps({"subqueries": [question]}, ensure_ascii=False)
        )
        rejection = json.dumps(
            {
                "coverage": "insufficient",
                "answerable": False,
                "keep": [],
                "reason": "证据只讨论涡流损耗，未建立与失超的联系",
            },
            ensure_ascii=False,
        )
        evidence_reviewer = FakeLLM(rejection, rejection)
        orchestrator = MultiAgentOrchestrator(
            MultiAgentConfig(enable_evidence_review=True),
            planner_llm=planner,
            evidence_reviewer_llm=evidence_reviewer,
        )
        answer_document_counts: list[int] = []

        def answer(_question: str, documents: list[Document]) -> str:
            answer_document_counts.append(len(documents))
            return INSUFFICIENT_EVIDENCE_REPLY if not documents else "错误答案"

        result = orchestrator.run(
            "检索增强：用户研究高温超导电机\n本轮问题：" + question,
            review_question=question,
            retrieve=lambda _query: [
                _doc(
                    "shielding",
                    "铜盘电磁屏蔽可降低纹波磁场在 NI HTS 绕组中引起的涡流和损耗。",
                )
            ],
            answer=answer,
        )

        self.assertEqual(result.answer, INSUFFICIENT_EVIDENCE_REPLY)
        self.assertEqual(result.documents, [])
        self.assertEqual(answer_document_counts, [0])
        self.assertEqual(result.rejection_stage, "evidence_review")
        self.assertIn(question, evidence_reviewer.prompts[0])
        self.assertIn(
            "未经证据建立的因果链也不能补齐",
            evidence_reviewer.prompts[0],
        )

    def test_eddy_loss_method_can_answer_ac_loss_existential_question(self) -> None:
        question = "现有论文中，是否有降低 HTS 线圈交流损耗的方法？"
        planner = FakeLLM(
            json.dumps({"subqueries": [question]}, ensure_ascii=False)
        )
        evidence_reviewer = FakeLLM(
            json.dumps(
                {
                    "coverage": "scoped",
                    "answerable": True,
                    "keep": [1],
                    "reason": "证据给出了特定绕组和工况下的降损方法",
                },
                ensure_ascii=False,
            )
        )
        orchestrator = MultiAgentOrchestrator(
            MultiAgentConfig(enable_evidence_review=True),
            planner_llm=planner,
            evidence_reviewer_llm=evidence_reviewer,
        )

        result = orchestrator.run(
            question,
            retrieve=lambda _query: [
                _doc(
                    "shielding",
                    "铜盘电磁屏蔽可降低纹波磁场在 NI HTS 转子绕组中引起的涡流和损耗。",
                )
            ],
            answer=lambda _question, documents: (
                "有。针对 NI HTS 转子绕组，可采用铜盘电磁屏蔽降低纹波场引起的涡流和损耗。"
                if documents
                else INSUFFICIENT_EVIDENCE_REPLY
            ),
        )

        self.assertNotEqual(result.answer, INSUFFICIENT_EVIDENCE_REPLY)
        self.assertEqual(
            [document.metadata["chunk_id"] for document in result.documents],
            ["shielding"],
        )
        self.assertIn(
            "存在性/举例：一个 direct 或 scoped 实例即可支持",
            evidence_reviewer.prompts[0],
        )
        self.assertIn("答案必须保留更窄的对象、条件、指标和适用范围", evidence_reviewer.prompts[0])

    def test_answer_review_refusal_clears_irrelevant_references(self) -> None:
        planner = FakeLLM('{"subqueries": ["失超"]}')
        rejection = json.dumps(
            {
                "coverage": "insufficient",
                "answerable": False,
                "answer": INSUFFICIENT_EVIDENCE_REPLY,
                "reason": "草稿回答的是涡流损耗而非失超",
            },
            ensure_ascii=False,
        )
        answer_reviewer = FakeLLM(rejection, rejection)
        orchestrator = MultiAgentOrchestrator(
            MultiAgentConfig(enable_answer_review=True),
            planner_llm=planner,
            answer_reviewer_llm=answer_reviewer,
        )

        result = orchestrator.run(
            "如何解决超导带材失超？",
            retrieve=lambda _query: [_doc("adjacent", "铜盘降低涡流损耗")],
            answer=lambda _question, _documents: "采用铜盘屏蔽可解决失超",
        )

        self.assertEqual(result.answer, INSUFFICIENT_EVIDENCE_REPLY)
        self.assertEqual(result.documents, [])
        self.assertFalse(result.degraded)
        self.assertEqual(result.rejection_stage, "answer_review")

    def test_single_evidence_rejection_requires_confirmation(self) -> None:
        rejected = json.dumps(
            {
                "coverage": "insufficient",
                "answerable": False,
                "keep": [],
                "reason": "首次判断不足",
            },
            ensure_ascii=False,
        )
        accepted = json.dumps(
            {
                "coverage": "direct",
                "answerable": True,
                "keep": [1],
                "reason": "复核后确认直接支持",
            },
            ensure_ascii=False,
        )
        reviewer = FakeLLM(rejected, accepted)
        orchestrator = MultiAgentOrchestrator(
            MultiAgentConfig(enable_evidence_review=True),
            planner_llm=FakeLLM('{"subqueries": ["问题"]}'),
            evidence_reviewer_llm=reviewer,
        )

        result = orchestrator.run(
            "问题",
            retrieve=lambda _query: [_doc("proof", "直接证据")],
            answer=lambda _question, documents: documents[0].page_content,
        )

        self.assertEqual(result.answer, "直接证据")
        self.assertEqual(len(reviewer.prompts), 2)
        self.assertTrue(result.degraded)
        self.assertTrue(any("二次复核" in item for item in result.warnings))

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
