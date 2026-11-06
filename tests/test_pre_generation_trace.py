import json
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from evaluation.evaluator import _safe_trace
from refmind.rag import evidence_bundle, graph
from refmind.rag.evidence_bundle import EvidenceBundle


QUESTION = "What was measured?"
GATES = ["target_before_review", "evidence_review", "evidence_guard", "generation_budget", "target_before_generation"]


def sid(number):
    return f"library:7:{number}"


def document(number):
    return Document(page_content=f"Synthetic passage {number}.", metadata={
        "group_id": 7, "doc_id": number, "filename": f"paper-{number}.pdf", "chunk_id": f"chunk-{number}",
        "evidence_level": "full_text", "page": 2, "paragraph_index": number, "nested": {"value": "original"},
    })


@contextmanager
def mocked_pipeline(review=None, budget=None):
    with ExitStack() as stack:
        model = stack.enter_context(patch.object(graph, "get_llm", return_value=Mock(name="offline_model")))
        reviewer = stack.enter_context(patch.object(graph.EvidenceReviewAgent, "review",
            side_effect=lambda question, documents: (documents if review is None else review(documents), None)))
        generator = stack.enter_context(patch.object(graph, "_generate_answer",
            side_effect=lambda question, documents, *args, **kwargs: "Synthetic answer." if documents else graph.NO_CONTEXT_REPLY))
        stack.enter_context(patch.object(graph.AnswerReviewAgent, "review", side_effect=lambda q, docs, draft: (draft, None)))
        stack.enter_context(patch.object(graph, "_run_text_hook", side_effect=lambda hook, text, **kwargs: text))
        stack.enter_context(patch.object(graph, "token_counter", return_value=(len, "synthetic")))
        stack.enter_context(patch.object(evidence_bundle, "token_counter", return_value=(len, "synthetic")))
        stack.enter_context(patch.multiple(graph.settings, context_max_tokens=8000, model_context_tokens=32768,
            answer_token_reserve=512, context_body_priority_enabled=False))
        if budget is not None:
            stack.enter_context(patch.object(graph, "enforce_generation_budget",
                side_effect=lambda docs, history, **kwargs: EvidenceBundle(budget(docs), history, {}, [])))
        yield SimpleNamespace(model=model, review=reviewer, generate=generator)


def answer(documents, *, required=(), guard=None):
    description = graph.describe_task(QUESTION, source_ids=[sid(1), sid(2), sid(3)],
                                      required_source_ids=[sid(n) for n in required]).to_dict()
    return graph.answer_from_evidence(QUESTION, documents, task_description=description, evidence_guard=guard)


class PreGenerationTraceTests(unittest.TestCase):
    def gates(self, result, first):
        trace = result["trace"]["pre_generation"]
        self.assertEqual(trace["scope"], "packed_to_generation")
        self.assertEqual(trace["first_blocking_stage"], first)
        self.assertEqual([row["stage"] for row in trace["gates"]], GATES)
        return {row["stage"]: row for row in trace["gates"]}

    def assert_gate(self, row, *, before, after, invoked=True, emptied=False, lost=False):
        self.assertEqual(row, {"stage": row["stage"], "invoked": invoked,
            "before_count": len(before), "after_count": len(after),
            "before_source_ids": [sid(n) for n in before], "after_source_ids": [sid(n) for n in after],
            "emptied": emptied, "required_sources_lost": lost})

    def assert_no_request(self, result, rejection):
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["rejection_stage"], rejection)
        generation = result["trace"]["stages"]["generation_context"]
        self.assertFalse(generation["request_sent"])
        self.assertEqual(generation["documents"], [])

    def test_review_empties_evidence_with_and_without_required_sources(self):
        for required, rejection in (((), "evidence_review"), ((1,), "target_coverage")):
            with self.subTest(required=required), mocked_pipeline(review=lambda docs: []):
                guard = Mock(side_effect=lambda docs: docs)
                result = answer([document(1)], required=required, guard=guard)
                gates = self.gates(result, "evidence_review")
                self.assert_gate(gates["evidence_review"], before=(1,), after=(), emptied=True, lost=bool(required))
                self.assert_gate(gates["evidence_guard"], before=(), after=())
                self.assert_no_request(result, rejection)
                guard.assert_called_once_with([])

    def test_guard_empties_evidence_without_overriding_target_coverage(self):
        for required, rejection in (((), "evidence_guard"), ((1,), "target_coverage")):
            with self.subTest(required=required), mocked_pipeline():
                result = answer([document(1)], required=required, guard=lambda docs: [])
                gates = self.gates(result, "evidence_guard")
                self.assert_gate(gates["evidence_guard"], before=(1,), after=(), emptied=True, lost=bool(required))
                self.assert_no_request(result, rejection)

    def test_review_removes_all_targets_but_leaves_background(self):
        with mocked_pipeline(review=lambda docs: docs[1:]):
            result = answer([document(1), document(2)], required=(1,))
        gates = self.gates(result, "evidence_review")
        self.assert_gate(gates["evidence_review"], before=(1, 2), after=(2,), lost=True)
        self.assert_gate(gates["target_before_generation"], before=(2,), after=(), emptied=True)
        self.assert_no_request(result, "target_coverage")

    def test_guard_leaving_one_required_source_does_not_report_total_target_loss(self):
        with mocked_pipeline():
            result = answer([document(1), document(2), document(3)], required=(1, 2), guard=lambda docs: docs[1:])
        gates = self.gates(result, None)
        self.assert_gate(gates["evidence_guard"], before=(1, 2, 3), after=(2, 3))
        self.assertIsNone(result["rejection_stage"])
        self.assertEqual(result["trace"]["source_coverage"]["missing_source_ids"], [sid(1)])
        self.assertTrue(result["trace"]["stages"]["generation_context"]["request_sent"])

    def test_budget_empties_evidence_after_guard(self):
        with mocked_pipeline(budget=lambda docs: []):
            result = answer([document(1)], guard=lambda docs: docs)
        gates = self.gates(result, "generation_budget")
        self.assert_gate(gates["evidence_guard"], before=(1,), after=(1,))
        self.assert_gate(gates["generation_budget"], before=(1,), after=(), emptied=True)
        self.assert_no_request(result, "packing")

    def test_budget_removes_all_targets_but_leaves_background(self):
        with mocked_pipeline(budget=lambda docs: docs[1:]):
            result = answer([document(1), document(2)], required=(1,))
        gates = self.gates(result, "generation_budget")
        self.assert_gate(gates["generation_budget"], before=(1, 2), after=(2,), lost=True)
        self.assert_gate(gates["target_before_generation"], before=(2,), after=(), emptied=True)
        self.assert_no_request(result, "target_coverage")

    def test_missing_target_before_review_does_not_invoke_reviewer(self):
        with mocked_pipeline() as calls:
            result = answer([document(2)], required=(1,))
        gates = self.gates(result, "target_before_review")
        self.assert_gate(gates["target_before_review"], before=(2,), after=(), emptied=True)
        self.assert_gate(gates["evidence_review"], before=(), after=(), invoked=False)
        self.assert_no_request(result, "target_coverage")
        calls.review.assert_not_called()
        calls.generate.assert_not_called()

    def test_no_guard_snapshot_keeps_documents_and_serializes(self):
        with mocked_pipeline():
            result = answer([document(1)])
        gates = self.gates(result, None)
        self.assert_gate(gates["evidence_guard"], before=(1,), after=(1,), invoked=False)
        serialized = json.loads(json.dumps(_safe_trace(result["trace"])))
        self.assertEqual(serialized["pre_generation"], result["trace"]["pre_generation"])
        for stage, invoked in (("evidence_review_input", True), ("evidence_review_output", True),
                               ("evidence_guard_output", False), ("generation_budget_output", True)):
            snapshot = serialized["stages"][stage]
            self.assertIs(snapshot["invoked"], invoked)
            self.assertEqual(snapshot["documents"][0]["page_content"], "Synthetic passage 1.")

    def test_guard_in_place_change_cannot_rewrite_review_snapshots(self):
        def change_in_place(documents):
            documents[0].page_content = "Changed by guard."
            documents[0].metadata["nested"]["value"] = "changed"
            return documents

        with mocked_pipeline():
            result = answer([document(1)], guard=change_in_place)
        self.gates(result, None)
        stages = result["trace"]["stages"]
        for stage in ("packed", "evidence_review_input", "evidence_review_output"):
            original = stages[stage]["documents"][0]
            self.assertEqual(original.page_content, "Synthetic passage 1.")
            self.assertEqual(original.metadata["nested"]["value"], "original")
        guarded = stages["evidence_guard_output"]["documents"][0]
        self.assertEqual(guarded.page_content, "Changed by guard.")
        self.assertEqual(guarded.metadata["nested"]["value"], "changed")
        self.assertIn("Changed by guard.", stages["generation_context"]["user_prompt"])

    def test_permission_error_still_aborts_without_generation(self):
        guard = Mock(side_effect=PermissionError("synthetic access revoked"))
        with mocked_pipeline() as calls:
            with self.assertRaisesRegex(PermissionError, "synthetic access revoked"):
                answer([document(1)], guard=guard)
        guard.assert_called_once()
        calls.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
