import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from evaluation.dataset import GoldenItem, GoldenSet
from evaluation.evaluator import evaluate_items
from refmind.agents.orchestration import AnswerReviewAgent, INSUFFICIENT_EVIDENCE_REPLY, MultiAgentConfig
from refmind.rag import graph


def payload(answerable=True, *, coverage=None, answer="Reviewed answer.", **extras):
    if coverage is None:
        coverage = "direct" if answerable else "insufficient"
    return {"answerable": answerable, "coverage": coverage, "answer": answer, **extras}


def response(value):
    return SimpleNamespace(content=json.dumps(value))


def documents():
    return [Document(page_content="The measured value was 12 A at 77 K.",
                     metadata={"group_id": 7, "doc_id": 1, "filename": "paper.pdf", "page": 2})]


def validated(phase, answerable, coverage, reason):
    return {"phase": phase, "status": "validated", "answerable": answerable,
            "coverage": coverage, "reason": reason}


class AnswerReviewTraceTests(unittest.TestCase):
    def reviewer(self, *responses, confirm=True, enabled=True):
        llm = Mock()
        llm.invoke.side_effect = [value if isinstance(value, Exception) else response(value) for value in responses]
        return AnswerReviewAgent(llm, MultiAgentConfig(enable_answer_review=enabled, confirm_refusal=confirm)), llm

    def test_success_preserves_only_validated_metadata(self):
        reviewer, llm = self.reviewer(payload(coverage="scoped", reason="Only one condition is supported.",
                                             raw_payload="do not trace"))
        self.assertEqual(reviewer.review("Question", documents(), "Draft"), ("Reviewed answer.", None))
        llm.invoke.assert_called_once()
        self.assertEqual(reviewer.last_review, {"status": "completed", "attempts": [
            validated("initial", True, "scoped", "Only one condition is supported.")]})

    def test_refusal_confirmation_toggle_and_both_valid_results(self):
        reject = payload(False, reason="Initial missing condition.")
        for confirm, rest, answer in (
            (False, (), INSUFFICIENT_EVIDENCE_REPLY),
            (True, (payload(False, reason="Still missing."),), INSUFFICIENT_EVIDENCE_REPLY),
            (True, (payload(reason="Confirmed supporting passage."),), "Reviewed answer."),
        ):
            with self.subTest(confirm=confirm, answer=answer):
                reviewer, llm = self.reviewer(reject, *rest, confirm=confirm)
                result, warning = reviewer.review("Question", documents(), "Draft")
                self.assertEqual(result, answer)
                self.assertEqual(bool(warning), answer != INSUFFICIENT_EVIDENCE_REPLY)
                self.assertEqual(llm.invoke.call_count, 1 + len(rest))
                expected = [validated("initial", False, "insufficient", "Initial missing condition.")]
                expected.extend(validated("confirmation", row["answerable"], row["coverage"], row["reason"]) for row in rest)
                self.assertEqual(reviewer.last_review, {"status": "completed", "attempts": expected})

    def test_second_failure_preserves_first_valid_attempt_and_old_warning(self):
        reviewer, llm = self.reviewer(payload(False, reason="First reason."), RuntimeError("private provider detail"))
        answer, warning = reviewer.review("Question", documents(), "Draft")
        self.assertEqual(answer, "Draft")
        self.assertIn("private provider detail", warning)
        self.assertEqual(llm.invoke.call_count, 2)
        self.assertEqual(reviewer.last_review, {"status": "failed", "attempts": [
            validated("initial", False, "insufficient", "First reason."),
            {"phase": "confirmation", "status": "failed", "error_type": "RuntimeError"}]})

    def test_invalid_existing_fields_do_not_admit_diagnostic_reason(self):
        for invalid in (payload(answerable="true", reason="unvalidated"),
                        payload(False, coverage="direct", reason="unvalidated"),
                        payload(True, coverage="insufficient", reason="unvalidated"),
                        payload(answer=" ", reason="unvalidated")):
            with self.subTest(invalid=invalid):
                reviewer, llm = self.reviewer(invalid)
                answer, warning = reviewer.review("Question", documents(), "Draft")
                self.assertEqual(answer, "Draft")
                self.assertTrue(warning)
                llm.invoke.assert_called_once()
                self.assertEqual(reviewer.last_review, {"status": "failed", "attempts": [
                    {"phase": "initial", "status": "failed", "error_type": "ValueError"}]})

    def test_optional_bad_or_long_reason_does_not_change_return_or_requests(self):
        for extras, expected in (({}, None), ({"reason": ["untrusted array"]}, None),
                                 ({"reason": "r" * 2000}, "r" * 1500)):
            with self.subTest(extras=list(extras)):
                reviewer, llm = self.reviewer(payload(**extras))
                self.assertEqual(reviewer.review("Question", documents(), "Draft"), ("Reviewed answer.", None))
                llm.invoke.assert_called_once()
                self.assertEqual(reviewer.last_review["attempts"], [validated("initial", True, "direct", expected)])

    def test_skip_states_and_reuse_clear_previous_observation(self):
        reviewer, llm = self.reviewer(payload(reason="First call."))
        reviewer.review("Question", documents(), "Draft")
        previous = reviewer.last_review
        self.assertEqual(reviewer.review("Question", [], "New draft"), ("New draft", None))
        self.assertEqual(reviewer.last_review, {"status": "no_documents", "attempts": []})
        self.assertEqual(previous["attempts"], [validated("initial", True, "direct", "First call.")])
        llm.invoke.assert_called_once()

        disabled, unused = self.reviewer(enabled=False)
        self.assertEqual(disabled.review("Question", documents(), "Draft"), ("Draft", None))
        self.assertEqual(disabled.last_review, {"status": "disabled", "attempts": []})
        unused.invoke.assert_not_called()

        unavailable = AnswerReviewAgent(None, MultiAgentConfig(enable_answer_review=True))
        self.assertEqual(unavailable.review("Question", documents(), "Draft"),
                         ("Draft", "未注入答案审校模型，已保留原答案"))
        self.assertEqual(unavailable.last_review, {"status": "no_model", "attempts": []})

    def test_graph_and_synthetic_draft_report_preserve_the_same_review_trace(self):
        item = GoldenItem(id="synthetic", question="What was measured?", ground_truth="12 A", evidence=())
        dataset = GoldenSet(corpus=(), items=(item,), page_tolerance=0, manifest=(), dataset_hash="synthetic", schema_version=3)
        llm = Mock()
        llm.invoke.return_value = response(payload(reason="The passage supports this value.", raw_payload="not serialized"))
        with (patch.object(graph, "get_llm", return_value=llm),
              patch.object(graph.settings, "evidence_review_enabled", False),
              patch.object(graph.settings, "answer_review_enabled", True),
              patch.object(graph, "_generate_answer", return_value="Draft"),
              patch.object(graph, "_run_text_hook", side_effect=lambda _hook, text, **kwargs: text)):
            report = evaluate_items(dataset, mode="answer", allow_draft=True,
                                    component=lambda question: graph.answer_from_evidence(question, documents()))
        llm.invoke.assert_called_once()
        self.assertFalse(report["review_gate"]["official"])
        row = report["items"][0]
        self.assertEqual(row["status"], "ok")
        self.assertNotIn("answer_review", row)
        trace = row["trace"]["answer_review"]
        self.assertEqual(json.loads(json.dumps(trace)), {"status": "completed", "attempts": [
            validated("initial", True, "direct", "The passage supports this value.")]})

    def test_graph_handles_no_documents_and_legacy_reviewer_without_observation(self):
        with patch.object(graph, "get_llm") as unused:
            result = graph.answer_from_evidence("Question", [])
        unused.assert_not_called()
        self.assertEqual(result["trace"]["answer_review"], {"status": "no_documents", "attempts": []})

        class PassReviewer:
            def __init__(self, *_args):
                pass

            def review(self, _question, _documents, draft):
                return draft, None

        with (patch.object(graph, "get_llm", return_value=Mock()),
              patch.object(graph.settings, "evidence_review_enabled", False),
              patch.object(graph, "_generate_answer", return_value="Draft"),
              patch.object(graph, "_run_text_hook", side_effect=lambda _hook, text, **kwargs: text),
              patch.object(graph, "AnswerReviewAgent", PassReviewer)):
            result = graph.answer_from_evidence("Question", documents())
        self.assertEqual(result["trace"]["answer_review"], {"status": "not_observed", "attempts": []})


if __name__ == "__main__":
    unittest.main()
