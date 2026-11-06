import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from refmind.rag import graph


class RefusalStageTraceTests(unittest.TestCase):
    def run_answer(self, *, draft="Generated answer.", hook=None, review=None,
                   review_enabled=True, review_fails=False, evidence_rejected=False):
        question = "What was measured?"
        evidence = Document(page_content="The measured value was 12 A at 77 K.",
                            metadata={"group_id": 7, "doc_id": 1, "filename": "paper.pdf", "page": 2})
        task = graph.describe_task(question, source_ids=["library:7:1"], required_source_ids=[]).to_dict()
        llm = Mock()
        llm.invoke.side_effect = RuntimeError("synthetic reviewer failure") if review_fails else AssertionError("unexpected reviewer call")
        with ExitStack() as stack:
            stack.enter_context(patch.object(graph, "get_llm", return_value=llm))
            stack.enter_context(patch.object(graph.settings, "answer_review_enabled", review_enabled))
            stack.enter_context(patch.object(graph.EvidenceReviewAgent, "review",
                                           side_effect=lambda q, ds: ([] if evidence_rejected else ds, None)))
            generation = stack.enter_context(patch.object(graph, "_generate_answer",
                side_effect=lambda q, ds, *args, **kwargs: draft if ds else graph.NO_CONTEXT_REPLY))
            stack.enter_context(patch.object(graph, "_run_text_hook",
                                           side_effect=lambda key, answer, **kwargs: answer if hook is None else hook))
            if review is not None:
                stack.enter_context(patch.object(graph.AnswerReviewAgent, "review", return_value=(review, None)))
            result = graph.answer_from_evidence(question, [evidence], task_description=task)
        if review_fails:
            llm.invoke.assert_called_once()
        else:
            llm.invoke.assert_not_called()
        snapshot = result["trace"]["stages"]["generation_context"]
        self.assertEqual(snapshot["request_sent"], not evidence_rejected)
        self.assertEqual(snapshot["documents"], generation.call_args.args[1])
        return result

    def assert_refusal_origin(self, result, stage):
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["rejection_stage"], stage)
        self.assertEqual(result["trace"]["answer_stages"]["after_citation_validation"], result["answer"])
        self.assertEqual(result["trace"]["stages"]["final_evidence"]["documents"], [])

    def test_generation_refusal_survives_an_unchanged_review(self):
        result = self.run_answer(draft=graph.NO_CONTEXT_REPLY, review=graph.NO_CONTEXT_REPLY)
        self.assert_refusal_origin(result, "generation")

    def test_generation_refusal_survives_disabled_review(self):
        result = self.run_answer(draft=graph.NO_CONTEXT_REPLY, review_enabled=False)
        self.assert_refusal_origin(result, "generation")
        self.assertEqual(result["review_status"], "unreviewed")

    def test_generation_refusal_survives_failed_review(self):
        result = self.run_answer(draft=graph.NO_CONTEXT_REPLY, review_fails=True)
        self.assert_refusal_origin(result, "generation")
        self.assertTrue(any("审校失败" in warning for warning in result["warnings"]))

    def test_after_generate_hook_introduces_refusal(self):
        result = self.run_answer(hook=graph.NO_CONTEXT_REPLY, review=graph.NO_CONTEXT_REPLY)
        self.assert_refusal_origin(result, "after_generate_hook")

    def test_answer_review_introduces_refusal(self):
        result = self.run_answer(review=graph.NO_CONTEXT_REPLY)
        self.assert_refusal_origin(result, "answer_review")

    def test_citation_validation_introduces_refusal_and_preserves_prior_snapshots(self):
        unsupported = "Review [文献999《Unknown》，第8页]"
        result = self.run_answer(hook="Hook answer.", review=unsupported)
        self.assert_refusal_origin(result, "citation_validation")
        trace = result["trace"]
        self.assertEqual(trace["answer_stages"], {"generated_draft": "Generated answer.",
            "after_generate_hook": "Hook answer.", "after_answer_review": unsupported,
            "after_citation_validation": graph.NO_CONTEXT_REPLY})
        self.assertEqual(len(trace["stages"]["generation_context"]["documents"]), 1)
        self.assertIn("12 A at 77 K", trace["stages"]["generation_context"]["user_prompt"])

    def test_recovered_refusal_gets_new_origin_if_later_stage_refuses(self):
        result = self.run_answer(draft=graph.NO_CONTEXT_REPLY, hook="Recovered answer.", review=graph.NO_CONTEXT_REPLY)
        self.assert_refusal_origin(result, "answer_review")
        self.assertEqual(result["trace"]["answer_stages"]["after_generate_hook"], "Recovered answer.")

    def test_recovered_refusal_has_no_final_origin(self):
        result = self.run_answer(draft=graph.NO_CONTEXT_REPLY, review="Recovered answer.")
        self.assertEqual(result["status"], "answered")
        self.assertIsNone(result["rejection_stage"])
        self.assertEqual(result["trace"]["answer_stages"]["after_citation_validation"], result["answer"])

    def test_pre_generation_evidence_refusal_keeps_existing_origin(self):
        result = self.run_answer(evidence_rejected=True)
        self.assert_refusal_origin(result, "evidence_review")
        self.assertEqual(result["trace"]["stages"]["generation_context"]["documents"], [])


if __name__ == "__main__":
    unittest.main()
