"""HTTP adapter contracts for the shared candidate path, without DB or models."""
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from refmind.web import jobs
from refmind.rag import graph, research_pipeline, research_queries, retrieval
from tests.test_research_multiquery import document


class WebResearchMultiQueryTests(TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.runner = object.__new__(jobs.JobRunner)
        self.task = {"id": "offline", "owner_id": 1, "payload": {"session_id": 5,
            "question": "Explain the method.", "group_id": 7, "mode": "library", "selected_doc_ids": [1]}}
        self.doc = document()
        self.row = SimpleNamespace(id=1, group_id=7, status="ready", index_version="v1",
            file_sha256="hash1", paper_title="Paper 1", filename="1.pdf")
        self.stack.enter_context(patch.object(jobs.storage, "get_document", return_value=self.row))
        self.stack.enter_context(patch.object(jobs.storage, "list_messages", return_value=[]))
        self.stack.enter_context(patch.object(jobs.store, "session_access", return_value={"id": 5, "group_id": 7}))
        self.access = self.stack.enter_context(patch.object(jobs.store, "library_access"))
        self.stack.enter_context(patch.object(self.runner, "_progress"))
        self.stack.enter_context(patch.object(self.runner, "_check", return_value=self.task))
        self.stack.enter_context(patch.object(self.runner, "_save_answer", side_effect=lambda task, output: output))
        self.stack.enter_context(patch.object(retrieval, "filter_ready_documents", side_effect=lambda group, docs, selected=None: docs))
        self.recalled = {"documents": [self.doc], "queries": ["Explain the method.", "branch"],
            "warnings": [], "degraded": False, "retrieval_failed": False,
            "scope_snapshot": ((7, 1, "v1", "hash1"),),
            "trace": {"pipeline": "research", "stages": {"raw": {"documents": [self.doc]}}, "timings": {},
                "queries": [{"query_id": "q1", "text": "Explain the method.", "status": "ok", "documents": [self.doc]}]}}
        self.candidates = self.stack.enter_context(patch.object(research_pipeline, "retrieve_research_candidates", side_effect=lambda *a, **kw: deepcopy(self.recalled)))
        self.rerank = self.stack.enter_context(patch.object(research_pipeline, "run_research_rerank", side_effect=lambda query, docs: docs))
        self.answer = self.stack.enter_context(patch.object(graph, "answer_from_evidence", return_value={
            "answer": "fixture answer", "status": "answered", "documents": [self.doc], "warnings": [],
            "trace": {"stages": {"generation_context": {"documents": [self.doc]}}}}))

    def test_library_uses_shared_helper_one_original_rerank_and_trace(self):
        output = self.runner._answer(self.task)
        self.candidates.assert_called_once_with("Explain the method.", 7, selected_doc_ids=[1], task_type="method")
        self.rerank.assert_called_once_with("Explain the method.", [self.doc])
        self.assertTrue(self.answer.call_args.kwargs["apply_research_enhancements"])
        self.assertIn("generation_context", output["trace"]["stages"])
        self.assertEqual(output["trace"]["queries"][0]["status"], "ok")
        self.assertEqual(output["trace"]["queries"][0]["documents"][0]["text"], self.doc.page_content)

    def test_all_failed_skips_generation_and_reports_service_error(self):
        self.recalled.update(documents=[], retrieval_failed=True, degraded=True, warnings=["所有查询调用失败"])
        output = self.runner._answer(self.task)
        self.rerank.assert_not_called()
        self.answer.assert_not_called()
        self.assertEqual(output["status"], "system_error")
        self.assertTrue(output["service_failed"])
        self.assertTrue(output["degraded"])

    def test_partial_failure_keeps_answer_and_both_warning_sources(self):
        self.recalled.update(degraded=True, warnings=["补充查询超时"])
        self.answer.return_value["warnings"] = ["回答范围已限定"]
        # Existing graph observations must not be overwritten by retrieval traces.
        self.answer.return_value["trace"]["stages"]["raw"] = {"graph_owned": True}
        output = self.runner._answer(self.task)
        self.assertEqual(output["status"], "answered")
        self.assertEqual(output["warnings"], ["补充查询超时", "回答范围已限定"])
        self.assertTrue(output["degraded"])
        self.assertEqual(output["trace"]["stages"]["raw"], {"graph_owned": True})

    def test_generation_guard_rechecks_permission_and_frozen_version(self):
        observed = []
        def answer(question, docs, **kwargs):
            self.assertEqual(kwargs["evidence_guard"](docs), docs)
            self.row.index_version = "v2"
            observed.extend(kwargs["evidence_guard"](docs))
            return {"answer": "scope changed", "status": "insufficient", "documents": []}
        self.answer.side_effect = answer
        output = self.runner._answer(self.task)
        self.assertEqual(observed, [])
        self.assertEqual(output["evidence"], [])
        self.assertGreaterEqual(self.access.call_count, 4)


if __name__ == "__main__":
    main()
