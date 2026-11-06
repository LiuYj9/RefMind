"""Offline regression of the opt-in shared research retrieval path."""
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch
import time

from langchain_core.documents import Document
from refmind.config import settings
from refmind.rag import research_pipeline as pipeline, research_queries as queries, retrieval, scientific_retrieval


def document(doc_id=1, chunk="e1", version="v1", text=None, channels=None):
    return Document(page_content=text or chunk, metadata={"group_id": 7, "doc_id": doc_id,
        "version": version, "file_sha256": f"hash{doc_id}", "chunk_id": chunk, "chunk_index": 1,
        "section": "Methods", "filename": f"{doc_id}.pdf",
        "retrieval_channels": channels or {"bm25": {"rank": 1}, "vector": {"rank": 2}},
        "retrieval_fusion": {"method": "weighted_rrf", "rank": 1, "c": 60,
            "weights": {"bm25": .5, "vector": .5}, "channels": ["bm25", "vector"]}})


class ResearchMultiQueryTests(TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.rows = {i: SimpleNamespace(id=i, group_id=7, status="ready", index_version="v1",
            file_sha256=f"hash{i}", paper_title=f"Paper {i}", filename=f"{i}.pdf") for i in (1, 2, 3)}
        self.stack.enter_context(patch.object(pipeline.storage, "get_document", side_effect=self.rows.get))
        self.stack.enter_context(patch.object(pipeline.storage, "list_documents", return_value=list(self.rows.values())))
        for name, value in {"research_multi_query_enabled": True, "research_max_queries": 3,
                            "research_query_workers": 3, "research_query_timeout": .5}.items():
            self.stack.enter_context(patch.object(settings, name, value))
        self.stack.enter_context(patch("refmind.rag.document_processor.load_group_documents", return_value=[]))

    def retrieve(self, fn, plan=None, **kwargs):
        retriever = Mock()
        retriever.invoke.side_effect = fn
        with patch.object(retrieval, "get_retriever", return_value=retriever) as factory:
            result = pipeline.retrieve_research_candidates("original", 7, allowed_doc_ids=[1, 2],
                query_plan=plan or {"queries": ["original", "branch"]}, **kwargs)
        return result, factory, retriever

    def test_flag_off_retains_baseline_entry_and_ignores_plan(self):
        with patch.object(settings, "research_multi_query_enabled", False), \
             patch.object(pipeline, "retrieve_for_task", return_value=[document()]) as old, \
             patch.object(queries, "plan_research_queries") as planner:
            result = pipeline.retrieve_research_candidates("original", 7, allowed_doc_ids=[1, 2],
                query_plan={"queries": ["injected"]})
        old.assert_called_once_with(7, "original", selected_doc_ids=(1, 2), task_type="method")
        planner.assert_not_called()
        self.assertEqual(result["queries"], ["original"])
        self.assertIsNone(result["scope_snapshot"])

    def test_original_first_three_max_shared_scope_and_deterministic_merge(self):
        def invoke(query):
            if query == "original":
                time.sleep(.015)
            return [document(chunk=query)]
        result, factory, retriever = self.retrieve(invoke, {"queries": ["branch", "branch2", "branch3", "original"]})
        self.assertEqual(result["queries"], ["original", "branch", "branch2"])
        factory.assert_called_once_with(7, selected_doc_ids=(1, 2))
        self.assertEqual([d.page_content for d in result["documents"]], result["queries"])
        self.assertEqual(retriever.invoke.call_count, 3)

    def test_cross_query_merge_preserves_query_ranks_and_distinct_source(self):
        first = document()
        copy_before = deepcopy(first.metadata)
        second = document(channels={"vector": {"rank": 1}})
        result, _, _ = self.retrieve(lambda q: [first, document(2)] if q == "original" else [second])
        docs = result["documents"]
        self.assertEqual(len(docs), 2)
        self.assertEqual(first.metadata, copy_before)
        runs = docs[0].metadata["retrieval_runs"]
        self.assertEqual([x["query_id"] for x in runs], ["q1", "q2"])
        self.assertIsNone(runs[0]["rrf_score"])
        self.assertIsNone(runs[1]["rrf_score"])
        self.assertEqual(runs[0]["fusion"]["weights"], {"bm25": .5, "vector": .5})
        self.assertEqual(runs[1]["channels"], {"vector": {"rank": 1}})
        self.assertIsNone(runs[1]["raw_scores"])
        self.assertNotIn("bm25", result["trace"]["stages"])
        self.assertEqual(result["trace"]["retrieval_counts"], {"queries": 2, "query_hits": 3, "direct_merged": 2, "expanded": 2})

    def test_partial_error_uses_success_all_error_is_service_failure(self):
        def partial(q):
            if q == "branch":
                raise RuntimeError("fake endpoint secret must not escape")
            return [document()]
        result, _, _ = self.retrieve(partial)
        self.assertTrue(result["degraded"])
        self.assertFalse(result["retrieval_failed"])
        self.assertEqual(len(result["documents"]), 1)
        self.assertNotIn("secret", repr(result["trace"]))
        result, _, _ = self.retrieve(lambda q: (_ for _ in ()).throw(RuntimeError("offline")))
        self.assertTrue(result["retrieval_failed"])
        self.assertEqual(result["documents"], [])

    def test_success_empty_is_not_service_failure(self):
        result, _, _ = self.retrieve(lambda q: [])
        self.assertFalse(result["retrieval_failed"])
        self.assertEqual([q["status"] for q in result["trace"]["queries"]], ["empty", "empty"])

    def test_timeout_is_partial_and_does_not_wait_for_slow_query(self):
        def invoke(q):
            if q == "branch":
                time.sleep(.12)
            return [document()]
        with patch.object(settings, "research_query_timeout", .015):
            started = time.perf_counter()
            result, _, _ = self.retrieve(invoke)
            elapsed = time.perf_counter() - started
        self.assertLess(elapsed, .1)
        self.assertEqual(result["trace"]["queries"][1]["status"], "timeout")
        self.assertFalse(result["retrieval_failed"])
        self.assertEqual(len(result["documents"][0].metadata["retrieval_runs"]), 1)
        time.sleep(.13)  # Let the fake worker finish before patched dependencies are removed.

    def test_scope_wrong_hash_and_version_are_filtered_before_merge(self):
        bad_hash = document(chunk="bad_hash")
        bad_hash.metadata["file_sha256"] = "wrong"
        result, _, _ = self.retrieve(lambda q: [document(), document(3), document(version="old"), bad_hash])
        self.assertEqual([d.metadata["chunk_id"] for d in result["documents"]], ["e1"])
        self.assertEqual(result["trace"]["queries"][0]["snapshot_rejected_count"], 3)

    def test_version_change_during_query_cannot_mix_ingestions(self):
        def invoke(q):
            self.rows[1].index_version = "v2"
            return [document(), document(version="v2")]
        result, _, _ = self.retrieve(invoke)
        self.assertEqual(result["documents"], [])
        self.assertTrue(result["scope_changed"])
        self.assertTrue(result["retrieval_failed"])

    def test_explicit_empty_selection_skips_planning_and_retriever(self):
        with patch.object(queries, "plan_research_queries") as planner, patch.object(retrieval, "get_retriever") as factory:
            result = pipeline.retrieve_research_candidates("original", 7, selected_doc_ids=[], allowed_doc_ids=[1, 2])
        planner.assert_not_called()
        factory.assert_not_called()
        self.assertEqual(result["documents"], [])
        self.assertFalse(result["retrieval_failed"])

    def test_all_failed_requests_never_reach_generation(self):
        with patch.object(queries, "plan_research_queries", return_value={"queries": ["original"]}), \
             patch.object(retrieval, "get_retriever", side_effect=RuntimeError("offline")), \
             patch("refmind.rag.graph.answer_from_evidence") as answer:
            result = pipeline.answer_research_question("original", 7, selected_doc_ids=[1])
        answer.assert_not_called()
        self.assertEqual(result["status"], "system_error")
        self.assertTrue(result["service_failed"])

    def test_expansion_happens_once_after_merge_and_keeps_runs(self):
        with patch.object(scientific_retrieval, "expand_research_candidates", wraps=scientific_retrieval.expand_research_candidates) as expand:
            result, _, _ = self.retrieve(lambda q: [document()])
        expand.assert_called_once()
        self.assertEqual(len(expand.call_args.args[1]), 1)
        self.assertEqual(len(result["documents"][0].metadata["retrieval_runs"]), 2)

    def test_planner_failure_and_measurement_change_keep_original(self):
        llm = Mock()
        llm.invoke.side_effect = RuntimeError("fake-secret")
        result = queries.plan_research_queries("原问题 77 K", planner_llm=llm)
        self.assertEqual(result["queries"], ["原问题 77 K"])
        self.assertTrue(result["degraded"])
        self.assertNotIn("fake-secret", str(result))
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content='{"subqueries":["条件 65 K", "方法 77 K"]}')
        result = queries.plan_research_queries("原问题 77 K", planner_llm=llm)
        self.assertEqual(result["queries"], ["原问题 77 K", "方法 77 K"])
        self.assertEqual(result["rejected_branches"], 1)

    def test_final_wrapper_reranks_once_against_original_and_keeps_scope(self):
        with patch.object(queries, "plan_research_queries", return_value={"queries": ["original", "branch"]}), \
             patch.object(retrieval, "get_retriever", return_value=Mock(invoke=Mock(return_value=[document()]))), \
             patch.object(pipeline, "rerank", side_effect=lambda q, docs, **kw: docs) as rerank:
            result = pipeline.retrieve_research_evidence("original", 7, selected_doc_ids=[1], allowed_doc_ids=[1, 2])
        rerank.assert_called_once()
        self.assertEqual(rerank.call_args.args[0], "original")
        self.assertEqual(rerank.call_args.kwargs["top_n"], 1)
        self.assertEqual(result["trace"]["scope"]["required_source_ids"], ["library:7:1"])

    def test_missing_scores_are_not_invented(self):
        doc = document()
        doc.metadata.pop("retrieval_channels")
        doc.metadata.pop("retrieval_fusion")
        result, _, _ = self.retrieve(lambda q: [doc])
        run = result["documents"][0].metadata["retrieval_runs"][0]
        self.assertIsNone(run["rrf_score"])
        self.assertIsNone(run["raw_scores"])
        self.assertFalse(result["trace"]["retrieval_channels"]["available"])


if __name__ == "__main__":
    main()
