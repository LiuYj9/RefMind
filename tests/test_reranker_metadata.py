"""Synthetic regressions for provenance across reranker backend switches."""

import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from langchain_core.documents import Document

from refmind.config import settings
from refmind.rag import reranker


class RerankerMetadataTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(
                page_content=f"Synthetic text {index} with  preserved spacing.",
                metadata={
                    "doc_id": index,
                    "chunk_id": f"chunk-{index}",
                    "retrieval_channels": {"bm25": {"rank": index}},
                    "retrieval_origin": {"direct_rank": index},
                    "rerank_score": 0.99,
                    "rerank_backend": "old_backend",
                    "rerank_degraded": True,
                    "rerank_reason": "old_reason",
                },
            )
            for index in (1, 2, 3)
        ]
        self.original = copy.deepcopy(self.docs)

    def tearDown(self):
        self.assertEqual(self.docs, self.original)

    def invoke(self, *, enabled=True, sdk=True, cross=None, embedding=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(settings, "rerank_enabled", enabled))
            stack.enter_context(patch.object(settings, "dashscope_api_key", "fixture-key"))
            stack.enter_context(patch.object(settings, "rerank_model", "fixture-reranker"))
            available = stack.enter_context(patch.object(reranker, "dashscope_sdk_available", return_value=sdk))
            direct = stack.enter_context(patch.object(reranker, "_dashscope_rerank"))
            fallback = stack.enter_context(patch.object(reranker, "_embedding_rerank"))
            for backend, response in ((direct, cross), (fallback, embedding)):
                if isinstance(response, BaseException):
                    backend.side_effect = response
                elif response is None:
                    backend.side_effect = AssertionError("Unexpected synthetic backend call")
                else:
                    backend.return_value = response
            result = reranker.rerank("Synthetic query", self.docs, top_n=2)
        return result, available, direct, fallback

    def assert_sources(self, result, indices):
        expected = [self.docs[index] for index in indices]
        self.assertEqual([document.page_content for document in result], [document.page_content for document in expected])
        for document, source in zip(result, expected, strict=True):
            self.assertIsNot(document, source)
            for field in ("doc_id", "chunk_id", "retrieval_channels", "retrieval_origin"):
                self.assertEqual(document.metadata[field], source.metadata[field])

    def test_disabled_clears_old_score_and_reason_without_backend_calls(self):
        result, available, direct, fallback = self.invoke(enabled=False)
        self.assert_sources(result, [0, 1])
        for backend in (available, direct, fallback):
            backend.assert_not_called()
        for document in result:
            self.assertEqual(document.metadata["rerank_backend"], "disabled")
            self.assertIs(document.metadata["rerank_degraded"], False)
            self.assertNotIn("rerank_score", document.metadata)
            self.assertNotIn("rerank_reason", document.metadata)

    def test_cross_encoder_success_clears_old_reason_and_preserves_real_zero_score(self):
        scored = [(0.7654321, self.docs[1]), (0.0, self.docs[0]), (-0.2, self.docs[2])]
        result, _, direct, fallback = self.invoke(cross=scored)
        self.assert_sources(result, [1, 0])
        direct.assert_called_once_with("Synthetic query", self.docs)
        fallback.assert_not_called()
        self.assertEqual([document.metadata["rerank_score"] for document in result], [0.7654, 0.0])
        for document in result:
            self.assertEqual(document.metadata["rerank_backend"], "cross_encoder")
            self.assertIs(document.metadata["rerank_degraded"], False)
            self.assertNotIn("rerank_reason", document.metadata)

    def test_embedding_fallback_keeps_current_score_and_current_reason(self):
        scored = [(0.2, self.docs[2]), (-0.13756, self.docs[0]), (-0.9, self.docs[1])]
        for sdk, reason in ((False, "cross_encoder_unavailable"), (True, "TimeoutError")):
            with self.subTest(sdk=sdk):
                result, _, direct, fallback = self.invoke(
                    sdk=sdk, cross=TimeoutError("Synthetic timeout"), embedding=scored
                )
                self.assert_sources(result, [2, 0])
                if sdk:
                    direct.assert_called_once_with("Synthetic query", self.docs)
                else:
                    direct.assert_not_called()
                fallback.assert_called_once_with("Synthetic query", self.docs)
                self.assertEqual([document.metadata["rerank_score"] for document in result], [0.2, -0.1376])
                for document in result:
                    self.assertEqual(document.metadata["rerank_backend"], "embedding_similarity")
                    self.assertIs(document.metadata["rerank_degraded"], True)
                    self.assertEqual(document.metadata["rerank_reason"], reason)

    def test_recall_order_clears_stale_score_and_reports_embedding_error(self):
        result, _, direct, fallback = self.invoke(
            cross=TimeoutError("Synthetic timeout"), embedding=ValueError("Synthetic embedding failure")
        )
        self.assert_sources(result, [0, 1])
        direct.assert_called_once_with("Synthetic query", self.docs)
        fallback.assert_called_once_with("Synthetic query", self.docs)
        for document in result:
            self.assertEqual(document.metadata["rerank_backend"], "recall_order")
            self.assertIs(document.metadata["rerank_degraded"], True)
            self.assertEqual(document.metadata["rerank_reason"], "ValueError")
            self.assertNotIn("rerank_score", document.metadata)

    def test_empty_successful_cross_encoder_response_does_not_trigger_fallback(self):
        result, _, direct, fallback = self.invoke(cross=[])
        self.assertEqual(result, [])
        direct.assert_called_once_with("Synthetic query", self.docs)
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
