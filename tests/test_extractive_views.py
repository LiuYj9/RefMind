"""Offline behavioural tests for request-local extractive evidence views."""
from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from refmind.rag.extractive_views import (
    OMISSION_MARKER, build_extractive_views, make_embedding_scorer,
    validate_extractive_view,
)

SENTENCES = [
    "本段介绍研究讨论的起点。",
    "下面补充一般背景说明。",
    "本段介绍相关领域的早期发展以及常见的写作结构，并把多种背景资料按叙述顺序列出，供读者了解文章的整体安排和章节衔接方式。",
    "另一段详细介绍行文安排，包括引言如何展开、论述如何过渡、附录怎样组织以及结尾如何回顾，主要作用是向读者说明文章的结构。",
    "接下来还包含一段较长的历史回顾和一般性说明，内容只是帮助读者了解文章的组织过程，与正在提取的具体研究结论之间保持相当距离。",
    "下面进入相关结论。",
    "铜屏蔽在研究工况下减小了线圈感生涡流。",
    "这一说明与前句相连。",
    "最后附带编辑性的阅读提示，重新概述文章采用的写作顺序和过渡安排，帮助读者从结构层面了解整篇文章的行文组织。",
]
TEXT = "".join(SENTENCES)


def document(text=TEXT, *, doc_id=1, chunk_id="chunk-a", **metadata):
    return Document(page_content=text, metadata={
        "group_id": 4, "doc_id": doc_id, "chunk_id": chunk_id, "chunk_index": 2,
        "version": "v1", "filename": f"paper-{doc_id}.pdf", "content_type": "paragraph",
        "page": 8, "nested": {"unchanged": [1, 2]}, **metadata,
    })


def focused(query, sentences):
    return [0.95 if "铜屏蔽" in s else 0.02 for s in sentences]


focused.score_type = "deterministic_offline_fixture"


class ExtractiveViewsTests(unittest.TestCase):
    def test_extracts_lossless_spans_and_preserves_identity_and_input(self):
        original = document()
        snapshot = deepcopy(original)
        result = build_extractive_views("铜屏蔽 降损", [original], scorer=focused)
        view, record = result.documents[0], result.records[0]
        self.assertEqual(record["status"], "extracted")
        self.assertEqual(view.metadata["evidence_view"], "extractive_text")
        self.assertTrue(view.metadata["view_id"].startswith("ev:"))
        self.assertEqual(view.metadata["selected_spans"], record["selected_spans"])
        self.assertEqual(view.metadata["compression_status"], "extracted")
        self.assertIsInstance(view.metadata["coverage_limit"], str)
        self.assertEqual(view.metadata["offset_unit"], "unicode_code_point")
        self.assertEqual(view.metadata["interval"], "half_open")
        self.assertIn(OMISSION_MARKER, view.page_content)
        self.assertEqual(original, snapshot)
        self.assertIsNot(view, original)
        for key, value in snapshot.metadata.items():
            self.assertEqual(view.metadata[key], value)
        self.assertEqual(view.metadata["source_text"], TEXT)
        self.assertEqual(record["source_chunk_hash"], hashlib.sha256(TEXT.encode("utf-8")).hexdigest())
        self.assertTrue(validate_extractive_view(TEXT, view.page_content, record))
        for kind in ("selected_spans", "omitted_spans"):
            for span in record[kind]:
                self.assertEqual(TEXT[span["start"]:span["end"]], span["quote"])
        view.metadata["nested"]["unchanged"].append(3)
        self.assertEqual(original.metadata["nested"]["unchanged"], [1, 2])

    def test_unicode_codepoints_and_whitespace_are_not_rewritten(self):
        text = "\n\n🧲e\u0301\r\n" + TEXT
        result = build_extractive_views("铜屏蔽 降损", [document(text)], scorer=focused)
        record = result.records[0]
        self.assertTrue(validate_extractive_view(text, result.documents[0].page_content, record))
        self.assertEqual(record["offset_unit"], "unicode_code_point")
        self.assertTrue(record["selected_spans"][0]["quote"].startswith("\n\n🧲e\u0301\r\n"))
        self.assertEqual(record["source_chunk_hash"], hashlib.sha256(text.encode("utf-8")).hexdigest())

    def test_numeric_unit_negative_object_and_condition_sentences_survive(self):
        sentences = list(SENTENCES)
        sentences[2] = "样品由铜制成，在77 K和50 Hz条件下测试，损耗下降3.2%。"
        sentences[4] = "这并不意味着所有工况下的系统总损耗都会下降。"
        text = "".join(sentences)
        result = build_extractive_views("铜屏蔽 降损", [document(text)], scorer=focused)
        view = result.documents[0].page_content
        self.assertIn(sentences[2], view)
        self.assertIn(sentences[4], view)
        self.assertIn(sentences[5], view)
        self.assertIn(sentences[7], view)

    def test_highly_similar_different_conditions_are_never_deduplicated(self):
        documents = [document(TEXT + "工况为77 K。", chunk_id="one"),
                     document(TEXT + "工况为20 K。", chunk_id="two")]
        result = build_extractive_views("铜屏蔽 降损", documents, scorer=focused)
        self.assertEqual(len(result.documents), 2)
        self.assertEqual([d.metadata["chunk_id"] for d in result.documents], ["one", "two"])
        self.assertIn("77 K", result.documents[0].page_content)
        self.assertIn("20 K", result.documents[1].page_content)

    def test_only_plain_prose_is_processed(self):
        inputs = [document(content_type=kind) for kind in
                  ("table", "equation", "figure", "figure_caption", "list_item", "reference", "abstract")]
        inputs += [document(is_atomic=True), document("<table><tr><td>text</td></tr></table>"),
                   document("$$E = mc^2$$"), document("| A | B |\n| -- | -- |")]
        def forbidden(*args):
            self.fail("structured evidence must not be scored")
        result = build_extractive_views("query", inputs, scorer=forbidden)
        for original, output, record in zip(inputs, result.documents, result.records):
            self.assertEqual(original.page_content, output.page_content)
            self.assertEqual(record["status"], "unchanged_not_plain_text")

    def test_hard_block_limit_retains_unprocessed_sources(self):
        inputs = [document(doc_id=i, chunk_id=str(i)) for i in range(15)]
        calls = []
        def score(query, sentences):
            calls.append(len(sentences))
            return focused(query, sentences)
        result = build_extractive_views("铜屏蔽 降损", inputs, scorer=score, max_blocks=999)
        self.assertEqual(len(calls), 12)
        self.assertEqual(len(result.documents), 15)
        self.assertEqual(result.coverage_limit["not_processed_due_to_block_limit"], [12, 13, 14])
        self.assertFalse(result.coverage_limit["full_evidence_coverage_claimed"])
        for i in (12, 13, 14):
            self.assertEqual(result.records[i]["status"], "fallback_block_limit")
            self.assertEqual(result.documents[i].page_content, TEXT)

    def test_sentence_limit_falls_back_whole_block(self):
        text = "".join("这是一句完整正文。" for _ in range(25))
        def forbidden(*args):
            self.fail("over-limit block must not be scored or truncated")
        result = build_extractive_views("完整正文", [document(text)], scorer=forbidden, max_sentences=999)
        self.assertEqual(result.records[0]["status"], "fallback_sentence_limit")
        self.assertEqual(result.documents[0].page_content, text)
        self.assertEqual(result.coverage_limit["sentence_limit_indices"], [0])

    def test_comparison_and_summary_round_robin_sources_but_keep_output_order(self):
        inputs = ([document(doc_id=1, chunk_id=f"a{i}") for i in range(10)]
                  + [document(doc_id=2, chunk_id="b"), document(doc_id=3, chunk_id="c")])
        for kind in ("comparison", "summary"):
            result = build_extractive_views("铜屏蔽 降损", inputs, scorer=focused, task_kind=kind, max_blocks=3)
            self.assertEqual(result.coverage_limit["selected_processing_indices"], [0, 10, 11])
            self.assertEqual([d.metadata["chunk_id"] for d in result.documents],
                             [d.metadata["chunk_id"] for d in inputs])
        result = build_extractive_views("铜屏蔽 降损", inputs, scorer=focused, max_blocks=3)
        self.assertEqual(result.coverage_limit["selected_processing_indices"], [0, 1, 2])

    def test_service_failure_retains_chunk_and_does_not_leak_error_payload(self):
        def unavailable(*args):
            raise RuntimeError("sensitive-provider-payload")
        result = build_extractive_views("铜屏蔽 降损", [document()], scorer=unavailable)
        self.assertEqual(result.documents[0].page_content, TEXT)
        self.assertEqual(result.records[0]["status"], "fallback_scoring_error")
        self.assertEqual(result.records[0]["reason"], "RuntimeError")
        self.assertNotIn("evidence_view", result.documents[0].metadata)
        self.assertNotIn("view_id", result.documents[0].metadata)
        self.assertNotIn("sensitive-provider-payload", str(result.audit_dict()))

    def test_scoring_failure_stops_further_service_attempts_in_request(self):
        calls = []
        def unavailable(*args):
            calls.append(1)
            raise TimeoutError("provider detail")
        result = build_extractive_views("query", [document(doc_id=i) for i in range(4)], scorer=unavailable)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.coverage_limit["scoring_attempts"], 1)
        self.assertEqual(result.records[1]["status"], "fallback_scoring_unavailable")
        self.assertTrue(all(d.page_content == TEXT for d in result.documents))

    def test_invalid_scores_fall_back(self):
        for value in (float("nan"), float("inf"), 2.0, True):
            def invalid(query, sentences):
                return [value] * len(sentences)
            result = build_extractive_views("query", [document()], scorer=invalid)
            self.assertEqual(result.records[0]["status"], "fallback_scoring_error")
            self.assertEqual(result.documents[0].page_content, TEXT)
        result = build_extractive_views("query", [document()], scorer=lambda q, s: [0.8])
        self.assertEqual(result.records[0]["status"], "fallback_scoring_error")

    def test_weak_match_and_ambiguous_omissions_fall_back(self):
        weak = build_extractive_views("query", [document()], scorer=lambda q, s: [0.1] * len(s))
        self.assertEqual(weak.records[0]["status"], "fallback_ambiguous_relevance")
        def ambiguous(query, sentences):
            scores = focused(query, sentences)
            scores[3] = 0.30
            return scores
        result = build_extractive_views("query", [document()], scorer=ambiguous)
        self.assertEqual(result.records[0]["status"], "fallback_ambiguous_context")
        self.assertEqual(result.documents[0].page_content, TEXT)

    def test_no_gain_returns_whole_chunk(self):
        short = "介绍。背景。铜屏蔽减小涡流。"
        result = build_extractive_views("铜屏蔽 降损", [document(short)], scorer=focused)
        self.assertEqual(result.records[0]["status"], "unchanged_no_gain")
        self.assertEqual(result.documents[0].page_content, short)

    def test_validation_failure_falls_back(self):
        with patch("refmind.rag.extractive_views.validate_extractive_view", return_value=False):
            result = build_extractive_views("铜屏蔽 降损", [document()], scorer=focused)
        self.assertEqual(result.records[0]["status"], "fallback_validation_failed")
        self.assertEqual(result.documents[0].page_content, TEXT)
        self.assertTrue(validate_extractive_view(TEXT, TEXT, result.records[0]))

    def test_validator_rejects_tampered_quotes_offsets_hashes_and_duplicates(self):
        result = build_extractive_views("铜屏蔽 降损", [document()], scorer=focused)
        good = result.records[0]
        bad = deepcopy(good); bad["selected_spans"][0]["quote"] += "changed"
        self.assertFalse(validate_extractive_view(TEXT, result.documents[0].page_content, bad))
        bad = deepcopy(good); bad["selected_spans"][0]["start"] = True
        self.assertFalse(validate_extractive_view(TEXT, result.documents[0].page_content, bad))
        bad = deepcopy(good); bad["selected_spans"].append(bad["selected_spans"][0])
        self.assertFalse(validate_extractive_view(TEXT, result.documents[0].page_content, bad))
        bad = deepcopy(good); bad["source_chunk_hash"] = "invalid"
        self.assertFalse(validate_extractive_view(TEXT, result.documents[0].page_content, bad))

    def test_default_is_explicitly_offline_lexical(self):
        with patch("refmind.llm.get_embedding_model", side_effect=AssertionError("no network")):
            result = build_extractive_views("铜屏蔽 涡流", [document()])
        self.assertEqual(result.score_type, "offline_lexical_overlap")
        self.assertEqual(result.records[0]["score_purpose"], "sentence_selection_only_not_answer_confidence")

    def test_embedding_scorer_is_lazy_caches_query_and_preserves_failures(self):
        class Embeddings:
            calls = 0
            batches = 0
            def embed_query(self, query):
                self.calls += 1
                return [1.0, 0.0]
            def embed_documents(self, sentences):
                self.batches += 1
                return [[1.0, 0.0] if "铜屏蔽" in s else [0.0, 1.0] for s in sentences]
        model = Embeddings()
        scorer = make_embedding_scorer(model)
        self.assertEqual(model.calls, 0)
        result = build_extractive_views("铜屏蔽 降损", [document(), document(doc_id=2)], scorer=scorer)
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.batches, 2)
        self.assertEqual(result.score_type, "sentence_query_cosine")
        model.embed_documents = lambda sentences: [[0.0]] * len(sentences)
        result = build_extractive_views("铜屏蔽 降损", [document()], scorer=scorer)
        self.assertEqual(result.records[0]["status"], "fallback_scoring_error")

    def test_empty_input_and_empty_query_and_zero_processing(self):
        result = build_extractive_views("", [])
        self.assertEqual(result.documents, [])
        for text in ("", " \n "):
            result = build_extractive_views("query", [document(text)])
            self.assertEqual(result.documents[0].page_content, text)
            self.assertTrue(validate_extractive_view(text, text, result.records[0]))
        result = build_extractive_views("", [document()])
        self.assertEqual(result.records[0]["status"], "fallback_empty_query")
        result = build_extractive_views("query", [document()], max_blocks=0, task_kind="comparison")
        self.assertEqual(result.coverage_limit["selected_processing_indices"], [])
        self.assertEqual(result.documents[0].page_content, TEXT)

    def test_limits_reject_boolean_or_noninteger(self):
        for kwargs in ({"max_blocks": True}, {"max_sentences": 2.5}):
            with self.assertRaises(ValueError):
                build_extractive_views("query", [document()], **kwargs)


if __name__ == "__main__":
    unittest.main()
