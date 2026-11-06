"""Synthetic origin contracts; patch all retrieval/storage collaborators, never call models."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from refmind.rag.scientific_retrieval import retrieve_for_task


def chunk(index, *, doc_id=101, version="v1", section="Methods", kind="text", direct=False):
    metadata = {"doc_id": doc_id, "group_id": 7, "version": version, "chunk_id": f"chunk-{index}",
                "chunk_index": index, "section": section, "content_type": kind,
                "fixture_tags": ["synthetic", "not-a-real-paper"]}
    if direct:
        metadata["_retrieval_identity"] = f"fixture-direct:{doc_id}:{version}:{index}"
    return Document(page_content=f"Synthetic source {doc_id} {version} chunk {index}.", metadata=metadata)


def signature(document):
    return document.metadata["doc_id"], document.metadata["version"], document.metadata["chunk_index"]


def parent(document, rank, offset):
    return {key: document.metadata[key] for key in ("doc_id", "version", "chunk_id", "chunk_index")} | {
        "direct_rank": rank, "offset": offset}


class RetrievalOriginContracts(unittest.TestCase):
    def retrieve(self, direct, scoped, *, task="method", expand=True, allowed=(101, 202)):
        retriever = Mock()
        retriever.invoke.return_value = direct
        def authorized(_group_id, values, selected):
            return [doc for doc in values if selected is None or doc.metadata["doc_id"] in selected]
        with patch("refmind.rag.retrieval.get_retriever", return_value=retriever), \
             patch("refmind.rag.document_processor.load_group_documents", return_value=scoped), \
             patch("refmind.rag.retrieval.filter_ready_documents", side_effect=authorized):
            return retrieve_for_task(7, "synthetic query", selected_doc_ids=allowed,
                                     task_type=task, expand_neighbors=expand)

    def test_later_direct_hit_first_seen_as_neighbor_keeps_identity_rank_and_order(self):
        direct = [chunk(1, direct=True), chunk(2, direct=True)]
        result = self.retrieve(direct, [chunk(index) for index in range(4)])
        self.assertEqual([doc.metadata["chunk_index"] for doc in result], [1, 0, 2, 3])
        self.assertEqual([doc.page_content for doc in result], [chunk(index).page_content for index in (1, 0, 2, 3)])
        later = next(doc for doc in result if doc.metadata["chunk_index"] == 2)
        self.assertEqual(later.metadata.get("_retrieval_identity"), direct[1].metadata["_retrieval_identity"])
        self.assertEqual(later.metadata["retrieval_origin"], {"direct_rank": 2,
                         "neighbor_of": [parent(direct[0], 1, 1)], "section_coverage": False})
        self.assertIs(later.metadata["retrieval_neighbor"], False)

    def test_overlapping_neighbor_records_all_direct_parents_once(self):
        direct = [chunk(1, direct=True), chunk(3, direct=True)]
        result = self.retrieve(direct, [chunk(index) for index in range(5)])
        self.assertEqual([doc.metadata["chunk_index"] for doc in result], [1, 0, 2, 3, 4])
        self.assertEqual([doc.page_content for doc in result], [chunk(index).page_content for index in (1, 0, 2, 3, 4)])
        overlap = next(doc for doc in result if doc.metadata["chunk_index"] == 2)
        self.assertEqual(overlap.metadata["retrieval_origin"], {"direct_rank": None,
                         "neighbor_of": [parent(direct[0], 1, 1), parent(direct[1], 2, -1)], "section_coverage": False})
        self.assertIs(overlap.metadata["retrieval_neighbor"], True)

    def test_neighbor_parent_boundaries_do_not_cross_paper_version_or_section(self):
        direct = [chunk(1, version="v2", direct=True)]
        scoped = [chunk(0, version="v1"), chunk(0, version="v2", section="Results"),
                  chunk(1, version="v2"), chunk(2, version="v2"), chunk(0, doc_id=202, version="v2")]
        result = self.retrieve(direct, scoped, allowed=(101,))
        self.assertEqual([signature(doc) for doc in result], [(101, "v2", 1), (101, "v2", 2)])
        self.assertEqual([doc.page_content for doc in result], [chunk(index, version="v2").page_content for index in (1, 2)])
        self.assertEqual(result[1].metadata["retrieval_origin"]["neighbor_of"], [parent(direct[0], 1, 1)])

    def test_origin_annotation_does_not_mutate_input_documents(self):
        direct = [chunk(1, direct=True), chunk(2, direct=True)]
        scoped = [chunk(index) for index in range(4)]
        before = [(doc.page_content, copy.deepcopy(doc.metadata)) for doc in direct + scoped]
        self.retrieve(direct, scoped)
        self.assertEqual([(doc.page_content, doc.metadata) for doc in direct + scoped], before)

    def test_summary_coverage_marks_sections_and_preserves_direct_identity(self):
        direct = [chunk(10, section="Results", direct=True)]
        scoped = [chunk(0), chunk(10, section="Results"), chunk(20, section="References"), chunk(0, doc_id=202)]
        result = self.retrieve(direct, scoped, task="summary")
        self.assertEqual([signature(doc) for doc in result], [(101, "v1", 0), (202, "v1", 0), (101, "v1", 10)])
        self.assertEqual([doc.page_content for doc in result], [chunk(0).page_content, chunk(0, doc_id=202).page_content, chunk(10).page_content])
        for document in result:
            self.assertIs(document.metadata["retrieval_origin"]["section_coverage"], True)
            self.assertEqual(document.metadata["retrieval_origin"]["neighbor_of"], [])
            self.assertIs(document.metadata["retrieval_neighbor"], False)
        direct_result = result[-1]
        self.assertEqual(direct_result.metadata["retrieval_origin"]["direct_rank"], 1)
        self.assertEqual(direct_result.metadata["_retrieval_identity"], direct[0].metadata["_retrieval_identity"])
        self.assertIsNone(result[0].metadata["retrieval_origin"]["direct_rank"])

    def test_summary_direct_reference_is_not_mislabeled_as_section_coverage(self):
        reference = chunk(20, section="References", direct=True)
        result = self.retrieve([reference], [chunk(0), chunk(10, section="Results"), chunk(20, section="References")], task="summary")
        self.assertEqual([doc.metadata["chunk_index"] for doc in result], [0, 10, 20])
        self.assertEqual([doc.page_content for doc in result], [chunk(index).page_content for index in (0, 10, 20)])
        self.assertEqual(result[-1].metadata["retrieval_origin"], {"direct_rank": 1, "neighbor_of": [], "section_coverage": False})
        self.assertIs(result[-1].metadata["retrieval_neighbor"], False)

    def test_no_neighbor_expansion_keeps_direct_order_and_provenance(self):
        direct = [chunk(3, direct=True), chunk(1, direct=True)]
        result = self.retrieve(direct, [chunk(index) for index in range(5)], expand=False)
        self.assertEqual([doc.metadata["chunk_index"] for doc in result], [3, 1])
        self.assertEqual([doc.page_content for doc in result], [chunk(index).page_content for index in (3, 1)])
        self.assertEqual([doc.metadata["retrieval_origin"] for doc in result], [
            {"direct_rank": 1, "neighbor_of": [], "section_coverage": False},
            {"direct_rank": 2, "neighbor_of": [], "section_coverage": False}])

    def test_atomic_direct_blocks_do_not_gain_neighbors(self):
        for kind in ("table", "equation", "figure", "figure_caption"):
            with self.subTest(kind=kind):
                direct = chunk(1, kind=kind, direct=True)
                result = self.retrieve([direct], [chunk(0), chunk(1, kind=kind), chunk(2)])
                self.assertEqual([doc.page_content for doc in result], [direct.page_content])
                self.assertEqual(result[0].metadata["retrieval_origin"], {"direct_rank": 1, "neighbor_of": [], "section_coverage": False})
                self.assertIs(result[0].metadata["retrieval_neighbor"], False)


if __name__ == "__main__":
    unittest.main()
