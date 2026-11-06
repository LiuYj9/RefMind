"""Offline scope, structural-evidence, coordinate and final-source boundaries."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import fitz
import numpy as np
from langchain_core.documents import Document

from refmind import citations
from refmind.parsing.coordinates import pdf_bbox
from refmind.parsing import image_store
from refmind.rag import compression, graph, retrieval
from refmind.config import settings


def doc(doc_id, text='same evidence', **metadata):
    return Document(page_content=text, metadata={'group_id': 7, 'doc_id': doc_id,
                    'filename': f'paper-{doc_id}.pdf', 'page': 1, **metadata})


class CoordinatesV3Tests(unittest.TestCase):
    def test_normalized_boxes_round_trip_unrotated_annotations_and_rotated_crops(self):
        for rotation in (0, 90, 180, 270):
            with self.subTest(rotation=rotation), fitz.open() as pdf:
                page = pdf.new_page(width=300, height=500)
                expected = fitz.Rect(30, 50, 90, 100)
                page.draw_rect(expected, color=(1, 0, 0), fill=(1, 0, 0))
                page.set_rotation(rotation)
                displayed = expected * page.rotation_matrix
                normalized = [displayed.x0 / page.rect.width * 1000,
                              displayed.y0 / page.rect.height * 1000,
                              displayed.x1 / page.rect.width * 1000,
                              displayed.y1 / page.rect.height * 1000]
                actual = pdf_bbox(page, normalized, 'normalized_1000')
                for left, right in zip(actual, expected):
                    self.assertAlmostEqual(left, right, places=4)
                image = image_store._render_bbox(page, normalized, 'normalized_1000')
                pixmap = fitz.Pixmap(image)
                center = pixmap.pixel(pixmap.width // 2, pixmap.height // 2)
                self.assertEqual(center[:3], (255, 0, 0))

    def test_unknown_units_and_nonfinite_boxes_are_not_guessed(self):
        with fitz.open() as pdf:
            page = pdf.new_page()
            self.assertIsNone(pdf_bbox(page, [0, 0, float('nan'), 10]))
            self.assertIsNone(pdf_bbox(page, [0, 0, 100, 100], 'unknown'))

    def test_failed_figure_crop_never_binds_first_page_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'paper.pdf'
            with fitz.open() as pdf:
                pdf.new_page()
                pdf.save(path)
            with patch.object(image_store, '_render_bbox', return_value=None), patch.object(image_store, '_first_embedded_image', return_value=b'wrong-image') as fallback:
                records = image_store.extract_pdf_figures(path, docstore_dir=Path(tmp)/'store', doc_id=1,
                         figure_blocks=[{'page': 1, 'bbox': [0, 0, 10, 10], 'block_id': 'figure2'}])
            self.assertEqual(records, [])
            fallback.assert_not_called()


class EvidenceBudgetV3Tests(unittest.TestCase):
    def test_atomic_blocks_are_admitted_whole_or_skipped_without_hiding_later_evidence(self):
        small = doc(1, '<table>123</table>', is_atomic=True)
        large = doc(1, 'X' * 200, is_atomic=True)
        later = doc(2, 'useful')
        with patch.object(settings, 'context_max_chars', 40):
            self.assertEqual(compression._apply_char_budget([large, small, later]), [small, later])
            model = Mock()
            self.assertIs(compression._compress_one(small, np.array([1., 0.]), model), small)
            model.embed_documents.assert_not_called()

    def test_identical_text_from_distinct_papers_survives_dedup(self):
        first, second = doc(1), doc(2)
        docs, _ = compression._dedup([first, second, first], [np.array([1., 0.])] * 3)
        self.assertEqual([d.metadata['doc_id'] for d in docs], [1, 2])


class RetrievalScopeV3Tests(unittest.TestCase):
    def setUp(self):
        retrieval.reset_retrievers()
        self.rows = [
            SimpleNamespace(id=doc_id, status=status, index_version=None, file_sha256=None)
            for doc_id, status in ((1, 'ready'), (2, 'indexing'), (3, 'ready'))
        ]
        patcher = patch.object(retrieval.storage, 'list_documents', side_effect=lambda _gid: self.rows)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_empty_selection_cannot_expand_to_whole_library(self):
        with patch.object(retrieval, 'get_vectorstore') as vector:
            self.assertIsNone(retrieval.build_retriever(7, selected_doc_ids=()))
        vector.assert_not_called()
        self.assertEqual(retrieval.allowed_ready_doc_ids(7, [1, 2, 99]), (1,))

    def test_both_recall_branches_pre_filter_and_final_guard_excludes_nonready(self):
        source = SimpleNamespace(invoke=lambda query: [doc(3), doc(2), doc(1)])
        vector = Mock()
        vector.as_retriever.return_value = source
        with patch.object(retrieval, 'get_vectorstore', return_value=vector):
            retriever = retrieval.build_retriever(7, documents=[doc(1), doc(2), doc(3)], selected_doc_ids=(1, 2))
            results = retriever.invoke('evidence')
        self.assertEqual([item.metadata['doc_id'] for item in results], [1])
        self.assertEqual(vector.as_retriever.call_args.kwargs['search_kwargs']['filter'], {'doc_id': {'$in': [1]}})
        self.rows[0].status = 'indexing'
        self.assertEqual(retriever.invoke('evidence'), [])

    def test_final_guard_reads_current_committed_identity_and_rejects_incomplete_pair(self):
        row = self.rows[0]
        first = doc(1)
        first.metadata.update(version='v1', file_sha256='a' * 64)
        row.index_version = 'v1'
        self.assertEqual(retrieval.filter_ready_documents(7, [first]), [])
        row.file_sha256 = 'a' * 64
        self.assertEqual(retrieval.filter_ready_documents(7, [first]), [first])

        second = doc(1)
        second.metadata.update(version='v2', file_sha256='b' * 64)
        row.index_version, row.file_sha256 = 'v2', 'b' * 64
        self.assertEqual(retrieval.filter_ready_documents(7, [first, second]), [second])

    def test_fusion_preserves_distinct_papers_even_with_identical_content(self):
        source = SimpleNamespace(invoke=lambda query: [doc(3), doc(1)])
        vector = Mock()
        vector.as_retriever.return_value = source
        with patch.object(retrieval, 'get_vectorstore', return_value=vector):
            retriever = retrieval.build_retriever(7, documents=[doc(1), doc(3)])
            results = retriever.invoke('evidence')
        self.assertEqual({d.metadata['doc_id'] for d in results}, {1, 3})


class FinalGroundingV3Tests(unittest.TestCase):
    def test_refusal_never_gets_bibliography(self):
        rendered = citations.finalize_answer_citations(graph.NO_CONTEXT_REPLY, [doc(1)])
        self.assertEqual(rendered, graph.NO_CONTEXT_REPLY)
        answer, documents, warnings = graph._finalize_grounded_answer(rendered, [doc(1)])
        self.assertEqual(documents, [])
        self.assertNotIn('参考来源', answer)

    def test_unknown_source_label_is_rejected_after_plugin(self):
        answer, documents, warnings = graph._finalize_grounded_answer('Unsupported [文献99《Other》，第8页]', [doc(1)])
        self.assertEqual(answer, graph.NO_CONTEXT_REPLY)
        self.assertEqual(documents, [])
        self.assertTrue(warnings)

    def test_unverified_session_fulltext_never_reaches_model(self):
        evidence = doc(1, evidence_origin='session_full_text', evidence_level='full_text', resource_id='asset-1', status='ready', identity_verified=False)
        with patch.object(graph, 'get_llm') as model:
            result = graph.answer_from_evidence('Question', [evidence])
        model.assert_not_called()
        self.assertEqual(result['documents'], [])

    def test_abstract_entrypoint_uses_reviews_and_posthook_validation(self):
        evidence = doc(1, evidence_origin='academic_search', evidence_level='abstract', paper_title='Paper', external_url='https://example.org/paper')
        with patch.object(settings, 'evidence_review_enabled', True), patch.object(settings, 'answer_review_enabled', True), patch.object(graph.EvidenceReviewAgent, 'review', return_value=([evidence], None)) as before, patch.object(graph.AnswerReviewAgent, 'review', side_effect=lambda question, docs, answer: (answer, None)) as after, patch.object(graph, '_generate_answer', return_value='Draft'), patch.object(graph, '_run_text_hook', return_value='Injected [文献999《Other》，第1页]'), patch.object(graph, 'get_llm', return_value=Mock()):
            result = graph.answer_from_evidence('Question', [evidence])
        before.assert_called_once()
        after.assert_called_once()
        self.assertEqual(result['answer'], graph.NO_CONTEXT_REPLY)
        self.assertEqual(result['documents'], [])

    def test_stage_trace_contains_actual_snapshots(self):
        trace = {}
        raw = [doc(1), doc(2)]
        with patch.object(graph, 'rerank', return_value=[raw[1]]), patch.object(graph, 'compress_context', return_value=[raw[1]]):
            graph._postprocess_documents('Question', raw, trace=trace)
        self.assertEqual(trace['stages']['raw_candidates']['documents'], raw)
        self.assertEqual(trace['stages']['reranked']['documents'], [raw[1]])
        self.assertEqual(trace['stages']['packed']['documents'], [raw[1]])


if __name__ == '__main__':
    unittest.main()
