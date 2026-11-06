"""Experimental ordering guards; synthetic fixtures, no provider or paper corpus."""
import copy
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from evaluation.runtime import effective_overrides, parse_overrides
from refmind.citations import build_citation_label
from refmind.config import settings
from refmind.config.settings import Settings
from refmind.rag import graph
from refmind.rag.evidence_bundle import pack_evidence
from refmind.rag.evidence_priority import evidence_kind, prioritize_evidence


def document(text, chunk, **extra):
    return Document(page_content=text, metadata={'doc_id': 1, 'group_id': 7, 'page': 1,
        'paper_title': 'Synthetic paper', 'chunk_id': chunk, **extra})


class EvidencePriorityTests(unittest.TestCase):
    def pair(self):
        return [document('Reference entry ' * 10, 'reference', section_path='References'),
                document('Measured result: 13 A at 77 K.', 'body', section_path='Results')]

    def test_configuration_is_disabled_by_default_and_override_is_consumed(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('CONTEXT_BODY_PRIORITY', None)
            self.assertFalse(Settings().context_body_priority_enabled)
            os.environ['CONTEXT_BODY_PRIORITY'] = 'true'
            selected = Settings()
            self.assertTrue(selected.context_body_priority_enabled)
            values = parse_overrides(['CONTEXT_BODY_PRIORITY=true'])
            self.assertTrue(effective_overrides(selected, values)['CONTEXT_BODY_PRIORITY']['effective'])
            selected.context_body_priority_enabled = False
            with self.assertRaisesRegex(ValueError, 'not consumed'):
                effective_overrides(selected, values)

    def test_explicit_switch_moves_whole_body_into_fixed_budget_and_can_be_reverted(self):
        docs = self.pair()
        original = copy.deepcopy([(d.page_content, d.metadata) for d in docs])
        options = dict(evidence_tokens=245, counter=len)
        before = pack_evidence('What was measured?', docs, **options)
        after = pack_evidence('What was measured?', docs, prioritize_body=True, **options)
        reverted = pack_evidence('What was measured?', docs, prioritize_body=False, **options)
        self.assertEqual([d.metadata['chunk_id'] for d in before.documents], ['reference'])
        self.assertEqual([d.metadata['chunk_id'] for d in after.documents], ['body'])
        self.assertEqual([d.page_content for d in before.documents], [d.page_content for d in reverted.documents])
        self.assertEqual(original, [(d.page_content, d.metadata) for d in docs])
        self.assertTrue(after.trace['evidence_priority']['enabled'])
        self.assertEqual(len(after.trace['evidence_priority']['moves']), 2)

    def test_bibliographic_query_keeps_reference_priority(self):
        docs = self.pair()
        for question in ('Which references are cited?', '这篇论文引用了哪些参考文献？', 'Locate the DOI.'):
            with self.subTest(question=question):
                result = prioritize_evidence(question, docs, mode='references', strategy='B_safe')
                self.assertEqual([id(d) for d in result], [id(d) for d in docs])

    def test_numbered_scientific_steps_and_mixed_body_are_not_references(self):
        candidates = [
            '[1] In 2020 we use an IEEE instrument at 77 K.\n[2] In 2021 we evaluate the result using doi:10.1234/example.',
            'We measured the current at 77 K.\nREFERENCES\n[1] A. Example, IEEE Trans., vol. 1, 2020.\n[2] B. Example, IEEE Trans., vol. 2, 2021.',
            'conditions determine the measured current.\nREFERENCES\n[1] A. Example, IEEE Trans., vol. 1, 2020.\n[2] B. Example, IEEE Trans., vol. 2, 2021.',
        ]
        for text in candidates:
            with self.subTest(text=text[:40]):
                self.assertNotEqual(evidence_kind(document(text, 'mixed'), strategy='B_safe')['kind'], 'reference')

    def test_tables_formulas_and_short_captions_remain_intact(self):
        docs = [document('| Year | Current |\n|---|---|\n| 2020 | 13 A |', 'table', content_type='table', is_atomic=True),
                document('I = V/R; 77 K.', 'formula', content_type='equation', is_atomic=True),
                document('Fig. 2: I(B), 77 K [4].', 'caption', content_type='figure_caption')]
        snapshot = copy.deepcopy([(d.page_content, d.metadata) for d in docs])
        result = prioritize_evidence('What was measured?', docs, mode='references', strategy='B_safe')
        self.assertEqual([id(d) for d in result], [id(d) for d in docs])
        self.assertEqual(snapshot, [(d.page_content, d.metadata) for d in docs])

    def test_image_pointer_is_never_read_mutated_or_demoted_by_this_policy(self):
        image = document('Figure', 'image', content_type='figure', image_path='fixture://not-opened/original.png')
        docs = [image, *self.pair()]
        result = prioritize_evidence('Explain the method.', docs, mode='references', strategy='B_safe')
        self.assertIs(result[0], image)
        self.assertEqual(image.metadata['image_path'], 'fixture://not-opened/original.png')
        self.assertCountEqual([id(d) for d in result], [id(d) for d in docs])

    def test_stable_order_never_merges_equal_text_from_different_sources(self):
        first = document('Same scientific text.', 'a')
        second = document('Same scientific text.', 'b')
        second.metadata['doc_id'] = 2
        result = prioritize_evidence('Explain.', [first, second], strategy='B_safe')
        self.assertEqual([id(d) for d in result], [id(first), id(second)])

    def test_comparison_and_summary_keep_whole_body_from_each_source(self):
        first = document('A' * 40, 'body-a', section_path='Results')
        second = document('B' * 40, 'body-b', section_path='Results')
        second.metadata['doc_id'] = 2
        references = document('Reference entry ' * 10, 'references', section_path='References')
        for kind in ('comparison', 'summary'):
            with self.subTest(kind=kind):
                result = pack_evidence('Compare or summarize the measurements.', [references, first, second],
                    task_type=kind, prioritize_body=True, evidence_tokens=205, counter=len,
                    required_source_ids=['library:7:1', 'library:7:2'])
                self.assertEqual({d.metadata['doc_id'] for d in result.documents}, {1, 2})
                expected = ['B' * 40, 'A' * 40] if kind == 'comparison' else ['A' * 40, 'B' * 40]
                self.assertEqual([d.page_content for d in result.documents], expected)
                self.assertEqual(result.trace['missing_required_source_ids'], [])

    def test_research_graph_consumes_setting_and_preserves_trace(self):
        docs = self.pair()
        for enabled in (False, True):
            with (self.subTest(enabled=enabled),
                patch.object(settings, 'context_body_priority_enabled', enabled),
                patch.object(settings, 'context_max_tokens', 4000),
                patch.object(graph, 'get_llm', return_value=Mock()),
                patch.object(graph.EvidenceReviewAgent, 'review', side_effect=lambda q, ds: (ds, None)),
                patch.object(graph, '_generate_answer', side_effect=lambda q, ds, *a, **k: 'Supported '+build_citation_label(ds[0], 1)),
                patch.object(graph, '_run_text_hook', side_effect=lambda h, a, **k: a),
                patch.object(graph.AnswerReviewAgent, 'review', side_effect=lambda q, ds, a: (a, None)),
            ):
                result = graph.answer_from_evidence('What was measured?', docs)
            self.assertEqual(result['trace']['packing']['evidence_priority']['enabled'], enabled)
            order = [d.metadata['chunk_id'] for d in result['trace']['stages']['packed']['documents']]
            self.assertEqual(order, ['body', 'reference'] if enabled else ['reference', 'body'])


if __name__ == '__main__':
    unittest.main()
