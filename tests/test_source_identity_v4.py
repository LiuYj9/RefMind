"""Stable selection identity, final source coverage, and answer-stage diagnostics."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from refmind import storage
from refmind.citations import build_citation_label
from refmind.config import settings
from refmind.rag import graph, research_pipeline
from refmind.rag.evidence_bundle import describe_task, source_identity, source_key
from refmind.services.paper_discovery import PaperCandidate
from refmind.web import store
from tests import test_web_jobs as web_fixtures


def doc(number, **metadata):
    return Document(page_content='Measured evidence for the same title.',
                    metadata={'group_id': 7, 'doc_id': number, 'paper_title': 'Same title',
                              'page': 1, 'paragraph_index': number, **metadata})


class SourceIdentityTests(unittest.TestCase):
    def test_two_same_title_papers_are_not_a_single_selected_paper(self):
        task = describe_task('这篇论文的方法是什么？', source_titles=['Same', 'Same'],
                             source_ids=['library:7:1', 'library:7:2'])
        self.assertEqual(task.source_titles, ['Same'])
        self.assertEqual(task.source_ids, ['library:7:1', 'library:7:2'])
        self.assertTrue(task.needs_clarification)
        self.assertNotEqual(task.resolution, 'selected_single_paper')

    def test_duplicate_selection_identity_remains_one_paper(self):
        task = describe_task('这篇论文的方法是什么？', source_titles=['Same', 'Alias'],
                             source_ids=['library:7:1', 'library:7:1'])
        self.assertFalse(task.needs_clarification)
        self.assertEqual(task.resolution, 'selected_single_paper')
        self.assertEqual(task.source_ids, ['library:7:1'])

    def test_legacy_three_argument_call_stays_compatible(self):
        task = describe_task('它的方法？', [], ['Known Paper'])
        self.assertEqual(task.resolution, 'selected_single_paper')
        self.assertIn('Known Paper', task.resolved_query)

    def test_library_identity_has_group_doc_but_not_title_or_version(self):
        self.assertEqual(source_identity(doc(1, version='a')), 'library:7:1')
        self.assertEqual(source_identity(doc(1, version='b', paper_title='Renamed')), 'library:7:1')
        self.assertNotEqual(source_identity(doc(1)), source_identity(doc(1, group_id=8)))
        self.assertNotEqual(source_key(doc(1, version='a')), source_key(doc(1, version='b')))

    def test_asset_and_abstract_identity_precede_local_doc_fields(self):
        self.assertEqual(source_identity(doc(1, resource_id='asset-a')), 'asset:asset-a')
        self.assertEqual(source_identity(doc(1, evidence_origin='academic_search',
                         evidence_id='paper-a')), 'abstract:paper-a')
        self.assertNotEqual(source_identity(doc(1, resource_id='asset-a')),
                            source_identity(doc(1, resource_id='asset-b')))

    def test_titles_never_determine_anonymous_identity(self):
        first = Document(page_content='A', metadata={'paper_title': 'Same'})
        second = Document(page_content='B', metadata={'paper_title': 'Same'})
        renamed = Document(page_content='A', metadata={'paper_title': 'Other'})
        self.assertNotEqual(source_identity(first), source_identity(second))
        self.assertEqual(source_identity(first), source_identity(renamed))

    def test_research_pipeline_passes_ids_and_clarifies_before_retrieval(self):
        rows = {number: SimpleNamespace(group_id=7, status='ready', paper_title='Same', filename='same.pdf')
                for number in (1, 2)}
        with (patch.object(research_pipeline.storage, 'get_document', side_effect=rows.get),
              patch.object(research_pipeline, 'retrieve_for_task') as retrieve):
            result = research_pipeline.retrieve_research_evidence('它的方法？', 7, selected_doc_ids=[1, 2])
        self.assertTrue(result['task_description']['needs_clarification'])
        self.assertEqual(result['task_description']['source_ids'], ['library:7:1', 'library:7:2'])
        retrieve.assert_not_called()


class FinalSourceCoverageTests(unittest.TestCase):
    def test_reviewer_cannot_hide_a_missing_same_title_paper_from_coverage(self):
        first, second = doc(1), doc(2)
        descriptor = describe_task('比较两篇论文的方法', source_titles=['Same title', 'Same title'],
                                   source_ids=['library:7:1', 'library:7:2']).to_dict()
        with (patch.object(graph, 'get_llm', return_value=Mock()),
              patch.object(graph.EvidenceReviewAgent, 'review', return_value=([first], None)),
              patch.object(graph, '_generate_answer', return_value='Supported ' + build_citation_label(first, 1)),
              patch.object(graph, '_run_text_hook', side_effect=lambda _hook, answer, **_: answer),
              patch.object(graph.AnswerReviewAgent, 'review', side_effect=lambda q, ds, answer: (answer, None))):
            result = graph.answer_from_evidence('比较两篇论文的方法', [first, second], task_description=descriptor)
        coverage = result['trace']['source_coverage']
        self.assertEqual(coverage['generation_source_ids'], ['library:7:1'])
        self.assertEqual(coverage['missing_source_ids'], ['library:7:2'])
        self.assertEqual(result['status'], 'scoped')
        self.assertTrue(any('缺少 1 个独立来源' in warning for warning in result['warnings']))

    def test_coverage_runs_after_final_budget_removes_whole_source(self):
        first, second = doc(1), doc(2)
        first.page_content, second.page_content = 'A' * 600, 'B' * 600
        scope = 'bound ' * 200

        class Reviewer:
            def __init__(self, *_):
                self.last_review = {'coverage': 'scoped', 'scope_limit': scope}
            def review(self, _question, documents):
                return documents, None

        descriptor = describe_task('比较两篇论文', source_titles=['Same title'],
                                   source_ids=['library:7:1', 'library:7:2']).to_dict()
        with (patch.object(graph, 'get_llm', return_value=Mock()),
              patch.object(graph, 'EvidenceReviewAgent', Reviewer),
              patch.object(graph, '_generate_answer', return_value='Supported ' + build_citation_label(first, 1)),
              patch.object(graph, '_run_text_hook', side_effect=lambda _hook, answer, **_: answer),
              patch.object(graph.AnswerReviewAgent, 'review', side_effect=lambda q, ds, answer: (answer, None)),
              patch.object(graph, 'token_counter', return_value=(len, 'test_characters')),
              patch.object(settings, 'context_max_tokens', 1900)):
            result = graph.answer_from_evidence('比较两篇论文', [first, second], task_description=descriptor)
        self.assertEqual(len(result['trace']['stages']['packed']['documents']), 2)
        self.assertEqual(len(result['trace']['stages']['generation_context']['documents']), 1)
        self.assertEqual(result['trace']['source_coverage']['missing_source_ids'], ['library:7:2'])

    def test_generation_hook_and_review_text_remain_distinguishable_after_final_refusal(self):
        original = doc(1)
        with (patch.object(graph, 'get_llm', return_value=Mock()),
              patch.object(graph.EvidenceReviewAgent, 'review', side_effect=lambda q, ds: (ds, None)),
              patch.object(graph, '_generate_answer', return_value='Original draft'),
              patch.object(graph, '_run_text_hook', return_value='Plugin draft'),
              patch.object(graph.AnswerReviewAgent, 'review', return_value=('Review [文献999《Unknown》，第8页]', None))):
            result = graph.answer_from_evidence('Question', [original])
        self.assertEqual(result['answer'], graph.NO_CONTEXT_REPLY)
        self.assertEqual(result['rejection_stage'], 'citation_validation')
        self.assertEqual(result['trace']['answer_stages'], {
            'generated_draft': 'Original draft', 'after_generate_hook': 'Plugin draft',
            'after_answer_review': 'Review [文献999《Unknown》，第8页]',
            'after_citation_validation': graph.NO_CONTEXT_REPLY})
        self.assertTrue(all(isinstance(value, str) for value in result['trace']['answer_stages'].values()))


class WebSourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = web_fixtures.WebJobTests(methodName='runTest')
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_same_title_library_docs_do_not_skip_clarification(self):
        fixture = self.fixture
        ids = []
        for _ in range(2):
            task = fixture.upload_task()
            fixture.runner._run(task['id'])
            ids.append(store.task(task['id'])['result']['document_id'])
        self.assertEqual(len(set(ids)), 2)
        for doc_id in ids:
            storage.update_document(doc_id, paper_title='Same title')
        task = store.create_task(fixture.uid, 'answer', {'session_id': fixture.session['id'],
            'question': '这篇论文的方法是什么？', 'mode': 'library', 'group_id': fixture.library['id'],
            'selected_doc_ids': ids})
        with (patch('refmind.rag.graph.answer_from_evidence') as answer,
              patch('refmind.rag.scientific_retrieval.retrieve_for_task') as retrieve):
            fixture.runner._run(task['id'])
        result = store.task(task['id'])['result']
        self.assertEqual(result['status'], 'clarification')
        self.assertEqual(len(result['task_description']['source_ids']), 2)
        answer.assert_not_called()
        retrieve.assert_not_called()

    def test_same_title_external_paper_ids_do_not_merge(self):
        fixture = self.fixture
        search = store.create_task(fixture.uid, 'search', {})
        store.update_task(search['id'], status='completed', result={'papers': [
            PaperCandidate('paper-a', 'Same title').to_dict(), PaperCandidate('paper-b', 'Same title').to_dict()]})
        task = store.create_task(fixture.uid, 'answer', {'session_id': fixture.session['id'],
            'question': '这篇论文的方法是什么？', 'mode': 'external', 'abstracts': [
                {'search_id': search['id'], 'paper_id': value} for value in ['paper-a', 'paper-b']]})
        with patch('refmind.rag.graph.answer_from_evidence') as answer:
            fixture.runner._run(task['id'])
        result = store.task(task['id'])['result']
        self.assertEqual(result['status'], 'clarification')
        self.assertEqual(result['task_description']['source_ids'], ['abstract:paper-a', 'abstract:paper-b'])
        answer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
