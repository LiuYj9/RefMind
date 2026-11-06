"""Synthetic offline contract: allowed retrieval scope versus required sources.

No dataset, source PDF, provider, index or model is used. Run against the staging
business-module overlay, never by accidentally importing the previous checkout.
"""
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from refmind.citations import build_citation_label
from refmind.config import settings
from refmind.rag import graph, research_pipeline
from refmind.rag.evidence_bundle import EvidenceBundle, describe_task, pack_evidence
from refmind.web import jobs


def sid(number, group=7):
    return f'library:{group}:{number}'


def doc(number, text='Measured result under its stated conditions.', title=None):
    return Document(page_content=text, metadata={'group_id': 7, 'doc_id': number,
        'filename': f'paper-{number}.pdf', 'paper_title': title or f'Paper {number}',
        'page': 1, 'paragraph_index': number, 'chunk_id': f'chunk-{number}',
        'evidence_level': 'full_text'})


def descriptor(question='Explain the method.', *, allowed=(1, 2, 3), required=(1, 2), task_type=None):
    task = describe_task(question, source_titles=[f'Paper {value}' for value in allowed],
        source_ids=[sid(value) for value in allowed],
        required_source_ids=[sid(value) for value in required])
    value = task.to_dict()
    if task_type is not None:
        value['task_type'] = task_type
    return value


@contextmanager
def mocked_answer_path():
    """Exercise business gates with fully observable, non-network model substitutes."""
    with ExitStack() as stack:
        model = stack.enter_context(patch.object(graph, 'get_llm', return_value=Mock(name='offline_model')))
        evidence_review = stack.enter_context(patch.object(graph.EvidenceReviewAgent, 'review',
            side_effect=lambda question, documents: (documents, None)))
        answer_review = stack.enter_context(patch.object(graph.AnswerReviewAgent, 'review',
            side_effect=lambda question, documents, answer: (answer, None)))
        generate = stack.enter_context(patch.object(graph, '_generate_answer',
            side_effect=lambda question, documents, *args, **kwargs:
                'Supported. ' + build_citation_label(documents[0], 1) if documents else graph.NO_CONTEXT_REPLY))
        stack.enter_context(patch.object(graph, '_run_text_hook', side_effect=lambda hook, text, **kwargs: text))
        for name, value in (('context_max_tokens', 8000), ('model_context_tokens', 32768),
                            ('answer_token_reserve', 512)):
            stack.enter_context(patch.object(settings, name, value))
        yield SimpleNamespace(model=model, evidence_review=evidence_review,
                              answer_review=answer_review, generate=generate)


class ScopeTargetsContractTests(unittest.TestCase):
    def assert_target_block(self, result):
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result['rejection_stage'], 'target_coverage')
        self.assertEqual(result['documents'], [])
        self.assertEqual(result['review_status'], 'unreviewed')
        self.assertEqual(result['trace']['packing']['missing_required_source_ids'],
                         result['trace']['source_coverage']['missing_source_ids'])
        context = result.get('trace', {}).get('stages', {}).get('generation_context', {})
        self.assertEqual(context.get('documents', []), [])
        self.assertFalse(context.get('request_sent', False))

    def test_legacy_selected_sources_and_explicit_empty_targets_have_distinct_meanings(self):
        allowed = [sid(1), sid(2)]
        legacy = describe_task('Explain the method.', source_titles=['A', 'B'], source_ids=allowed)
        explicit = describe_task('Explain the method.', source_titles=['A', 'B'], source_ids=allowed,
                                 required_source_ids=[])
        self.assertEqual(legacy.to_dict()['required_source_ids'], allowed)
        self.assertEqual(explicit.to_dict()['required_source_ids'], [])
        self.assertEqual(explicit.to_dict()['source_ids'], allowed)

    def test_duplicate_identity_and_same_title_do_not_change_single_paper_enumeration(self):
        question = '这篇论文的三个处理步骤是什么？'
        one = describe_task(question, source_titles=['Same title', 'Same title'],
            source_ids=[sid(1), sid(1)], required_source_ids=[sid(1), sid(1)])
        self.assertEqual(one.task_type, 'method')
        self.assertFalse(one.needs_clarification)
        self.assertEqual(one.resolution, 'selected_single_paper')
        self.assertEqual(one.to_dict()['required_source_ids'], [sid(1)])
        two = describe_task(question, source_titles=['Same title', 'Same title'],
            source_ids=[sid(1), sid(2)], required_source_ids=[sid(1), sid(2)])
        self.assertTrue(two.needs_clarification)
        self.assertEqual(two.task_type, 'method')

    def test_unauthorized_required_source_raises_before_any_get_llm(self):
        with self.assertRaises(ValueError):
            describe_task('Explain.', source_ids=[sid(1)], required_source_ids=[sid(2)])
        invalid = {'original_question': 'Explain.', 'resolved_query': 'Explain.', 'task_type': 'method',
                   'source_ids': [sid(1)], 'required_source_ids': [sid(2)]}
        with mocked_answer_path() as calls:
            with self.assertRaises(ValueError):
                graph.answer_from_evidence('Explain.', [doc(1)], task_description=invalid)
        calls.model.assert_not_called()
        calls.evidence_review.assert_not_called()
        calls.generate.assert_not_called()

    def test_large_allowed_scope_with_two_present_targets_has_no_missing_target_warning(self):
        task = descriptor('Compare the methods.', allowed=tuple(range(1, 101)), required=(1, 2), task_type='comparison')
        with mocked_answer_path() as calls:
            result = graph.answer_from_evidence('Compare the methods.', [doc(1), doc(2)], task_description=task)
        self.assertNotEqual(result['status'], 'insufficient')
        self.assertEqual(result['trace']['source_coverage']['missing_source_ids'], [])
        self.assertFalse(result['warnings'])
        calls.generate.assert_called_once()

    def test_expanding_allowed_scope_does_not_change_real_partial_missing_targets(self):
        results = []
        for allowed in ((1, 2), tuple(range(1, 51))):
            task = descriptor('Compare the methods.', allowed=allowed, required=(1, 2), task_type='comparison')
            with mocked_answer_path():
                results.append(graph.answer_from_evidence('Compare the methods.', [doc(1)], task_description=task))
        for result in results:
            self.assertEqual(result['trace']['source_coverage']['missing_source_ids'], [sid(2)])
            self.assertEqual(result['status'], 'scoped')
            self.assertTrue(result['warnings'])
        self.assertEqual(results[0]['warnings'], results[1]['warnings'])

    def test_pipeline_allowed_only_keeps_targets_unresolved_and_searches_that_allowlist(self):
        rows = {number: SimpleNamespace(id=number, group_id=7, status='ready', paper_title=f'Paper {number}',
                                       filename=f'paper-{number}.pdf') for number in (1, 2, 3)}
        with (patch.object(research_pipeline.storage, 'get_document', side_effect=rows.get),
              patch.object(research_pipeline.storage, 'list_documents', return_value=list(rows.values())),
              patch.object(research_pipeline, 'retrieve_for_task', return_value=[doc(2)]) as retrieve,
              patch.object(research_pipeline, 'rerank', side_effect=lambda question, docs, **kwargs: docs)):
            result = research_pipeline.retrieve_research_evidence('Explain the method.', 7, allowed_doc_ids=[1, 2, 3])
        self.assertEqual(retrieve.call_args.kwargs['selected_doc_ids'], (1, 2, 3))
        self.assertEqual(result['task_description']['required_source_ids'], [])
        self.assertEqual(result['task_description']['target_resolution'], 'unresolved_allowed_scope')

    def test_pipeline_selected_targets_remain_the_only_search_scope_with_larger_allowed(self):
        rows = {number: SimpleNamespace(id=number, group_id=7, status='ready', paper_title=f'Paper {number}',
                                       filename=f'paper-{number}.pdf') for number in (1, 2, 3)}
        with (patch.object(research_pipeline.storage, 'get_document', side_effect=rows.get),
              patch.object(research_pipeline.storage, 'list_documents', return_value=list(rows.values())),
              patch.object(research_pipeline, 'retrieve_for_task', return_value=[doc(2)]) as retrieve,
              patch.object(research_pipeline, 'rerank', side_effect=lambda question, docs, **kwargs: docs)):
            result = research_pipeline.retrieve_research_evidence('它的方法是什么？', 7,
                selected_doc_ids=[2], allowed_doc_ids=[1, 2, 3])
        self.assertEqual(retrieve.call_args.kwargs['selected_doc_ids'], (2,))
        self.assertEqual(result['task_description']['required_source_ids'], [sid(2)])
        self.assertIn('Paper 2', result['task_description']['resolved_query'])
        self.assertNotIn('Paper 1', result['task_description']['resolved_query'])

    def test_pipeline_rejects_outside_nonready_and_wrong_group_before_retrieval_or_model(self):
        for selected, allowed, status, group in (([2], [1], 'ready', 7),
                                               ([1], [1], 'indexing', 7),
                                               ([1], [1], 'ready', 8)):
            with self.subTest(selected=selected, allowed=allowed, status=status, group=group):
                rows = {number: SimpleNamespace(id=number, group_id=group, status=status,
                    paper_title=f'Paper {number}', filename=f'{number}.pdf') for number in (1, 2)}
                with (patch.object(research_pipeline.storage, 'get_document', side_effect=rows.get),
                      patch.object(research_pipeline.storage, 'list_documents', return_value=list(rows.values())),
                      patch.object(research_pipeline, 'retrieve_for_task') as retrieve,
                      patch.object(graph, 'get_llm') as model):
                    with self.assertRaises((ValueError, PermissionError)):
                        research_pipeline.answer_research_question('Explain.', 7,
                            selected_doc_ids=selected, allowed_doc_ids=allowed)
                retrieve.assert_not_called()
                model.assert_not_called()

    def test_all_targets_missing_in_initial_input_blocks_before_get_llm(self):
        task = descriptor(allowed=(1, 2, 3), required=(1, 2))
        with mocked_answer_path() as calls:
            result = graph.answer_from_evidence('Explain.', [doc(3)], task_description=task)
        self.assert_target_block(result)
        calls.model.assert_not_called()
        calls.evidence_review.assert_not_called()
        calls.generate.assert_not_called()

    def test_first_pack_losing_all_targets_blocks_before_get_llm(self):
        task = descriptor(allowed=(1, 2, 3), required=(1, 2))
        packed = EvidenceBundle([doc(3)], [], {}, [])
        with mocked_answer_path() as calls, patch.object(graph, 'pack_evidence', return_value=packed):
            result = graph.answer_from_evidence('Explain.', [doc(1), doc(3)], task_description=task)
        self.assert_target_block(result)
        calls.model.assert_not_called()
        calls.evidence_review.assert_not_called()
        calls.generate.assert_not_called()

    def test_review_losing_all_targets_blocks_generation_even_if_background_remains(self):
        task = descriptor(allowed=(1, 2), required=(1,))
        with mocked_answer_path() as calls:
            calls.evidence_review.side_effect = lambda question, documents: ([doc(2)], None)
            result = graph.answer_from_evidence('Explain.', [doc(1), doc(2)], task_description=task)
        self.assert_target_block(result)
        calls.generate.assert_not_called()
        calls.answer_review.assert_not_called()

    def test_guard_losing_all_targets_blocks_generation_even_if_background_remains(self):
        task = descriptor(allowed=(1, 2), required=(1,))
        with mocked_answer_path() as calls:
            result = graph.answer_from_evidence('Explain.', [doc(1), doc(2)], task_description=task,
                evidence_guard=lambda documents: [document for document in documents if document.metadata['doc_id'] == 2])
        self.assert_target_block(result)
        calls.generate.assert_not_called()
        calls.answer_review.assert_not_called()

    def test_final_budget_losing_all_targets_blocks_generation_even_if_background_remains(self):
        task = descriptor(allowed=(1, 2), required=(1,))
        with (mocked_answer_path() as calls,
              patch.object(graph, 'enforce_generation_budget', return_value=EvidenceBundle([doc(2)], [], {}, []))):
            result = graph.answer_from_evidence('Explain.', [doc(1), doc(2)], task_description=task)
        self.assert_target_block(result)
        calls.generate.assert_not_called()
        calls.answer_review.assert_not_called()

    def test_unresolved_allowed_comparison_summary_are_scoped_but_method_is_not(self):
        for task_type in ('comparison', 'summary', 'method'):
            with self.subTest(task_type=task_type):
                task = descriptor(allowed=(1, 2, 3), required=(), task_type=task_type)
                task['target_resolution'] = 'unresolved_allowed_scope'
                with mocked_answer_path():
                    result = graph.answer_from_evidence('Explain the methods.', [doc(1)], task_description=task)
                self.assertEqual(result['trace']['source_coverage']['missing_source_ids'], [])
                if task_type in {'comparison', 'summary'}:
                    self.assertEqual(result['status'], 'scoped')
                    self.assertTrue(result['warnings'])
                else:
                    self.assertEqual(result['status'], 'answered')
                    self.assertFalse(result['warnings'])

    def test_partial_selected_scope_does_not_scope_ordinary_method_numeric_or_table_questions(self):
        for task_type in ('method', 'numeric', 'table'):
            with self.subTest(task_type=task_type), mocked_answer_path():
                task = descriptor(allowed=(1, 2), required=(1, 2), task_type=task_type)
                result = graph.answer_from_evidence('Explain the measured result.', [doc(1)], task_description=task)
                self.assertEqual(result['status'], 'answered')
                self.assertFalse(result['warnings'])
                self.assertEqual(result['trace']['source_coverage']['missing_source_ids'], [sid(2)])

    def test_pack_explicit_no_targets_suppresses_missing_target_warning_without_reordering(self):
        docs = [doc(2, text='b' * 40), doc(1, text='a' * 40)]
        arguments = {'task_type': 'comparison', 'evidence_tokens': 110, 'counter': len}
        legacy = pack_evidence('Compare.', docs, **arguments)
        explicit = pack_evidence('Compare.', docs, required_source_ids=[], **arguments)
        self.assertEqual([value.metadata['doc_id'] for value in legacy.documents], [2])
        self.assertEqual([value.metadata['doc_id'] for value in explicit.documents], [2])
        self.assertTrue(any('目标文献' in warning for warning in legacy.warnings))
        self.assertFalse(any('目标文献' in warning for warning in explicit.warnings))

    def test_single_target_does_not_borrow_the_first_allowed_paper_title(self):
        task = describe_task('它的方法是什么？', source_titles=['Allowed A', 'Target B'],
            source_ids=[sid(1), sid(2)], required_source_ids=[sid(2)])
        self.assertEqual(task.resolution, 'selected_single_paper')
        self.assertNotIn('Allowed A', task.resolved_query)
        self.assertIn(sid(2), task.resolved_query)

    def test_serialized_null_required_is_not_an_implicit_scope_fallback(self):
        task = descriptor(allowed=(1,), required=(1,))
        task['required_source_ids'] = None
        with mocked_answer_path() as calls, self.assertRaises(ValueError):
            graph.answer_from_evidence('Explain.', [doc(1)], task_description=task)
        calls.model.assert_not_called()

    def test_existing_web_default_library_and_mixed_calls_keep_their_acl_contract(self):
        # Existing web source_ids records explicit selections, not the full ACL.
        for mode in ('library', 'mixed'):
            with self.subTest(mode=mode), mocked_answer_path(), ExitStack() as stack:
                runner = object.__new__(jobs.JobRunner)
                task = {'id': 'offline-task', 'owner_id': 1, 'payload': {
                    'session_id': 5, 'question': 'Explain the method.', 'group_id': 7, 'mode': mode}}
                asset = {'id': 'asset-a', 'session_id': 5, 'group_id': 9, 'doc_id': 2,
                         'paper': {'title': 'Selected asset'},
                         'file_info': {'path': 'synthetic-source-not-read', 'sha256': 'a' * 64}}
                if mode == 'mixed':
                    task['payload']['asset_ids'] = ['asset-a']
                stack.enter_context(patch.object(jobs.store, 'session_access', return_value={'id': 5, 'group_id': 7}))
                stack.enter_context(patch.object(jobs.store, 'library_access'))
                stack.enter_context(patch.object(jobs.store, 'asset', return_value=asset))
                stack.enter_context(patch.object(jobs.storage, 'list_messages', return_value=[]))
                stack.enter_context(patch.object(runner, '_progress'))
                stack.enter_context(patch.object(runner, '_check', return_value=task))
                stack.enter_context(patch.object(runner, '_source'))
                stack.enter_context(patch.object(runner, '_save_answer', side_effect=lambda task, output: output))
                retrieve = stack.enter_context(patch('refmind.rag.scientific_retrieval.retrieve_for_task',
                    side_effect=lambda group, query, **kwargs: [doc(1)] if group == 7 else [doc(2)]))
                shared_retrieve = stack.enter_context(patch('refmind.rag.research_pipeline.retrieve_research_candidates',
                    return_value={"documents": [doc(1)], "trace": {"stages": {}, "timings": {}},
                                  "warnings": [], "degraded": False, "retrieval_failed": False}))
                stack.enter_context(patch('refmind.rag.research_pipeline.run_research_rerank',
                    side_effect=lambda query, documents: documents))
                stack.enter_context(patch('refmind.rag.reranker.rerank', side_effect=lambda query, documents, **kwargs: documents))
                stack.enter_context(patch('refmind.rag.retrieval.filter_ready_documents',
                    side_effect=lambda group, documents, selected=None: documents))
                result = runner._answer(task)
                self.assertEqual(result['status'], 'answered')
                self.assertTrue(result['evidence'])
                if mode == 'library':
                    self.assertIsNone(shared_retrieve.call_args.kwargs['selected_doc_ids'])
                    retrieve.assert_not_called()
                    self.assertEqual(result['task_description']['required_source_ids'], [])
                    self.assertEqual(result['trace']['source_coverage']['allowed_source_ids'], [])
                else:
                    self.assertIsNone(retrieve.call_args_list[0].kwargs['selected_doc_ids'])
                    shared_retrieve.assert_not_called()
                    self.assertEqual(result['task_description']['required_source_ids'], ['asset:asset-a'])
                    self.assertEqual(result['trace']['source_coverage']['missing_source_ids'], [])


if __name__ == '__main__':
    unittest.main()
