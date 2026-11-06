"""Synthetic channel-rank contracts; real installed LangChain RRF is the oracle."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import sys
import types
from threading import Barrier
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from refmind.rag import retrieval, scientific_retrieval, research_pipeline
from refmind.rag import document_processor


def doc(number, text=None, *, index=0, version='v1', **metadata):
    return Document(page_content=text or f'Synthetic evidence {number}/{index}.', metadata={
        'group_id': 7, 'doc_id': number, 'filename': f'synthetic-{number}.pdf', 'version': version,
        'chunk_id': f'chunk-{number}-{index}', 'chunk_index': index, 'page': 1,
        'section': 'Methods', **metadata})


def identity(document):
    metadata = document.metadata
    return tuple(metadata.get(name) for name in ('group_id', 'doc_id', 'version', 'chunk_id')), document.page_content


def snapshot(documents):
    return [(value.page_content, copy.deepcopy(value.metadata)) for value in documents]


class FixtureRetriever(BaseRetriever):
    responses: dict[str, list[Document]]
    barrier: Any = None
    on_invoke: Any = None
    k: int = 20

    def _get_relevant_documents(self, query, *, run_manager):
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        if self.on_invoke is not None:
            self.on_invoke()
        return self.responses.get(query, self.responses.get('*', []))


class FixtureScope(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.rows = {number: SimpleNamespace(id=number, group_id=7, status='ready',
            index_version=None, file_sha256=None, paper_title=f'Paper {number}', filename=f'synthetic-{number}.pdf') for number in range(1, 11)}
        self.stack.enter_context(patch.object(retrieval.storage, 'list_documents', side_effect=lambda group: list(self.rows.values()) if group == 7 else []))
        self.stack.enter_context(patch.object(research_pipeline.storage, 'get_document', side_effect=self.rows.get))
        retrieval.reset_retrievers()

    def build(self, bm25, vector, *, selected=(1, 2, 3), barrier=None):
        left = FixtureRetriever(responses=bm25, barrier=barrier)
        right = FixtureRetriever(responses=vector)
        store = Mock(name='offline_vectorstore')
        store.as_retriever.return_value = right
        self.stack.enter_context(patch.object(retrieval, 'get_vectorstore', return_value=store))
        self.stack.enter_context(patch.object(retrieval.BM25Retriever, 'from_documents', return_value=left))
        result = retrieval.build_retriever(7, documents=[doc(number) for number in selected], selected_doc_ids=selected)
        return result, left, right, store

    def library_oracle(self, left, right, query, *, selected=(1, 2, 3)):
        # Exactly the installed implementation; no hand-implemented RRF formula.
        branches = [retrieval._ScopedRetriever(source=value, group_id=7, allowed_ids=selected)
                    for value in (left, right)]
        return EnsembleRetriever(retrievers=branches, weights=[.5, .5], c=60,
                                 id_key='_retrieval_identity').invoke(query)


class ChannelFusionContractTests(FixtureScope):
    def test_overlap_keeps_both_authorized_branch_ranks_without_scores(self):
        a, b, c = doc(1), doc(2), doc(3)
        result, left, right, _ = self.build({'*': [a, b]}, {'*': [c, a]})
        actual = result.invoke('query')
        expected = self.library_oracle(left, right, 'query')
        self.assertEqual([identity(value) for value in actual], [identity(value) for value in expected])
        overlap = next(value for value in actual if value.metadata['doc_id'] == 1)
        self.assertEqual(overlap.metadata.get('retrieval_channels'), {'bm25': {'rank': 1}, 'vector': {'rank': 2}})
        for rank, value in enumerate(actual, 1):
            self.assertTrue(all(set(hit) == {'rank'} for hit in value.metadata['retrieval_channels'].values()))
            self.assertEqual(value.metadata['retrieval_fusion'], {'method': 'weighted_rrf', 'rank': rank,
                'c': 60, 'weights': {'bm25': .5, 'vector': .5}, 'channels': ['bm25', 'vector']})

    def test_same_text_different_papers_and_versions_remain_distinct(self):
        a = doc(1, 'Identical scientific text.')
        b = doc(2, 'Identical scientific text.')
        c = doc(1, 'Identical scientific text.', version='v2')
        result, left, right, _ = self.build({'*': [a, b]}, {'*': [c, b]})
        actual = result.invoke('query')
        self.assertEqual(len(actual), 3)
        self.assertEqual(len({value.metadata['_retrieval_identity'] for value in actual}), 3)
        self.assertEqual([identity(value) for value in actual], [identity(value) for value in self.library_oracle(left, right, 'query')])

    def test_ties_and_backend_duplicates_follow_the_real_library(self):
        a, b = doc(1), doc(2)
        for left_docs, right_docs in (([a, b], [b, a]), ([a, a, b], [b])):
            with self.subTest(duplicates=len(left_docs) == 3):
                result, left, right, _ = self.build({'*': left_docs}, {'*': right_docs})
                actual = result.invoke('query')
                self.assertEqual([identity(value) for value in actual], [identity(value) for value in self.library_oracle(left, right, 'query')])
                first = next(value for value in actual if value.metadata['doc_id'] == 1)
                self.assertEqual(first.metadata.get('retrieval_channels', {}).get('bm25'), {'rank': 1})

    def test_calls_replace_stale_channels_and_do_not_mutate_sources_or_prior_results(self):
        a = doc(1, retrieval_channels={'stale': {'rank': 99}}, retrieval_fusion={'method': 'stale'})
        b = doc(2)
        originals = snapshot([a, b])
        result, _, _, _ = self.build({'q1': [a, b], 'q2': [b, a]}, {'q1': [b, a], 'q2': [a]})
        first = result.invoke('q1')
        first_snapshot = snapshot(first)
        second = result.invoke('q2')
        self.assertEqual(snapshot(first), first_snapshot)
        self.assertEqual(snapshot([a, b]), originals)
        a_second = next(value for value in second if value.metadata['doc_id'] == 1)
        self.assertEqual(a_second.metadata.get('retrieval_channels'), {'bm25': {'rank': 2}, 'vector': {'rank': 1}})
        first[0].metadata['retrieval_channels']['bm25']['rank'] = 999
        first[0].metadata['retrieval_fusion']['weights']['bm25'] = 999
        self.assertEqual(a_second.metadata['retrieval_channels']['bm25']['rank'], 2)
        self.assertEqual(second[0].metadata['retrieval_fusion']['weights']['bm25'], .5)
        self.assertEqual(first[1].metadata['retrieval_fusion']['weights']['bm25'], .5)
        self.assertEqual(snapshot([a, b]), originals)

    def test_two_concurrent_queries_do_not_exchange_channel_ranks(self):
        a, b = doc(1), doc(2)
        result, _, _, _ = self.build({'q1': [a, b], 'q2': [b, a]}, {'q1': [b], 'q2': [a]}, barrier=Barrier(2))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(result.invoke, query) for query in ('q1', 'q2')]
            first, second = [future.result(timeout=10) for future in futures]
        by_id_first = {value.metadata['doc_id']: value for value in first}
        by_id_second = {value.metadata['doc_id']: value for value in second}
        self.assertEqual(by_id_first[1].metadata.get('retrieval_channels'), {'bm25': {'rank': 1}})
        self.assertEqual(by_id_second[1].metadata.get('retrieval_channels'), {'bm25': {'rank': 2}, 'vector': {'rank': 1}})
        self.assertEqual(by_id_first[2].metadata.get('retrieval_channels'), {'bm25': {'rank': 2}, 'vector': {'rank': 1}})

    def test_rank_is_after_scope_filter_and_revoked_rows_stay_filtered(self):
        outside, wrong_group, not_ready, valid = doc(9), doc(1, group_id=99), doc(3), doc(1)
        self.rows[3].status = 'indexing'
        result, _, right, store = self.build({'*': [outside, wrong_group, not_ready, valid]}, {'*': [outside, valid]}, selected=(1, 2, 3))
        actual = result.invoke('query')
        self.assertEqual([value.metadata['doc_id'] for value in actual], [1])
        self.assertEqual(actual[0].metadata.get('retrieval_channels'), {'bm25': {'rank': 1}, 'vector': {'rank': 1}})
        self.assertEqual(store.as_retriever.call_args.kwargs['search_kwargs']['filter'], {'doc_id': {'$in': [1, 2]}})
        right.on_invoke = lambda: setattr(self.rows[1], 'status', 'indexing')
        self.assertEqual(result.invoke('query'), [])


def fusion(rank):
    return {'method': 'weighted_rrf', 'rank': rank, 'c': 60,
            'weights': {'bm25': .5, 'vector': .5}, 'channels': ['bm25', 'vector']}


class ExpansionChannelContractTests(FixtureScope):
    def expanded(self):
        first = doc(1, index=1, _retrieval_identity='direct-first',
            retrieval_channels={'bm25': {'rank': 1}}, retrieval_fusion=fusion(1))
        later = doc(1, index=2, _retrieval_identity='direct-later',
            retrieval_channels={'bm25': {'rank': 2}, 'vector': {'rank': 1}}, retrieval_fusion=fusion(2))
        pool = [doc(1, index=index, retrieval_channels={'stale': {'rank': 99}},
                    retrieval_fusion={'method': 'stale'}) for index in range(4)]
        before = snapshot([first, later, *pool])
        source = FixtureRetriever(responses={'*': [first, later]})
        with patch.object(retrieval, 'get_retriever', return_value=source), patch.object(document_processor, 'load_group_documents', return_value=pool):
            actual = scientific_retrieval.retrieve_for_task(7, 'synthetic query', selected_doc_ids=(1,))
        self.assertEqual(snapshot([first, later, *pool]), before)
        return actual, first, later, pool

    def test_neighbor_first_later_direct_keeps_both_new_metadata_fields(self):
        actual, _, later, _ = self.expanded()
        self.assertEqual([value.metadata['chunk_index'] for value in actual], [1, 0, 2, 3])
        restored = next(value for value in actual if value.metadata['chunk_index'] == 2)
        self.assertEqual(restored.metadata.get('retrieval_channels'), later.metadata['retrieval_channels'])
        self.assertEqual(restored.metadata.get('retrieval_fusion'), later.metadata['retrieval_fusion'])
        self.assertEqual(restored.metadata['retrieval_origin']['direct_rank'], 2)
        self.assertFalse(restored.metadata['retrieval_neighbor'])

    def test_neighbor_only_does_not_borrow_or_keep_stale_direct_channels(self):
        actual, _, _, _ = self.expanded()
        for value in actual:
            if value.metadata['retrieval_origin']['direct_rank'] is None:
                self.assertNotIn('retrieval_channels', value.metadata)
                self.assertNotIn('retrieval_fusion', value.metadata)

    def test_later_direct_nested_metadata_is_deeply_copied(self):
        actual, _, later, pool = self.expanded()
        restored = next(value for value in actual if value.metadata['chunk_index'] == 2)
        restored.metadata['retrieval_channels']['bm25']['rank'] = 777
        restored.metadata['retrieval_fusion']['weights']['bm25'] = 777
        self.assertEqual(later.metadata['retrieval_channels']['bm25']['rank'], 2)
        self.assertEqual(later.metadata['retrieval_fusion']['weights']['bm25'], .5)
        self.assertEqual(pool[2].metadata['retrieval_channels'], {'stale': {'rank': 99}})


class PipelineChannelTraceContractTests(FixtureScope):
    def traced(self, documents):
        with patch.object(research_pipeline, 'retrieve_for_task', return_value=documents), patch.object(research_pipeline, 'rerank', side_effect=lambda query, docs, **kwargs: docs):
            return research_pipeline.retrieve_research_evidence('Explain the synthetic method.', 7, allowed_doc_ids=[1, 2, 3])

    def test_trace_lists_observed_direct_hits_in_true_per_channel_rank_order(self):
        a = doc(1, retrieval_origin={'direct_rank': 1}, retrieval_channels={'bm25': {'rank': 2}, 'vector': {'rank': 1}}, retrieval_fusion=fusion(1))
        b = doc(2, retrieval_origin={'direct_rank': 2}, retrieval_channels={'bm25': {'rank': 1}}, retrieval_fusion=fusion(2))
        neighbor = doc(3, retrieval_origin={'direct_rank': None, 'section_coverage': True}, retrieval_channels={'vector': {'rank': 999}}, retrieval_fusion=fusion(999))
        trace = self.traced([a, neighbor, b])['trace']
        self.assertEqual(trace.get('retrieval_channels'), {'available': True, 'rank_scope': 'authorized_inputs_to_fusion'})
        self.assertEqual([identity(value) for value in trace['stages']['bm25']['documents']], [identity(b), identity(a)])
        self.assertEqual([identity(value) for value in trace['stages']['vector']['documents']], [identity(a)])
        for name in ('bm25', 'vector'):
            self.assertFalse(trace['stages'][name]['scores_collected'])
            self.assertEqual(trace['stages'][name]['rank_scope'], 'authorized_inputs_to_fusion')

    def test_legacy_empty_and_neighbor_only_inputs_are_unavailable_not_fake_empty_hits(self):
        for documents in ([doc(1)], [], [doc(1, retrieval_origin={'direct_rank': None, 'section_coverage': True})]):
            with self.subTest(case=len(documents)):
                trace = self.traced(documents)['trace']
                info = trace.get('retrieval_channels') or {}
                self.assertIs(info.get('available'), False)
                self.assertTrue(info.get('reason'))
                for name in ('bm25', 'vector'):
                    self.assertNotIn('documents', trace['stages'].get(name, {}))

    def test_mixed_missing_direct_metadata_is_incomplete_not_partial_observation(self):
        a = doc(1, retrieval_origin={'direct_rank': 1}, retrieval_channels={'bm25': {'rank': 1}}, retrieval_fusion=fusion(1))
        b = doc(2, retrieval_origin={'direct_rank': 2})
        trace = self.traced([a, b])['trace']
        self.assertEqual(trace.get('retrieval_channels', {}).get('available'), False)
        self.assertEqual(trace.get('retrieval_channels', {}).get('reason'), 'incomplete_channel_metadata')
        self.assertNotIn('documents', trace['stages'].get('bm25', {}))
        self.assertNotIn('documents', trace['stages'].get('vector', {}))

    def test_answer_wrapper_preserves_observation_available_or_unavailable(self):
        observed = doc(1, retrieval_origin={'direct_rank': 1},
            retrieval_channels={'bm25': {'rank': 1}}, retrieval_fusion=fusion(1))
        for documents, available in (([observed], True), ([doc(1)], False)):
            with self.subTest(available=available):
                fake_graph = types.ModuleType('refmind.rag.graph')
                fake_graph.answer_from_evidence = Mock(return_value={'answer': 'Synthetic answer.',
                    'status': 'answered', 'documents': documents, 'trace': {'stages': {}, 'timings': {}}})
                with (patch.dict(sys.modules, {'refmind.rag.graph': fake_graph}),
                      patch.object(research_pipeline, 'retrieve_for_task', return_value=documents),
                      patch.object(research_pipeline, 'rerank', side_effect=lambda query, docs, **kwargs: docs)):
                    result = research_pipeline.answer_research_question('Explain the method.', 7, allowed_doc_ids=[1])
                info = result['trace'].get('retrieval_channels') or {}
                self.assertIs(info.get('available'), available)
                if available:
                    self.assertEqual(result['trace']['stages']['bm25']['documents'], documents)
                    self.assertEqual(result['trace']['stages']['vector']['documents'], [])
                else:
                    self.assertTrue(info.get('reason'))
                    self.assertNotIn('documents', result['trace']['stages'].get('bm25', {}))
                fake_graph.answer_from_evidence.assert_called_once()


if __name__ == '__main__':
    unittest.main()
