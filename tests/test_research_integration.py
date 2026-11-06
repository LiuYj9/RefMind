import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from refmind.config import settings
from refmind.rag import graph, reranker, research_pipeline
from refmind.agents.orchestration import EvidenceReviewAgent, AnswerReviewAgent, MultiAgentConfig


class ResearchIntegrationTests(unittest.TestCase):
    def evidence(self):
        return Document(page_content='The measured value is 12 A at 77 K.', metadata={'doc_id': 1, 'group_id': 7, 'filename': 'paper.pdf', 'paper_title': 'Paper', 'page': 2})

    def test_final_refusal_does_not_erase_actual_generation_evidence(self):
        doc = self.evidence()
        with patch.object(graph, 'get_llm', return_value=Mock()), patch.object(graph.EvidenceReviewAgent, 'review', side_effect=lambda q, ds: (ds, None)), patch.object(graph, '_generate_answer', return_value='A draft') as generate, patch.object(graph, '_run_text_hook', side_effect=lambda _hook, answer, **kwargs: answer), patch.object(graph.AnswerReviewAgent, 'review', return_value=(graph.NO_CONTEXT_REPLY, None)):
            result = graph.answer_from_evidence('How much at 77 K?', [doc])
        stages = result['trace']['stages']
        self.assertEqual(len(stages['packed']['documents']), 1)
        self.assertEqual(stages['generation_context']['documents'], generate.call_args.args[1])
        self.assertEqual(stages['final_evidence']['documents'], [])
        self.assertEqual(result['status'], 'insufficient')

    def test_rejected_pack_is_not_counted_as_generated_context(self):
        with patch.object(graph, 'get_llm', return_value=Mock()), patch.object(graph.EvidenceReviewAgent, 'review', return_value=([], None)):
            result = graph.answer_from_evidence('How much?', [self.evidence()])
        self.assertEqual(len(result['trace']['stages']['packed']['documents']), 1)
        self.assertEqual(result['trace']['stages']['generation_context']['documents'], [])

    def test_refusal_recheck_toggle_actually_changes_call_count(self):
        response = SimpleNamespace(content=json.dumps({'coverage': 'insufficient', 'answerable': False}))
        for enabled in (False, True):
            llm = Mock()
            llm.invoke.return_value = response
            config = MultiAgentConfig(enable_evidence_review=True, enable_answer_review=True, confirm_refusal=enabled)
            EvidenceReviewAgent(llm, config).review('q', [self.evidence()])
            self.assertEqual(llm.invoke.call_count, 2 if enabled else 1)
            llm.reset_mock()
            AnswerReviewAgent(llm, config).review('q', [self.evidence()], 'draft')
            self.assertEqual(llm.invoke.call_count, 2 if enabled else 1)

    def test_rerank_fallback_is_distinguished_from_cross_encoder(self):
        doc = self.evidence()
        with patch.object(settings, 'rerank_enabled', True), patch.object(settings, 'dashscope_api_key', ''), patch.object(reranker, '_embedding_rerank', return_value=[(.8, doc)]):
            result = reranker.rerank('q', [doc])
        self.assertEqual(result[0].metadata['rerank_backend'], 'embedding_similarity')
        self.assertTrue(result[0].metadata['rerank_degraded'])
        self.assertNotIn('rerank_backend', doc.metadata)
        with patch.object(settings, 'rerank_enabled', True), patch.object(settings, 'dashscope_api_key', ''), patch.object(reranker, '_embedding_rerank', side_effect=RuntimeError('offline')):
            result = reranker.rerank('q', [doc])
        self.assertEqual(result[0].metadata['rerank_backend'], 'recall_order')

    def test_production_wrapper_passes_selected_scope_and_resolved_query(self):
        row = SimpleNamespace(group_id=7, status='ready', paper_title='Selected Paper', filename='p.pdf')
        doc = self.evidence()
        with patch.object(research_pipeline.storage, 'get_document', return_value=row), patch.object(research_pipeline, 'retrieve_for_task', return_value=[doc]) as retrieve, patch.object(research_pipeline, 'rerank', side_effect=lambda q, ds, **kw: ds):
            result = research_pipeline.retrieve_research_evidence('它的方法是什么？', 7, selected_doc_ids=[1])
        self.assertEqual(retrieve.call_args.kwargs['selected_doc_ids'], (1,))
        self.assertIn('Selected Paper', retrieve.call_args.args[1])
        self.assertEqual(result['trace']['pipeline'], 'research')

    def test_permission_revocation_after_review_prevents_generation(self):
        def revoked(_documents):
            raise PermissionError('membership revoked')
        with patch.object(graph, 'get_llm', return_value=Mock()), patch.object(graph.EvidenceReviewAgent, 'review', side_effect=lambda q, ds: (ds, None)), patch.object(graph, '_generate_answer') as generate:
            with self.assertRaises(PermissionError):
                graph.answer_from_evidence('How much?', [self.evidence()], evidence_guard=revoked)
        generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
