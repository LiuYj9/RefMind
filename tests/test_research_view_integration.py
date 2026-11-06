"""Offline checks of the actual view -> review -> generation boundary."""
from unittest import TestCase
from unittest.mock import patch
from types import SimpleNamespace
from urllib.parse import parse_qs
from langchain_core.documents import Document
from tests.test_scope_targets_contract import doc, descriptor, mocked_answer_path
from refmind.config import settings
from refmind.rag import graph, extractive_views, reranker
from refmind.citations import build_citation_href


class ResearchViewIntegration(TestCase):
    def test_review_and_generation_consume_same_view_not_original(self):
        original=doc(1, text='Original full paragraph.')
        view=doc(1, text='Kept sentence.\n[… omitted …]\nKept condition.')
        view.metadata.update(evidence_view='extractive_text', selected_spans=[{'start':0,'end':5,'text':'Kept'}])
        result=SimpleNamespace(documents=[view], records=[{'status':'extracted'}], coverage_limit={})
        with (mocked_answer_path() as calls,
              patch.object(settings,'research_extractive_compression_enabled',True,create=True),
              patch('refmind.llm.get_embedding_model',return_value=object()),
              patch.object(extractive_views,'make_embedding_scorer',return_value=lambda *a: []),
              patch.object(extractive_views,'build_extractive_views',return_value=result)):
            answer=graph.answer_from_evidence('Explain.',[original],task_description=descriptor(allowed=(1,),required=(1,)),apply_research_enhancements=True)
        reviewed=calls.evidence_review.call_args.args[1][0]
        generated=calls.generate.call_args.args[1][0]
        self.assertEqual(reviewed.page_content,view.page_content)
        self.assertEqual(generated.page_content,view.page_content)
        self.assertEqual(generated.metadata['selected_spans'],view.metadata['selected_spans'])
        self.assertTrue(answer['trace']['compression']['enabled'])

    def test_legacy_entry_does_not_activate_view_even_with_feature_flag(self):
        with (mocked_answer_path(),patch.object(settings,'research_extractive_compression_enabled',True,create=True),
              patch.object(extractive_views,'build_extractive_views') as build):
            graph.answer_from_evidence('Explain.',[doc(1)],task_description=descriptor(allowed=(1,),required=(1,)))
        build.assert_not_called()

    def test_resource_same_page_citations_keep_chunk_identity(self):
        hrefs=[build_citation_href(Document(page_content='source',metadata={
            'resource_id':'r1','evidence_level':'full_text','page':8,'chunk_id':key,'paragraph_index':i}))
            for i,key in enumerate(['a','b'],1)]
        self.assertNotEqual(hrefs[0],hrefs[1])
        self.assertEqual(parse_qs(hrefs[1][1:])['chunk_id'],['b'])

    def test_capacity_falls_back_whole_batch_without_truncating_candidates(self):
        docs=[doc(1,'x'*5000),doc(2)]
        with (patch.object(settings,'rerank_enabled',True),patch.object(settings,'rerank_model','gte-rerank-v2'),
              patch.object(settings,'dashscope_api_key','test'),patch.object(reranker,'dashscope_sdk_available',return_value=True),
              patch.object(reranker,'_dashscope_rerank') as service,
              patch.object(reranker,'_embedding_rerank',return_value=[(.7,docs[0]),(.5,docs[1])])):
            ranked=reranker.rerank('query',docs,top_n=2)
        service.assert_not_called()
        self.assertEqual(len(ranked),2)
        self.assertEqual(ranked[0].page_content,docs[0].page_content)
        self.assertEqual(ranked[0].metadata['rerank_backend'],'embedding_similarity')
        self.assertEqual(ranked[0].metadata['rerank_capacity']['estimator'],'utf8_bytes_upper_bound')
