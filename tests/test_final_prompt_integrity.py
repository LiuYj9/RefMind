"""Offline post-review budget and final citation destination regressions."""
from contextlib import contextmanager
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from refmind.citations import build_citation_href, build_citation_label
from refmind.config import settings
from refmind.rag import graph
from refmind.rag.evidence_bundle import enforce_generation_budget


def evidence(number=1, text="Observed result.", **metadata):
    return Document(page_content=text, metadata={"group_id": 7, "doc_id": number,
        "paper_title": f"Paper {number}", "page": 2, "paragraph_index": number, **metadata})


class CitationDestinationTests(unittest.TestCase):
    def test_unknown_compact_local_destinations_are_rejected(self):
        for body in ("Claim [[1]](?open_citation=g999-d999-p2-r1)",
                     "Claim [1](https://unknown.example/paper)",
                     "Claim [[1]](javascript:alert(1))", "Claim [[1]]("):
            with self.subTest(body=body):
                answer, docs, warnings = graph._finalize_grounded_answer(body, [evidence()])
                self.assertEqual(answer, graph.NO_CONTEXT_REPLY)
                self.assertEqual(docs, [])
                self.assertTrue(warnings)

    def test_valid_compact_links_are_bound_by_destination_before_renumbering(self):
        docs = [evidence(1), evidence(2)]
        # The display number may be 1 although its destination is evidence 2.
        answer, final_docs, warnings = graph._finalize_grounded_answer(
            f'Claim [[1]]({build_citation_href(docs[1])} "old tooltip")', docs)
        self.assertEqual(final_docs, docs)
        self.assertEqual(warnings, ())
        self.assertIn('[[1]](?open_citation=g7-d2-p2-r2', answer)
        self.assertNotIn('g7-d1-', answer)
        self.assertNotIn('old tooltip', answer)

    def test_session_resource_requires_exact_page(self):
        doc = evidence(resource_id="asset-abc", evidence_level="full_text")
        good = f'Claim [1]({build_citation_href(doc)})'
        answer, docs, warnings = graph._finalize_grounded_answer(good, [doc])
        self.assertEqual(docs, [doc])
        self.assertIn('open_resource=asset-abc&page=2', answer)
        self.assertFalse(warnings)
        answer, docs, warnings = graph._finalize_grounded_answer(
            good.replace('page=2', 'page=3'), [doc])
        self.assertEqual(answer, graph.NO_CONTEXT_REPLY)
        self.assertEqual(docs, [])

    def test_external_doi_parentheses_and_quoted_title_are_supported(self):
        doc = evidence(evidence_origin="academic_search", evidence_level="abstract",
                       external_url="https://doi.org/10.1000/example(2025)")
        answer, docs, warnings = graph._finalize_grounded_answer(
            f'Claim [[1]]({build_citation_href(doc)} "known (source)")', [doc])
        self.assertEqual(docs, [doc])
        self.assertFalse(warnings)
        self.assertIn('https://doi.org/10.1000/example(2025)', answer)
        self.assertNotIn('known (source)', answer)

    def test_wrong_paragraph_and_out_of_range_number_are_rejected(self):
        doc = evidence()
        for body in (f'Claim [[2]]({build_citation_href(doc)})',
                     f'Claim [[1]]({build_citation_href(doc).replace("-r1", "-r2")})'):
            answer, docs, warnings = graph._finalize_grounded_answer(body, [doc])
            self.assertEqual(answer, graph.NO_CONTEXT_REPLY)
            self.assertEqual(docs, [])
            self.assertTrue(warnings)

    def test_refusal_never_preserves_compact_links_or_bibliography(self):
        doc = evidence()
        rendered = graph.NO_CONTEXT_REPLY + '\n\n---\n\n### 参考来源\n\n' + f'[[1]]({build_citation_href(doc)})'
        answer, docs, warnings = graph._finalize_grounded_answer(rendered, [doc])
        self.assertEqual(answer, graph.NO_CONTEXT_REPLY)
        self.assertEqual(docs, [])
        self.assertEqual(warnings, ())


class ActualPromptBudgetTests(unittest.TestCase):
    def test_review_scope_is_once_but_each_table_view_keeps_its_limit(self):
        scope = 'Only the measured condition is supported.'
        docs = [evidence(n, review_scope_limit=scope, coverage_limit=f'rows {n}') for n in (1, 2)]
        rendered = graph.format_documents(docs)
        self.assertEqual(rendered.count(scope), 1)
        self.assertIn('rows 1', rendered)
        self.assertIn('rows 2', rendered)
        self.assertEqual(rendered.count('【本轮证据适用范围】'), 1)

    def test_final_budget_skips_whole_atomic_block_and_can_admit_later_blocks(self):
        docs = [evidence(1, 'x' * 200, is_atomic=True),
                evidence(2, 'a' * 40, is_atomic=True), evidence(3, 'b' * 60)]
        def measure(actual, history):
            size = sum(len(d.page_content) for d in actual)
            return {'prompt_tokens': 20 + size, 'evidence_tokens': size, 'image_token_reserve': 0}
        bundle = enforce_generation_budget(docs, [], measure=measure,
            evidence_tokens=100, model_context_tokens=200, output_reserve=30)
        self.assertEqual(bundle.documents, docs[1:])
        self.assertEqual(docs[0].page_content, 'x' * 200)
        self.assertEqual(bundle.trace['evidence_tokens'], 100)
        self.assertEqual(bundle.trace['omitted'][0]['reason'], 'post_review_payload_exceeds_budget')

    def test_final_budget_drops_complete_history_pairs(self):
        history = [HumanMessage(content='old' * 40), AIMessage(content='answer' * 40)]
        def measure(actual, messages):
            text = sum(len(d.page_content) for d in actual)
            return {'prompt_tokens': 20 + text + sum(len(m.content) for m in messages),
                    'evidence_tokens': text, 'image_token_reserve': 0}
        bundle = enforce_generation_budget([evidence(text='x' * 70)], history, measure=measure,
            evidence_tokens=100, model_context_tokens=150, output_reserve=30)
        self.assertEqual(bundle.history, [])
        self.assertEqual(bundle.trace['dropped_history_pairs'], 1)
        self.assertEqual(len(bundle.documents), 1)
        self.assertLessEqual(bundle.trace['prompt_tokens'] + 30, 150)

    def test_review_scope_repacking_matches_actual_model_messages_and_trace(self):
        scope = 'bound ' * 200
        original = [evidence(1, 'A' * 600), evidence(2, 'B' * 600)]
        calls, stages = [], []

        class ScopeReviewer:
            def __init__(self, *_args):
                self.last_review = {'coverage': 'scoped', 'scope_limit': scope}
            def review(self, _question, docs):
                return docs, None

        class PassReviewer:
            def __init__(self, *_args):
                pass
            def review(self, _question, _docs, answer):
                return answer, None

        def model(prompt):
            calls.append(prompt.to_messages())
            return AIMessage(content='Supported. ' + build_citation_label(original[0], 1))

        @contextmanager
        def stage(name):
            stages.append(name)
            yield

        history = [HumanMessage(content='Earlier question'), AIMessage(content='Earlier answer')]
        with (patch.object(graph, 'get_llm', return_value=RunnableLambda(model)),
              patch.object(graph, 'EvidenceReviewAgent', ScopeReviewer),
              patch.object(graph, 'AnswerReviewAgent', PassReviewer),
              patch.object(graph, 'build_visual_content', return_value=[]),
              patch.object(graph, '_run_text_hook', side_effect=lambda _hook, text, **_: text),
              patch.object(graph, 'token_counter', return_value=(len, 'characters_for_test')),
              patch.object(graph, 'billing_stage', side_effect=stage),
              patch.object(settings, 'context_max_tokens', 1900),
              patch.object(settings, 'model_context_tokens', 8192),
              patch.object(settings, 'answer_token_reserve', 512)):
            result = graph.answer_from_evidence('What was measured?', original, history=history)

        self.assertEqual(stages, ['evidence_review', 'generation', 'answer_review'])
        self.assertEqual(len(calls), 1)
        snapshot = result['trace']['stages']['generation_context']
        self.assertTrue(snapshot['request_sent'])
        self.assertEqual(len(snapshot['documents']), 1)
        self.assertEqual(calls[0][0].content, snapshot['system_prompt'])
        self.assertEqual(calls[0][-1].content, snapshot['user_prompt'])
        self.assertEqual([m.content for m in calls[0][1:-1]], [m.content for m in history])
        self.assertEqual(snapshot['rendered_evidence'], graph.format_documents(snapshot['documents']))
        self.assertEqual(snapshot['rendered_evidence'].count(scope.strip()), 1)
        packing = result['trace']['packing']
        self.assertEqual(packing['evidence_tokens'], len(snapshot['rendered_evidence']))
        self.assertLessEqual(packing['evidence_tokens'], packing['evidence_budget'])
        self.assertLessEqual(packing['prompt_tokens'] + packing['image_token_reserve'] + 512, 8192)
        self.assertEqual(len(packing['omitted']), 1)
        self.assertTrue(packing['post_review_checked'])
        self.assertIn('A' * 600, snapshot['user_prompt'])
        self.assertNotIn('B' * 600, snapshot['user_prompt'])
        self.assertNotIn('review_scope_limit', original[0].metadata)


if __name__ == '__main__':
    unittest.main()
