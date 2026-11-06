"""Synthetic scope/intent contract tests; no real QA, model, storage or application imports."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

WORKER_PATH = Path(os.getenv("REFMIND_SCOPE_TEST_WORKER", str(Path(__file__).resolve().parents[1] / "evaluation/worker.py"))).resolve()
spec = importlib.util.spec_from_file_location("evaluation._scope_intent_test_worker", WORKER_PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def module(name, **values):
    value = ModuleType(name)
    value.__dict__.update(values)
    return value


@contextmanager
def fake_application(settings=None):
    """Fake every refmind import; only real LangChain message value types are used."""
    memory = Mock(session_id=17)
    storage = module("refmind.storage", init_db=Mock(), create_session=Mock(return_value=SimpleNamespace(id=17)))
    graph = module("refmind.rag.graph", NO_CONTEXT_REPLY="fixture insufficient", ACADEMIC_NO_CONTEXT_REPLY="fixture abstract insufficient",
                   answer_research_question=Mock(return_value={"answer": "fixture answer", "status": "answered", "documents": []}),
                   retrieve_research_evidence=Mock(return_value={"documents": []}))
    retriever = Mock()
    retriever.invoke.return_value = []
    legacy = Mock(return_value={"answer": "fixture legacy answer", "status": "answered", "documents": []})
    rag = module("refmind.rag", graph=graph, answer_question=legacy, build_retriever=Mock(return_value=retriever))
    root = module("refmind", storage=storage, rag=rag)
    fake_modules = {"refmind": root, "refmind.storage": storage, "refmind.rag": rag,
                    "refmind.rag.graph": graph,
                    "refmind.rag.memory": module("refmind.rag.memory", RelevantMemory=Mock(return_value=memory)),
                    "refmind.config": module("refmind.config", settings=settings),
                    "refmind.agents": module("refmind.agents"),
                    "refmind.agents.orchestration": module("refmind.agents.orchestration", canonicalize_retrieval_query=lambda value: value)}
    with patch.dict(sys.modules, fake_modules):
        yield SimpleNamespace(graph=graph, legacy=legacy, rag=rag, storage=storage)


class SingleTurnScopeTests(unittest.TestCase):
    def run_worker(self, mode, pipeline):
        with tempfile.TemporaryDirectory(prefix="evaluation-intent-fixture-") as temporary:
            workspace = Path(temporary).resolve()
            (workspace / ".refmind-eval").write_text("synthetic fixture", encoding="utf-8")
            snapshot = {"dataset_hash": "fixture-hash", "question_splits": ["dev"], "group_id": 7,
                        "source_doc_ids": {"source-a": 101, "source-b": 202, "source-c": 303}}
            (workspace / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
            values = {name: workspace / name for name in ("database_path", "upload_dir", "parsed_dir", "docstore_dir", "chroma_persist_dir")}
            values.update(project_root=workspace, llm_model="fixture", embedding_model="fixture", rerank_model="fixture", rerank_enabled=False,
                          multi_agent_enabled=False, recall_top_k=5, retrieval_top_k=5, rerank_top_n=3, context_max_chars=2000,
                          context_compression_enabled=True, context_max_tokens=6000, model_context_tokens=32768,
                          answer_token_reserve=4096, refusal_confirmation_enabled=False)
            # No evidence or ground_truth attributes exist: neither can manufacture targets.
            item = SimpleNamespace(question="synthetic standalone request", track="local", question_split="dev",
                                   corpus_scope=("source-a", "source-b", "source-c"))
            selected = SimpleNamespace(items=[item], require_review=Mock(return_value={"official": False}))
            golden = SimpleNamespace(schema_version=3, select=Mock(return_value=selected))
            def evaluate(_golden, *, component, **_kwargs):
                component(item)
                return {"schema_version": 3, "status": "completed"}
            with ExitStack() as stack:
                app = stack.enter_context(fake_application(SimpleNamespace(**values)))
                stack.enter_context(patch.dict(os.environ, {"REFMIND_EVAL_WORKSPACE": str(workspace)}))
                stack.enter_context(patch("evaluation.dataset.load_golden_set", return_value=golden))
                stack.enter_context(patch("evaluation.dataset.verify_corpus"))
                stack.enter_context(patch("evaluation.evaluator.evaluate_items", side_effect=evaluate))
                stack.enter_context(patch.object(worker, "fingerprint", return_value={"dataset_hash": "fixture-hash"}))
                worker.run({"mode": mode, "pipeline": pipeline, "golden": "synthetic", "split": "dev",
                            "allow_draft": True, "k_values": [1, 3]})
                return app

    def test_research_retrieval_corpus_is_allowed_without_invented_selection(self):
        app = self.run_worker("retrieval", "research")
        call = app.graph.retrieve_research_evidence.call_args
        self.assertEqual(call.kwargs, {"allowed_doc_ids": [101, 202, 303]})
        app.rag.build_retriever.assert_not_called()

    def test_research_answer_corpus_is_allowed_without_invented_selection(self):
        app = self.run_worker("answer", "research")
        self.assertEqual(app.graph.answer_research_question.call_args.kwargs, {"allowed_doc_ids": [101, 202, 303]})
        app.legacy.assert_not_called()

    def test_legacy_retrieval_and_answer_keep_original_scope_arguments(self):
        for mode in ("retrieval", "answer"):
            with self.subTest(mode=mode):
                app = self.run_worker(mode, "legacy")
                if mode == "retrieval":
                    self.assertEqual(app.rag.build_retriever.call_args.kwargs["selected_doc_ids"], (101, 202, 303))
                else:
                    self.assertEqual(app.legacy.call_args.kwargs, {"selected_doc_ids": [101, 202, 303]})
                app.graph.answer_research_question.assert_not_called()
                app.graph.retrieve_research_evidence.assert_not_called()


class DialogueScopeTests(unittest.TestCase):
    def request(self, actions=None):
        return {"scenario_id": "synthetic-scope-scenario", "question": "synthetic follow-up",
                "corpus_scope": ["source-a", "source-b"], "input": actions or {}}

    def test_default_corpus_remains_only_allowed_on_followup(self):
        with fake_application() as app:
            component = worker.dialogue_component(7, {"source-a": 101, "source-b": 202}, pipeline="research")
            result = component(self.request(), {})
            self.assertIsNone(result["state"]["selected_source_ids"])
            self.assertFalse(result["state"]["selected_source_ids_explicit"])
            result = component(self.request(), result["state"])
            for call in app.graph.answer_research_question.call_args_list:
                self.assertEqual(call.kwargs["allowed_doc_ids"], [101, 202])
                self.assertNotIn("selected_doc_ids", call.kwargs)

    def test_explicit_selection_persists_and_empty_selection_never_falls_back(self):
        with fake_application() as app:
            component = worker.dialogue_component(7, {"source-a": 101, "source-b": 202}, pipeline="research")
            result = component(self.request({"selected_source_ids": ["source-b"]}), {})
            self.assertTrue(result["state"]["selected_source_ids_explicit"])
            result = component(self.request(), result["state"])
            result = component(self.request({"selected_source_ids": []}), result["state"])
            component(self.request(), result["state"])
            self.assertEqual([call.kwargs["selected_doc_ids"] for call in app.graph.answer_research_question.call_args_list],
                             [[202], [202], [], []])

    def test_old_implicit_scope_state_is_not_promoted_to_targets(self):
        with fake_application() as app:
            component = worker.dialogue_component(7, {"source-a": 101, "source-b": 202}, pipeline="research")
            component(self.request(), {"selected_source_ids": ["source-a", "source-b"]})
            self.assertNotIn("selected_doc_ids", app.graph.answer_research_question.call_args.kwargs)

    def test_out_of_scope_explicit_selection_fails_before_answer(self):
        with fake_application() as app:
            component = worker.dialogue_component(7, {"source-a": 101, "source-b": 202, "other-source": 404}, pipeline="research")
            with self.assertRaisesRegex(ValueError, "outside scenario"):
                component(self.request({"selected_source_ids": ["other-source"]}), {})
            app.graph.answer_research_question.assert_not_called()

    def test_legacy_default_keeps_original_selection_and_state(self):
        with fake_application() as app:
            component = worker.dialogue_component(7, {"source-a": 101, "source-b": 202}, pipeline="legacy")
            result = component(self.request(), {})
            self.assertEqual(app.legacy.call_args.kwargs["selected_doc_ids"], [101, 202])
            self.assertEqual(result["state"]["selected_source_ids"], ["source-a", "source-b"])
            self.assertNotIn("selected_source_ids_explicit", result["state"])
            app.graph.answer_research_question.assert_not_called()


if __name__ == "__main__":
    unittest.main()
