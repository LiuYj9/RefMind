"""Phase 0: metric invariants, runner boundaries, dataset validation and isolation."""
from __future__ import annotations
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.dataset import GoldenItem, GoldenSet, load_golden_set
from evaluation.evaluator import evaluate_items, judge_answers
from evaluation.metrics.retrieval import evaluate_retrieval, is_relevant, ndcg_at_k, mrr, recall_at_k
from evaluation.run_eval import worker_environment, _workspace, main


def dataset(refusal=False):
    items = [GoldenItem("q1", "question", "truth", (("paper.pdf", 1),))]
    if refusal:
        items.append(GoldenItem("q2", "outside", "cannot answer", (), True))
    return GoldenSet(("paper.pdf",), tuple(items), 0, (), "hash", 1)


class MetricTests(unittest.TestCase):
    def test_duplicate_pages_do_not_inflate_ndcg(self):
        self.assertEqual(ndcg_at_k([{"filename": "paper.pdf", "page": 1}]*5, {("paper.pdf", 1)}, 5), 1)

    def test_missing_page_cannot_count_for_any_metric(self):
        docs = [{"filename": "paper.pdf"}]
        ev = {("paper.pdf", 1)}
        self.assertFalse(is_relevant(docs[0], ev))
        self.assertEqual((mrr(docs, ev), recall_at_k(docs, ev, 1), ndcg_at_k(docs, ev, 1)), (0, 0, 0))

    def test_duplicate_does_not_displace_original_rank(self):
        docs = [{"filename": "paper.pdf", "page": p} for p in (1, 1, 2)]
        expected = (1 + 1/math.log2(4))/(1+1/math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(docs, {("paper.pdf", 1), ("paper.pdf", 2)}, 3), expected)

    def test_page_range_and_tolerance(self):
        docs = [{"filename": "PAPER.pdf", "page_start": 2, "page_end": 3}]
        self.assertEqual(recall_at_k(docs, {("paper.pdf", 2), ("paper.pdf", 3)}, 1), 1)
        self.assertEqual(recall_at_k(docs, {("paper.pdf", 4)}, 1), 0)
        self.assertEqual(recall_at_k(docs, {("paper.pdf", 4)}, 1, 1), 1)

    def test_duplicate_page_ranges_cannot_earn_twice(self):
        ev = {("paper.pdf", 1), ("paper.pdf", 2)}
        docs = [{"filename": "paper.pdf", "page_start": 1, "page_end": 2}]*2
        self.assertAlmostEqual(ndcg_at_k(docs, ev, 2), 1/(1+1/math.log2(3)))

    def test_hand_calculated_mrr_and_recall(self):
        docs = [{"filename": "paper.pdf", "page": p} for p in (9, 1, 2)]
        scores = evaluate_retrieval(docs, [("paper.pdf", 1), ("paper.pdf", 2)], (1, 2, 3))
        self.assertEqual(scores.recall, {1: 0, 2: .5, 3: 1})
        self.assertEqual(scores.mrr, .5)

    def test_metric_bounds_with_repetition_and_missing_metadata(self):
        import itertools
        for pages in itertools.product((None, 0, 1, 2, 9), repeat=3):
            docs = [{"filename": "paper.pdf", "page": p} for p in pages]
            scores = evaluate_retrieval(docs, [("paper.pdf", 1), ("paper.pdf", 2)], (1, 3))
            self.assertTrue(all(0 <= v <= 1 for v in [scores.mrr, *scores.recall.values(), *scores.ndcg.values()]))

    def test_invalid_k_and_tolerance_rejected(self):
        for ks, tolerance in [((0,), 0), ((-1,), 0), ((), 0), ((1,), -1)]:
            with self.assertRaises(ValueError):
                evaluate_retrieval([], [], ks, tolerance)

    def test_invalid_numeric_page_never_becomes_a_valid_hit(self):
        for page in (True, -1, 0, 1.9, float('inf'), float('nan')):
            self.assertFalse(is_relevant({'filename': 'paper.pdf', 'page': page}, {('paper.pdf', 1)}))


class RunnerTests(unittest.TestCase):
    def test_retrieval_calls_only_retriever(self):
        retriever = Mock(return_value=[{"page_content": "source", "metadata": {"filename": "paper.pdf", "page": 1}}])
        report = evaluate_items(dataset(), mode="retrieval", component=retriever)
        retriever.assert_called_once_with("question")
        self.assertIsNone(report["items"][0]["answer"])
        self.assertEqual(report["retrieval_stage"], "raw_hybrid_candidates")

    def test_answer_measures_actual_final_context(self):
        report = evaluate_items(dataset(), mode="answer", component=lambda q: {
            "answer": "truth", "is_refusal": False, "documents": []})
        self.assertEqual(report["summary"]["retrieval"]["recall"]["@1"], 0)
        self.assertEqual(report["retrieval_stage"], "final_answer_context")

    def test_failures_are_not_successful_refusals(self):
        report = evaluate_items(dataset(True), mode="answer", component=lambda q: {"answer": "", "service_failed": True})
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["summary"]["num_errors"], 2)
        self.assertEqual(report["summary"]["refusal"]["correct_refusal"]["rate"], 0)

    def test_empty_answer_is_error(self):
        report = evaluate_items(dataset(), mode="answer", component=lambda q: {"answer": ""})
        self.assertEqual(report["items"][0]["status"], "error")
        self.assertIsNone(report["items"][0]["is_refusal"])

    def test_partial_answer_not_misclassified_by_refusal_keywords(self):
        report = evaluate_items(dataset(), mode="answer", component=lambda q: {"answer": "未找到其他条件，但实验温度为77K。"})
        self.assertIsNone(report["items"][0]["is_refusal"])
        self.assertEqual(report["summary"]["refusal"]["false_refusal"]["unclassified"], 1)

    def test_one_failure_does_not_drop_remaining_questions(self):
        report = evaluate_items(dataset(True), mode="retrieval", component=Mock(side_effect=[ValueError("secret"), []]))
        self.assertEqual(len(report["items"]), 2)
        self.assertEqual(report["summary"]["retrieval"]["scored"], 0)
        self.assertNotIn("secret", json.dumps(report))

    def test_judge_reads_saved_answers_without_regeneration(self):
        answer = evaluate_items(dataset(), mode="answer", component=lambda q: {"answer": "truth", "is_refusal": False})
        original = copy.deepcopy(answer)
        judge = Mock(return_value={k: 1 for k in ("faithfulness", "answer_relevance", "context_precision", "context_recall")})
        report = judge_answers(answer, judge=judge)
        judge.assert_called_once()
        self.assertEqual(answer, original)
        self.assertEqual(report["summary"]["generation"]["faithfulness"]["mean"], 1)

    def test_failed_judge_has_explicit_coverage(self):
        answer = evaluate_items(dataset(), mode="answer", component=lambda q: {"answer": "truth", "is_refusal": False})
        report = judge_answers(answer, judge=Mock(return_value={"faithfulness": None}))
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["summary"]["generation"]["faithfulness"]["scored"], 0)
        self.assertIsNone(report["summary"]["generation"]["faithfulness"]["mean"])

    def test_judge_rejects_retrieval_reports(self):
        with self.assertRaises(ValueError):
            judge_answers({"schema_version": 2, "mode": "retrieval"}, judge=Mock())


class DatasetTests(unittest.TestCase):
    def write(self, root, mutate=None):
        data = {"corpus": ["paper.pdf"], "items": [{"id": "q", "question": "question", "ground_truth": "truth",
                "evidence": [{"filename": "paper.pdf", "page": 1}]}]}
        if mutate:
            mutate(data)
        p = root / "golden.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_legacy_dataset_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(len(load_golden_set(self.write(root), project_root=root).items), 1)

    def test_invalid_labels_rejected(self):
        mutations = [lambda d: d["items"].append(d["items"][0]),
                     lambda d: d["items"][0].update(expect_refusal="false"),
                     lambda d: d["items"][0]["evidence"][0].update(page=0),
                     lambda d: d["items"][0]["evidence"][0].update(filename="other.pdf"),
                     lambda d: d["items"][0].update(ground_truth=""),
                     lambda d: d.update(corpus=["../escape.pdf"])]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mutate in mutations:
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    load_golden_set(self.write(root, mutate), project_root=root)

    def test_holdout_requires_explicit_unlock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gs = load_golden_set(self.write(root, lambda d: d["items"][0].update(split="holdout")), project_root=root)
            with self.assertRaises(ValueError):
                gs.select("holdout")
            self.assertEqual(len(gs.select("holdout", True).items), 1)

    def v2(self, root):
        import fitz, hashlib
        pdf = fitz.open()
        pdf.new_page().insert_text((72,72), "Source evidence: critical temperature is 77 K.")
        pdf.save(root/'paper.pdf')
        pdf.close()
        return {"schema_version": 2, "corpus": [{"id": "p1", "path": "paper.pdf", "pages": 1,
                "sha256": hashlib.sha256((root/'paper.pdf').read_bytes()).hexdigest(), "family": "study1", "split": "dev"}],
                "items": [{"id": "q1", "question": "temperature?", "ground_truth": "77 K", "split": "dev",
                "evidence": [{"filename": "paper.pdf", "page": 1, "quote": "critical temperature is 77 K"}]}]}

    def test_real_pdf_hash_page_and_quote_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.v2(root)
            p = root/'golden.json'
            p.write_text(json.dumps(data), encoding='utf-8')
            self.assertEqual(len(load_golden_set(p, project_root=root, verify_files=True).items), 1)
            for field, bad_value in (("quote", "fabricated text not in source"), ("page", 2)):
                bad = copy.deepcopy(data)
                bad["items"][0]["evidence"][0][field] = bad_value
                p.write_text(json.dumps(bad), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_golden_set(p, project_root=root, verify_files=True)
            p.write_text(json.dumps(data), encoding='utf-8')
            (root/'paper.pdf').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                load_golden_set(p, project_root=root, verify_files=True)

    def test_v2_cross_split_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.v2(root)
            data['items'][0]['split'] = 'holdout'
            p = root/'golden.json'
            p.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_golden_set(p, project_root=root)

    def test_duplicate_document_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.v2(root)
            clone = {**data['corpus'][0], 'id': 'p2', 'path': 'another.pdf', 'split': 'holdout', 'family': 'study2'}
            data['corpus'].append(clone)
            p = root/'golden.json'
            p.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_golden_set(p, project_root=root)


class IsolationTests(unittest.TestCase):
    def test_environment_is_copy_and_all_storage_redirected(self):
        before = dict(os.environ)
        root = Path(tempfile.gettempdir()) / "eval-test"
        env = worker_environment(root)
        self.assertEqual(before, dict(os.environ))
        for key in ("DATABASE_PATH", "UPLOAD_DIR", "PARSED_DIR", "DOCSTORE_DIR", "CHROMA_PERSIST_DIR"):
            self.assertTrue(Path(env[key]).is_relative_to(root))
        self.assertEqual(env["IMAGE_SUMMARY_ENABLED"], "false")

    def test_workspace_cannot_target_daily_data(self):
        with self.assertRaises(ValueError):
            _workspace(Path(__file__).resolve().parents[1] / "data", create=False)

    def test_cli_denies_billable_modes_before_calling_backend(self):
        for mode in ("prepare", "retrieval", "answer", "judge"):
            with self.subTest(mode=mode), patch("evaluation.run_eval.run_backend") as backend, patch('sys.stderr'):
                with self.assertRaises(SystemExit) as error:
                    main(["--mode", mode])
                self.assertEqual(error.exception.code, 2)
                backend.assert_not_called()

    def test_worker_timeout_is_reported_as_failure(self):
        from evaluation.run_eval import run_backend
        with tempfile.TemporaryDirectory() as tmp, patch('evaluation.run_eval.subprocess.run', side_effect=subprocess.TimeoutExpired('worker', 1800)):
            report = run_backend({'mode': 'retrieval'}, Path(tmp))
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['error']['type'], 'TimeoutExpired')

    def test_real_subprocess_settings_do_not_change_parent(self):
        from refmind.config import settings
        original = (settings.database_path, settings.retrieval_top_k)
        with tempfile.TemporaryDirectory() as tmp:
            env = worker_environment(Path(tmp))
            code = 'from refmind.config import settings; import json; print(json.dumps({"db":str(settings.database_path)})); settings.retrieval_top_k=999'
            proc = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["db"], str(Path(tmp)/"db/evaluation.db"))
        self.assertEqual(original, (settings.database_path, settings.retrieval_top_k))


if __name__ == "__main__":
    unittest.main()
