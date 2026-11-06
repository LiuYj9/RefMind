"""V3 split isolation, human-review gates, AND/OR facts and truthful construction."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from evaluation.construction import CORE_CATEGORIES, inventory, manifest, migrate_legacy
from evaluation.dataset import load_golden_set
from evaluation.evaluator import evaluate_items
from evaluation.metrics.fact_evidence import evaluate_fact_evidence
from evaluation.fingerprints import REPORT_SOURCE_FILES, implementation_hashes


def fixture(root):
    import fitz
    sources = []
    for index, split in enumerate(("dev", "id_holdout", "ood_holdout"), 1):
        path = root / f"p{index}.pdf"
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), f"Study {index}: temperature is 77 K and current is 10 mA.")
        pdf.save(path); pdf.close()
        sources.append({"id": f"p{index}", "path": path.name, "pages": 1,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "family": f"family{index}", "question_split": split, "domain": "physics"})
    item = {"id": "q1", "question": "What temperature was measured?", "ground_truth": "77 K",
            "question_split": "dev", "track": "local", "domain": "physics", "family_ids": ["family1"],
            "corpus_scope": ["p1"], "category": "numerical", "annotation_status": "draft",
            "human_review": {"status": "pending"}, "expect_refusal": False,
            "evidence": [{"id": "e1", "source_id": "p1", "page": 1, "quote": "temperature is 77 K"}],
            "fact_slots": [{"id": "temperature", "required": True, "evidence_sets": [["e1"]]}]}
    return {"schema_version": 3, "corpus": sources, "items": [item]}


def write(root, data):
    path = root / "golden.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def approve(item):
    item["annotation_status"] = "human_approved"
    item["human_review"] = {"status": "approved", "reviewer": "test reviewer", "reviewed_at": "2026-11-05"}


class V3ContractTests(unittest.TestCase):
    def test_source_hash_and_quote_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = fixture(root)
            golden = load_golden_set(write(root, data), project_root=root, verify_files=True)
            self.assertEqual(golden.schema_version, 3)
            self.assertEqual(golden.items[0].corpus_scope, ("p1",))
            data["items"][0]["evidence"][0]["quote"] = "fabricated unavailable source quote"
            with self.assertRaises(ValueError):
                load_golden_set(write(root, data), project_root=root, verify_files=True)

    def test_development_never_loads_holdout_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = fixture(root)
            data["items"][0]["corpus_scope"].append("p2")
            with self.assertRaisesRegex(ValueError, "development corpus_scope"):
                load_golden_set(write(root, data), project_root=root)

    def test_acceptance_scope_is_independent_of_positive_evidence_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = fixture(root); item = data["items"][0]
            item.update(question_split="id_holdout", family_ids=["family2"], corpus_scope=["p1", "p2", "p3"])
            item["evidence"][0]["source_id"] = "p2"
            golden = load_golden_set(write(root, data), project_root=root)
            with self.assertRaises(ValueError):
                golden.select("id_holdout")
            selected = golden.select("holdout", True)
            self.assertEqual(len(selected.scoped_manifest()), 3)
            item["evidence"][0]["source_id"] = "p1"
            with self.assertRaisesRegex(ValueError, "positive evidence"):
                load_golden_set(write(root, data), project_root=root)

    def test_family_cannot_cross_source_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = fixture(root)
            data["corpus"][1]["family"] = "family1"
            with self.assertRaisesRegex(ValueError, "family leaked"):
                load_golden_set(write(root, data), project_root=root)

    def test_fact_slot_cannot_reference_invented_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = fixture(root)
            data["items"][0]["fact_slots"][0]["evidence_sets"] = [["invented"]]
            with self.assertRaisesRegex(ValueError, "unknown evidence"):
                load_golden_set(write(root, data), project_root=root)

    def test_draft_runner_requires_explicit_opt_in_before_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); golden = load_golden_set(write(root, fixture(root)), project_root=root)
            component = Mock(return_value=[])
            with self.assertRaisesRegex(ValueError, "human review required"):
                evaluate_items(golden, mode="retrieval", component=component)
            component.assert_not_called()
            report = evaluate_items(golden, mode="retrieval", component=component, allow_draft=True)
            self.assertFalse(report["review_gate"]["official"])
            self.assertEqual(report["review_gate"]["label"], "draft_diagnostic")

    def test_draft_flag_never_bypasses_holdout_human_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = fixture(root); item = data["items"][0]
            item.update(question_split="ood_holdout", family_ids=["family3"], corpus_scope=["p3"])
            item["evidence"][0]["source_id"] = "p3"
            golden = load_golden_set(write(root, data), project_root=root)
            with self.assertRaisesRegex(ValueError, "human review required"):
                golden.select("ood_holdout", True).require_review(allow_draft=True)

    def test_human_approval_cannot_restore_exposed_holdout_independence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=fixture(root);item=data["items"][0]
            item.update(question_split="ood_holdout",family_ids=["family3"],corpus_scope=["p3"])
            item["evidence"][0]["source_id"]="p3";approve(item)
            data["independence_audit"]={"events":[{"affected_question_split":"ood_holdout","affected_family_ids":["family3"],
                "eligible_for_blind_acceptance":False,"reason":"synthetic exposure fixture"}]}
            golden=load_golden_set(write(root,data),project_root=root).select("ood_holdout",True)
            gate=golden.require_review()
            self.assertEqual(gate["label"],"historical_regression")
            self.assertFalse(gate["official"])
            self.assertTrue(gate["human_review_complete"])
            self.assertFalse(gate["independent_eligible"])
            with self.assertRaisesRegex(ValueError,"cannot be approved into independence"):
                golden.require_review(independent_acceptance=True)

    def test_independent_mode_accepts_clean_holdout_with_dev_distractors_and_rejects_dev(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=fixture(root);item=data["items"][0];approve(item)
            dev=load_golden_set(write(root,data),project_root=root)
            with self.assertRaisesRegex(ValueError,"independent acceptance"):
                dev.require_review(independent_acceptance=True)
            item.update(question_split="ood_holdout",family_ids=["family3"],corpus_scope=["p3","p1"])
            item["evidence"][0]["source_id"]="p3"
            golden=load_golden_set(write(root,data),project_root=root).select("ood_holdout",True)
            self.assertTrue(golden.require_review(independent_acceptance=True)["independent_eligible"])

    def test_approval_needs_named_dated_review_and_fact_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = fixture(root); item = data["items"][0]
            item["annotation_status"] = "human_approved"
            with self.assertRaises(ValueError):
                load_golden_set(write(root, data), project_root=root)
            approve(item); item["fact_slots"] = []
            golden = load_golden_set(write(root, data), project_root=root)
            with self.assertRaisesRegex(ValueError, "human review required"):
                golden.require_review()

    def test_trace_is_explicit_and_credential_fields_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = fixture(root); approve(data["items"][0])
            golden = load_golden_set(write(root, data), project_root=root)
            result = {"answer": "77 K", "is_refusal": False, "documents": [],
                      "trace": {"stages": {"retrieval": {"count": 0, "api_key": "secret-value"}}, "budget": {"input_tokens": 12}}}
            report = evaluate_items(golden, mode="answer", component=lambda _: result)
            self.assertTrue(report["review_gate"]["official"])
            self.assertTrue(report["items"][0]["trace"]["available"])
            self.assertNotIn("secret-value", json.dumps(report))
            self.assertEqual(report["items"][0]["trace"]["budget"]["input_tokens"], 12)

    def test_component_review_fields_are_json_safe_without_replacing_gold(self):
        @dataclass
        class TaskFixture:
            resolved_query: str
            conditions: tuple
            authorization: str
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=fixture(root);approve(data["items"][0])
            golden=load_golden_set(write(root,data),project_root=root)
            output={"answer":"fixture refusal","status":"insufficient","is_refusal":True,"documents":[],
                    "question":"component cannot replace question","ground_truth":"component cannot replace gold","category":"invented",
                    "evidence_review":{"coverage":"insufficient","reason":"fixture mismatch","api_key":"fixture secret"},
                    "warnings":("fixture warning",{"nonfinite":float('nan')}),"rejection_stage":"evidence_review",
                    "task_description":TaskFixture("fixture resolved query",("77 K",),"fixture secret"),"review_status":"model_reviewed"}
            report=evaluate_items(golden,mode="answer",component=lambda _:output)
            row=report["items"][0]
            self.assertEqual(row["question"],data["items"][0]["question"])
            self.assertEqual(row["ground_truth"],"77 K")
            self.assertEqual(row["category"],"numerical")
            self.assertEqual(row["evidence_review"],{"coverage":"insufficient","reason":"fixture mismatch"})
            self.assertEqual(row["task_description"],{"resolved_query":"fixture resolved query","conditions":["77 K"]})
            self.assertEqual(row["warnings"],["fixture warning",{"nonfinite":None}])
            self.assertEqual(row["rejection_stage"],"evidence_review")
            self.assertEqual(row["review_status"],"model_reviewed")
            self.assertEqual(len(row["component_review_fields_present"]),5)
            self.assertNotIn("fixture secret",json.dumps(report,allow_nan=False))
            self.assertEqual(output["evidence_review"]["api_key"],"fixture secret")

    def test_service_error_retains_review_diagnostics_and_explicit_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=fixture(root);approve(data["items"][0])
            golden=load_golden_set(write(root,data),project_root=root)
            output={"status":"generation_error","service_failed":True,"warnings":("fixture service warning",),
                    "rejection_stage":None,"evidence_review":{"coverage":"sufficient"}}
            row=evaluate_items(golden,mode="answer",component=lambda _:output)["items"][0]
            self.assertEqual(row["status"],"error")
            self.assertEqual(row["warnings"],["fixture service warning"])
            self.assertIn("rejection_stage",row["component_review_fields_present"])
            self.assertIsNone(row["rejection_stage"])
            self.assertNotIn("task_description",row)

    def test_stage_metrics_identify_evidence_lost_during_packing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = fixture(root); approve(data["items"][0])
            golden = load_golden_set(write(root, data), project_root=root)
            doc = {"page_content": "temperature is 77 K", "metadata": {"filename": "p1.pdf", "page": 1}}
            result = {"documents": [], "trace": {"stages": {"raw": {"documents": [doc]}, "packed": {"documents": []}}}}
            report = evaluate_items(golden, mode="retrieval", component=lambda _: result, k_values=(1,))
            stages = report["items"][0]["stage_evidence"]
            self.assertTrue(stages["raw"]["fact_evidence"]["@1"]["all_required_covered"])
            self.assertFalse(stages["packed"]["fact_evidence"]["@1"]["all_required_covered"])

    def test_local_runner_rejects_external_without_fake_page_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = fixture(root); item = data["items"][0]
            item.update(track="external", evidence=[], fact_slots=[], external_gold={"work_ids": ["doi:example"]})
            golden = load_golden_set(write(root, data), project_root=root)
            with self.assertRaisesRegex(ValueError, "dedicated adapters"):
                evaluate_items(golden, mode="retrieval", component=Mock(), allow_draft=True)

    def test_answer_context_uses_generation_input_after_review_not_packed_or_citations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = fixture(root); approve(data["items"][0])
            golden = load_golden_set(write(root, data), project_root=root)
            doc = {"page_content": "temperature is 77 K", "metadata": {"filename": "p1.pdf", "page": 1}}
            result = {"answer": "77 K", "documents": [doc], "is_refusal": False,
                      "trace": {"pipeline": "research", "stages": {"packed": {"documents": [doc]},
                                  "generation_context": {"documents": []}, "final_evidence": {"documents": [doc]}}}}
            report = evaluate_items(golden, mode="answer", component=lambda _: result, k_values=(1,))
            row=report["items"][0]
            self.assertEqual(row["documents"], [])
            self.assertEqual(len(row["returned_evidence"]), 1)
            self.assertEqual(row["context_source"], "trace.stages.generation_context")
            self.assertFalse(row["fact_evidence"]["@1"]["all_required_covered"])
            self.assertTrue(row["stage_evidence"]["packed"]["fact_evidence"]["@1"]["all_required_covered"])


class FactSlotTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [
            {"id": "temp_a", "filename": "a.pdf", "page": 1, "quote": "Temperature is 77 K"},
            {"id": "temp_b", "filename": "a.pdf", "page": 2, "quote": "At 77 K temperature"},
            {"id": "value", "filename": "a.pdf", "page": 3, "quote": "Current is 10 mA"},
            {"id": "condition", "filename": "a.pdf", "page": 4, "quote": "Under zero applied field"}]
        self.slots = [
            {"id": "temperature", "evidence_sets": [["temp_a"], ["temp_b"]]},
            {"id": "current_with_condition", "evidence_sets": [["value", "condition"]]}]

    def doc(self, page, text, filename="a.pdf"):
        return {"page_content": text, "metadata": {"filename": filename, "page": page}}

    def test_or_alternative_and_required_slots(self):
        docs = [self.doc(2, "At 77 K temperature"), self.doc(3, "Current is 10 mA"), self.doc(4, "Under zero applied field")]
        result = evaluate_fact_evidence(docs, self.evidence, self.slots, (1, 2, 3))
        self.assertEqual(result["@1"]["slot_coverage"], .5)
        self.assertFalse(result["@2"]["all_required_covered"])
        self.assertTrue(result["@3"]["all_required_covered"])

    def test_right_page_wrong_excerpt_never_proves_fact(self):
        docs = [self.doc(1, "A different experiment at 300 K")]
        result = evaluate_fact_evidence(docs, self.evidence, self.slots, (1,))
        self.assertEqual(result["@1"]["covered_slots"], 0)

    def test_wrong_paper_and_si_prefix_case_never_match(self):
        docs = [self.doc(1, "Temperature is 77 K", "b.pdf"), self.doc(3, "Current is 10 MA")]
        result = evaluate_fact_evidence(docs, self.evidence, self.slots, (2,))
        self.assertEqual(result["@2"]["covered_slots"], 0)

    def test_missing_annotations_are_null_not_perfect(self):
        self.assertIsNone(evaluate_fact_evidence([], [], []))


class ConstructionTests(unittest.TestCase):
    def test_report_fingerprints_cover_scientific_dependencies_without_index_changes(self):
        required={"refmind/rag/evidence_bundle.py","refmind/rag/reranker.py","refmind/agents/orchestration.py",
                  "refmind/citations.py","refmind/rag/provenance.py","refmind/contracts.py",
                  "refmind/llm/factory.py","refmind/llm/openai_metering.py"}
        self.assertTrue(required<=set(REPORT_SOURCE_FILES))
        self.assertFalse(any('.env' in name for name in REPORT_SOURCE_FILES))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in REPORT_SOURCE_FILES:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture source bytes')
            hashes=implementation_hashes(root)
            self.assertEqual(set(hashes),set(REPORT_SOURCE_FILES))
            self.assertTrue(all(value==hashlib.sha256(b'fixture source bytes').hexdigest() for value in hashes.values()))

    def test_final_user_quotas_are_exact_and_not_authored_counts(self):
        self.assertEqual(sum(CORE_CATEGORIES.values()), 600)
        self.assertEqual(CORE_CATEGORIES["method"], 70)
        self.assertEqual(CORE_CATEGORIES["comparison"], 70)
        self.assertEqual(CORE_CATEGORIES["summary"], 50)
        config = manifest()
        self.assertEqual(sum(d["families"] for d in config["core_domains"].values()), 60)
        with tempfile.TemporaryDirectory() as tmp:
            status = inventory(config, Path(tmp))
        self.assertEqual(status["tracks"]["local"]["authored"], 0)
        self.assertEqual(status["tracks"]["local"]["missing_authored"], 600)
        self.assertEqual(status["tracks"]["structure"]["missing_authored"], 120)

    def test_actual_legacy_100_migrate_without_overwrite_or_fake_review(self):
        project = Path(__file__).resolve().parents[1]
        source = project / "evaluation/datasets/real_v1.json"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "v3.json"
            info = migrate_legacy(source, output)
            golden = load_golden_set(output, project_root=project)
            self.assertEqual(info["splits"], {"dev": 70, "id_holdout": 30})
            self.assertEqual(len(golden.items), 100)
            self.assertTrue(all(i.review["status"] == "pending" and not i.fact_slots for i in golden.items))
            self.assertEqual(sum(i.category.startswith("legacy_") for i in golden.items), 50)
            output.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                migrate_legacy(source, output)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
