"""Independent retrieval/answer/judge runners. No storage, settings or API imports."""
from __future__ import annotations
import math
import time
from dataclasses import fields, is_dataclass
from statistics import mean
from typing import Callable

from .dataset import GoldenSet, load_golden_set
from .metrics.retrieval import evaluate_retrieval
from .metrics.fact_evidence import evaluate_fact_evidence

COMPONENT_REVIEW_FIELDS = ("evidence_review", "warnings", "rejection_stage", "task_description", "review_status")


def _documents(documents):
    return [{"page_content": d.page_content, "metadata": dict(d.metadata or {})}
            if hasattr(d, "page_content") else dict(d) for d in documents]


def _error(exc):
    # Do not put provider exception messages (possibly containing credentials) in reports.
    return {"type": type(exc).__name__, "message": "component failed; inspect local diagnostic log"}


def _base(item):
    return {"id": item.id, "question": item.question, "ground_truth": item.ground_truth,
            "category": item.category, "split": item.split, "expect_refusal": item.expect_refusal,
            "status": "ok", "error": None, "answer": None, "is_refusal": None,
            "retrieval": None, "generation": None, "documents": [], "latency_s": None,
            "track": item.track, "domain": item.domain, "question_split": item.question_split,
            "family_ids": list(item.family_ids), "corpus_scope": list(item.corpus_scope),
            "annotation_status": item.annotation_status, "fact_evidence": None,
            "trace": {"available": False, "stages": {}}, "operational_status": None,
            "component_review_fields_present": []}


def _safe_trace(value):
    """Remove credential fields; preserve supplied stage data without inventing missing stages."""
    if hasattr(value, "page_content") and hasattr(value, "metadata"):
        return {"page_content": value.page_content, "metadata": _safe_trace(value.metadata)}
    if is_dataclass(value) and not isinstance(value, type):
        return _safe_trace({field.name: getattr(value, field.name) for field in fields(value)})
    if isinstance(value, dict):
        return {str(k): _safe_trace(v) for k, v in value.items()
                if not any(word in str(k).casefold() for word in
                           ("api_key", "authorization", "password", "secret", "access_token"))}
    if isinstance(value, (list, tuple)):
        return [_safe_trace(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(type(value).__name__)


def _stage_metrics(trace, item, k_values, tolerance):
    stages = trace.get("stages", {}) if isinstance(trace, dict) else {}
    if not isinstance(stages, dict):
        return {}
    out = {}
    for name, stage in stages.items():
        if not isinstance(stage, dict) or not isinstance(stage.get("documents"), list):
            continue
        docs = _documents(stage["documents"])
        out[name] = {"documents": len(docs), "retrieval": evaluate_retrieval(
            [d["metadata"] for d in docs], item.evidence, k_values, tolerance).as_dict() if item.evidence else None,
            "fact_evidence": evaluate_fact_evidence(docs, item.evidence_details, item.fact_slots, k_values)}
    return out


def evaluate_items(golden: GoldenSet, *, mode: str, component: Callable,
                   k_values=(1, 3, 5, 10), allow_draft=False, component_accepts_item=False) -> dict:
    """Retrieval never calls an answerer; answer measures its actual final context."""
    if mode not in {"retrieval", "answer"}:
        raise ValueError("mode must be retrieval or answer")
    # Validate K even when every question expects refusal.
    evaluate_retrieval([], [], k_values, golden.page_tolerance)
    if any(item.track != "local" for item in golden.items):
        raise ValueError("local RAG runner only accepts local tasks; external/dialogue/structure use dedicated adapters")
    review_gate = golden.require_review(allow_draft=allow_draft) if golden.schema_version >= 3 else {
        "official": False, "label": "legacy_diagnostic", "approved": 0, "draft": len(golden.items)}
    results = []
    for item in golden.items:
        row = _base(item)
        started = time.perf_counter()
        try:
            output = component(item if component_accepts_item else item.question)
            if isinstance(output, dict):
                # Whitelist result metadata: a component cannot overwrite the question or gold labels.
                for name in COMPONENT_REVIEW_FIELDS:
                    if name in output:
                        row[name] = _safe_trace(output[name])
                        row["component_review_fields_present"].append(name)
                supplied = output.get("trace")
                if supplied is not None:
                    row["trace"] = {"available": True, **_safe_trace(supplied)} if isinstance(supplied, dict) else {
                        "available": True, "stages": _safe_trace(supplied)}
                row["operational_status"] = output.get("status")
                row["stage_evidence"] = _stage_metrics(row["trace"], item, k_values, golden.page_tolerance)
            if mode == "retrieval":
                docs = _documents(output.get("documents", []) if isinstance(output, dict) else output)
            else:
                if (output.get("service_failed") or output.get("retrieval_failed")
                        or output.get("status") in {"error", "service_error", "retrieval_error", "generation_error", "budget_exceeded", "timeout"}):
                    raise RuntimeError("answer service failed")
                answer = output.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("empty answer is a failure, not a refusal")
                row["answer"] = answer
                refusal = output.get("is_refusal")
                if type(refusal) is not bool:
                    refusal = True if output.get("rejection_stage") else None
                row["is_refusal"] = refusal
                row["classification"] = "structured" if refusal is not None else "unclassified"
                returned_docs = _documents(output.get("documents", []))
                row["returned_evidence"] = returned_docs
                stages = row["trace"].get("stages", {})
                actual_context = stages.get("generation_context") if isinstance(stages, dict) else None
                if isinstance(actual_context, dict) and isinstance(actual_context.get("documents"), list):
                    docs = _documents(actual_context["documents"])
                    row["context_source"] = "trace.stages.generation_context"
                else:
                    docs = returned_docs
                    row["context_source"] = "component_documents_legacy_unverified"
                    if row["trace"].get("pipeline") == "research":
                        # Returned citations and pre-review packing cannot reconstruct what the generator saw.
                        docs = []
                        row["context_source"] = "missing_generation_context"
            row["documents"] = docs
            context_missing = row.get("context_source") == "missing_generation_context"
            row["fact_evidence"] = None if context_missing else evaluate_fact_evidence(docs, item.evidence_details, item.fact_slots, k_values)
            if item.evidence and not context_missing:
                row["retrieval"] = evaluate_retrieval(
                    [d["metadata"] for d in docs], item.evidence,
                    k_values, golden.page_tolerance).as_dict()
        except Exception as exc:
            row["status"], row["error"] = "error", _error(exc)
        row["latency_s"] = round(time.perf_counter()-started, 6)
        results.append(row)
    report = {"schema_version": 3 if golden.schema_version >= 3 else 2, "mode": mode, "dataset_hash": golden.dataset_hash,
              "review_gate": review_gate,
              "retrieval_stage": "raw_hybrid_candidates" if mode == "retrieval" else "final_answer_context",
              "config": {"k_values": list(k_values), "page_tolerance": golden.page_tolerance},
              "items": results}
    return finalize_report(report)


def judge_answers(answer_report: dict, *, judge: Callable) -> dict:
    """Score saved answers only; no retrieval, indexing or answer generation."""
    import copy
    if answer_report.get("schema_version") not in (2, 3) or answer_report.get("mode") != "answer":
        raise ValueError("judge input must be a saved answer report")
    report = copy.deepcopy(answer_report)
    report["mode"] = "judge"
    report["source_answer_run_id"] = answer_report.get("run_id")
    for row in report["items"]:
        if row["status"] != "ok" or row["expect_refusal"] or row["is_refusal"] is True:
            continue
        try:
            scores = judge(question=row["question"], answer=row["answer"],
                           contexts=[d["page_content"] for d in row["documents"]],
                           ground_truth=row["ground_truth"])
            values = scores.as_dict() if hasattr(scores, "as_dict") else dict(scores)
            keys = ("faithfulness", "answer_relevance", "context_precision", "context_recall")
            if any(type(values.get(k)) not in (int, float) or not math.isfinite(values[k])
                   or not 0 <= values[k] <= 1 for k in keys):
                raise ValueError("missing or invalid judge metrics")
            row["generation"] = {k: values[k] for k in keys}
        except Exception as exc:
            row["judge_error"] = _error(exc)
    return finalize_report(report)


def finalize_report(report: dict) -> dict:
    rows = report["items"]
    ok = [r for r in rows if r["status"] == "ok"]
    errors = [r for r in rows if r["status"] != "ok" or r.get("judge_error")]
    report["status"] = "partial" if errors else "completed"
    summary = {"num_items": len(rows), "num_ok": len(ok), "num_errors": len(errors),
               "num_answerable": sum(not r["expect_refusal"] for r in rows),
               "num_refusal": sum(r["expect_refusal"] for r in rows)}
    eligible = [r for r in rows if not r["expect_refusal"]]
    scored = [r for r in eligible if r["retrieval"] is not None]
    retrieval = {"eligible": len(eligible), "scored": len(scored), "recall": {}, "ndcg": {}, "mrr": None}
    for metric in ("recall", "ndcg"):
        for k in report["config"]["k_values"]:
            values = [r["retrieval"][metric][f"@{k}"] for r in scored]
            retrieval[metric][f"@{k}"] = round(mean(values), 4) if values else None
    if scored:
        retrieval["mrr"] = round(mean(r["retrieval"]["mrr"] for r in scored), 4)
    summary["retrieval"] = retrieval
    fact_rows = [r for r in eligible if r.get("fact_evidence") is not None]
    summary["fact_evidence"] = {"eligible": len(eligible), "annotated": len(fact_rows), "by_k": {}}
    for k in report["config"]["k_values"]:
        values = [r["fact_evidence"][f"@{k}"] for r in fact_rows]
        summary["fact_evidence"]["by_k"][f"@{k}"] = {
            "slot_coverage_mean": mean(v["slot_coverage"] for v in values) if values else None,
            "all_required_rate": mean(v["all_required_covered"] for v in values) if values else None}
    summary["slices"] = {}
    for key in ("domain", "question_split", "category"):
        summary["slices"][key] = {str(value): {"items": sum(r.get(key) == value for r in rows),
            "errors": sum(r.get(key) == value and r["status"] != "ok" for r in rows)}
            for value in sorted({r.get(key, "legacy") for r in rows}, key=str)}
    summary["refusal"] = {}
    if report["mode"] in {"answer", "judge"}:
        for label, expected in (("correct_refusal", True), ("false_refusal", False)):
            group = [r for r in rows if r["expect_refusal"] == expected]
            numerator = sum(r["status"] == "ok" and r["is_refusal"] is True for r in group)
            unclassified = sum(r["status"] == "ok" and r["is_refusal"] is None for r in group)
            summary["refusal"][label] = {"numerator": numerator, "denominator": len(group),
                "unclassified": unclassified, "rate": numerator/len(group) if group else None}
    summary["generation"] = {}
    for key in ("faithfulness", "answer_relevance", "context_precision", "context_recall"):
        values = [r["generation"][key] for r in rows if r.get("generation")]
        summary["generation"][key] = {"scored": len(values), "eligible": len(eligible),
                                      "mean": round(mean(values), 4) if values else None}
    times = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)
    summary["latency"] = {"samples": len(times), "mean_s": mean(times) if times else None}
    for name, fraction in (("p50_s", .5), ("p95_s", .95), ("p99_s", .99)):
        summary["latency"][name] = times[max(0, math.ceil(fraction*len(times))-1)] if times else None
    report["summary"] = summary
    return report
