"""Independent retrieval/answer/judge runners. No storage, settings or API imports."""
from __future__ import annotations
import math
import time
from statistics import mean
from typing import Callable

from .dataset import GoldenSet, load_golden_set
from .metrics.retrieval import evaluate_retrieval


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
            "retrieval": None, "generation": None, "documents": [], "latency_s": None}


def evaluate_items(golden: GoldenSet, *, mode: str, component: Callable,
                   k_values=(1, 3, 5, 10)) -> dict:
    """Retrieval never calls an answerer; answer measures its actual final context."""
    if mode not in {"retrieval", "answer"}:
        raise ValueError("mode must be retrieval or answer")
    # Validate K even when every question expects refusal.
    evaluate_retrieval([], [], k_values, golden.page_tolerance)
    results = []
    for item in golden.items:
        row = _base(item)
        started = time.perf_counter()
        try:
            output = component(item.question)
            if mode == "retrieval":
                docs = _documents(output)
            else:
                if output.get("service_failed") or output.get("retrieval_failed"):
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
                docs = _documents(output.get("documents", []))
            row["documents"] = docs
            if item.evidence:
                row["retrieval"] = evaluate_retrieval(
                    [d["metadata"] for d in docs], item.evidence,
                    k_values, golden.page_tolerance).as_dict()
        except Exception as exc:
            row["status"], row["error"] = "error", _error(exc)
        row["latency_s"] = round(time.perf_counter()-started, 6)
        results.append(row)
    report = {"schema_version": 2, "mode": mode, "dataset_hash": golden.dataset_hash,
              "retrieval_stage": "raw_hybrid_candidates" if mode == "retrieval" else "final_answer_context",
              "config": {"k_values": list(k_values), "page_tolerance": golden.page_tolerance},
              "items": results}
    return finalize_report(report)


def judge_answers(answer_report: dict, *, judge: Callable) -> dict:
    """Score saved answers only; no retrieval, indexing or answer generation."""
    import copy
    if answer_report.get("schema_version") != 2 or answer_report.get("mode") != "answer":
        raise ValueError("judge input must be a v2 answer report")
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
