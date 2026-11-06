"""Independent external, four-turn dialogue and PDF-object evaluation adapters.

No network/model imports occur here. Callbacks run the actual component; recorded
replay is explicitly labeled and never counted as a fresh live pipeline run.
"""
from __future__ import annotations

import copy
import json
import math
import re
from statistics import mean
import time
import unicodedata

from .evaluator import _safe_trace
from .metrics.fact_evidence import evaluate_fact_evidence, evidence_hit


def canonical_work(value):
    value = str(value or "").strip()
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", "", value, flags=re.I)
    return value.casefold()


def external_ranking(papers, gold, k_values=(1, 3, 5, 10)):
    qrels = {canonical_work(key): grade for key, grade in gold.get("qrels", {}).items()}
    if any(type(grade) is not int or grade not in (0, 1, 2) for grade in qrels.values()):
        raise ValueError("external qrels use explicit grades 0/1/2")
    aliases = {canonical_work(k): canonical_work(v) for k,v in gold.get("aliases", {}).items()}
    target = {canonical_work(key) for key in gold.get("required_work_ids", [])}
    def paper_key(paper):
        values=[paper.get("doi"),paper.get("paper_id")]
        for key,value in (paper.get("external_ids") or {}).items():
            if key.casefold()=="arxiv":
                values.append("arxiv:"+re.sub(r"v\d+$","",str(value),flags=re.I))
            elif key.casefold()=="doi":
                values.append(value)
        keys=[aliases.get(key,key) for value in values if (key:=canonical_work(value))]
        return next((key for key in keys if key in qrels or key in target),keys[0] if keys else "")
    ranked=[key for paper in papers if (key:=paper_key(paper))]
    unique = list(dict.fromkeys(ranked))
    relevant = {key for key, grade in qrels.items() if grade == 2}
    results = {}
    for k in k_values:
        if type(k) is not int or k < 1:
            raise ValueError("K must be positive")
        top = unique[:k]
        unknown = [key for key in top if key not in qrels]
        dcg = sum((2**qrels.get(key, 0)-1)/math.log2(rank+2) for rank,key in enumerate(top))
        ideal = sum((2**grade-1)/math.log2(rank+2) for rank,grade in
                    enumerate(sorted(qrels.values(), reverse=True)[:k]))
        results[f"@{k}"] = {
            "known_recall": len(set(top)&relevant)/len(relevant) if relevant else None,
            "required_work_recall": len(set(top)&target)/len(target) if target else None,
            "precision": None if unknown or not qrels else sum(qrels.get(key)==2 for key in top)/k,
            "ndcg": None if unknown or not ideal else dcg/ideal,
            "judged": len(top)-len(unknown), "returned": len(top), "unjudged": unknown}
    first = next((rank for rank,key in enumerate(unique,1) if key in relevant or key in target), None)
    return {"by_k": results, "known_mrr": 1/first if first else 0.0,
            "duplicate_rate": (len(ranked)-len(unique))/len(ranked) if ranked else 0.0,
            "limitations": "Recall is against known relevant works; unjudged results are not silently treated as irrelevant."}


def _base_report(golden, track, allow_draft, execution):
    if any(item.track != track for item in golden.items):
        raise ValueError(f"{track} runner received a different track")
    return {"schema_version": 3, "track": track, "execution": execution,
            "dataset_hash": golden.dataset_hash, "review_gate": golden.require_review(allow_draft=allow_draft),
            "items": []}


def _summary(report):
    rows = report["items"]
    errors = [row for row in rows if row["status"] == "error"]
    report["status"] = "partial" if errors else "completed"
    report["summary"] = {"items": len(rows), "errors": len(errors), "scored": len(rows)-len(errors),
        "splits": {name: sum(row["question_split"] == name for row in rows)
                   for name in sorted({row["question_split"] for row in rows})}}
    return report


def _row(item):
    return {"id": item.id, "domain": item.domain, "question_split": item.question_split,
            "status": "ok", "error": None}


def evaluate_external(golden, component, *, allow_draft=False, execution="component", k_values=(1,3,5,10)):
    report = _base_report(golden, "external", allow_draft, execution)
    for item in golden.items:
        row, started = _row(item), time.perf_counter()
        try:
            # Gold answers/qrels are never passed into the component.
            output = component({"id": item.id, "query": item.question,
                                "corpus_scope": list(item.corpus_scope)})
            gold = item.task_metadata["external_gold"]
            papers = output.get("papers", [])
            row["ranking"] = external_ranking(papers, gold, k_values)
            row["actual_status"] = output.get("status")
            row["expected_status_match"] = output.get("status") == gold["expected_status"] if gold.get("expected_status") else None
            fetch, parse, answer = (output.get(name) or {} for name in ("fetch", "parse", "answer"))
            identity = fetch.get("identity_status") == "matched"
            expected_sha = gold.get("fulltext_sha256")
            hash_ok = bool(expected_sha and fetch.get("sha256") == expected_sha)
            parsed = parse.get("status") == "ready" and parse.get("source_sha256") == fetch.get("sha256")
            docs = answer.get("documents", [])
            facts = evaluate_fact_evidence(docs, item.evidence_details, item.fact_slots, (max(1,len(docs)),))
            evidence_available = (next(iter(facts.values()))["all_required_covered"] if facts else
                all(any(evidence_hit(doc, ev) for doc in docs) for ev in item.evidence_details) if item.evidence_details else None)
            row["fulltext"] = {"identity_matched": identity, "pinned_bytes_match": hash_ok,
                "parsed_pinned_bytes": parsed and identity and hash_ok,
                "evidence_available": evidence_available,
                "fulltext_stage_complete": bool(identity and hash_ok and parsed
                    and answer.get("evidence_level") == "fulltext" and evidence_available is True),
                "answer_correctness": "not_scored_by_evidence_availability"}
            ing = output.get("ingestion") or {}
            row["ingestion"] = {"status": ing.get("status"),
                "authorized_write": ing.get("approved") is True if ing.get("status") == "ingested" else None,
                "expectation_match": ing.get("status") == gold["ingestion_expected"] if gold.get("ingestion_expected") else None}
            task=gold.get("task_type")
            if task=="lookup":
                row["task_passed"]=row["ranking"]["by_k"][f"@{max(k_values)}"]["required_work_recall"]==1
            elif task=="fulltext_read":
                row["task_passed"]=row["fulltext"]["fulltext_stage_complete"]
            elif task=="abstract_boundary" and gold.get("abstract_support") in {"supported","insufficient"}:
                row["task_passed"]=(answer.get("status")=="fulltext_required") if gold["abstract_support"]=="insufficient" else answer.get("evidence_level")=="abstract"
            else:
                row["task_passed"]=None
            row["task_pass_limitations"]="Stage completion and evidence availability; not a substitute for answer correctness adjudication."
            row["trace"] = _safe_trace(output.get("trace", {}))
        except Exception as exc:
            row.update(status="error", error={"type": type(exc).__name__, "message": "external component failed"})
        row["latency_s"] = time.perf_counter()-started
        report["items"].append(row)
    report=_summary(report)
    scored=[row["task_passed"] for row in report["items"] if row.get("task_passed") is not None]
    report["summary"].update(task_checks_scored=len(scored),task_checks_passed=sum(scored),
                             task_check_rate=mean(scored) if scored else None)
    return report


def _matches(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and _matches(actual[k],v) for k,v in expected.items())
    return actual == expected


def evaluate_dialogues(golden, component, *, allow_draft=False, execution="component"):
    report = _base_report(golden, "dialogue", allow_draft, execution)
    for item in golden.items:
        row, state = _row(item), {}
        row["turns"] = []
        for index, turn in enumerate(item.task_metadata["turns"]):
            observed = {"turn_index": index, "status": "ok", "checks": {}}
            try:
                request = {"scenario_id": item.id, "turn_index": index,
                           "question": turn["question"], "corpus_scope": list(item.corpus_scope),
                           "input":copy.deepcopy(turn.get("input",{}))}
                output = component(request, copy.deepcopy(state))
                state = copy.deepcopy(output.get("state", state))
                expected = turn["expected"]
                if "status" in expected:
                    observed["checks"]["status"] = output.get("status") == expected["status"]
                if "status_in" in expected:
                    observed["checks"]["status"] = output.get("status") in expected["status_in"]
                if "state" in expected:
                    observed["checks"]["state"] = _matches(state, expected["state"])
                docs = output.get("documents", [])
                source_ids = {d.get("metadata", {}).get("source_id") for d in docs}
                if "required_source_ids" in expected:
                    observed["checks"]["required_sources"] = set(expected["required_source_ids"]) <= source_ids
                if "forbidden_source_ids" in expected:
                    observed["checks"]["forbidden_sources"] = not set(expected["forbidden_source_ids"]) & source_ids
                if "evidence_sets" in expected:
                    by_id = {ev["id"]:ev for ev in item.evidence_details}
                    hit = {eid for eid,ev in by_id.items() if any(evidence_hit(d,ev) for d in docs)}
                    observed["checks"]["evidence"] = any(set(group) <= hit for group in expected["evidence_sets"])
                if not observed["checks"]:
                    raise ValueError("turn has no explicit behavioral checks")
                observed["passed"] = all(observed["checks"].values())
                observed["trace"] = _safe_trace(output.get("trace", {}))
            except Exception as exc:
                observed.update(status="error", passed=False, error={"type":type(exc).__name__,"message":"dialogue turn failed"})
            row["turns"].append(observed)
        row["group_passed"] = len(row["turns"]) == 4 and all(t["passed"] for t in row["turns"])
        row["answer_correctness"] = "behavioral_checks_only_unless_source_fact_checks_annotated"
        if any(t["status"] == "error" for t in row["turns"]):
            row["status"] = "error"
        report["items"].append(row)
    report = _summary(report)
    report["summary"]["group_pass_rate"] = mean(r["group_passed"] for r in report["items"]) if report["items"] else None
    report["summary"]["groups"] = len(report["items"])
    report["summary"]["turns"] = sum(len(r["turns"]) for r in report["items"])
    return report


def _bbox(obj, page_size):
    bbox = obj.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in bbox):
        return None
    space = obj.get("bbox_coordinate_system")
    if space == "normalized_1000":
        box = [v/1000 for v in bbox]
    elif space == "pdf_points" and page_size and min(page_size) > 0:
        box = [bbox[0]/page_size[0],bbox[1]/page_size[1],bbox[2]/page_size[0],bbox[3]/page_size[1]]
    else:
        return None
    return box if 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1 else None


def _iou(a,b):
    if a is None or b is None:
        return 0.0
    overlap = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union else 0.0


def _kind(obj):
    kind = obj.get("type", "unknown")
    return {"paragraph":"text", "formula":"equation", "image":"figure"}.get(kind,kind)


def structure_scores(predicted, gold, *, iou_threshold=.5):
    objects, page_size = gold["objects"], gold.get("page_size")
    edges = {g: sorted([p for p,pred in enumerate(predicted) if _kind(pred) == _kind(obj)
                           and _iou(_bbox(pred,page_size),_bbox(obj,page_size)) >= iou_threshold],
                       key=lambda p:-_iou(_bbox(predicted[p],page_size),_bbox(obj,page_size)))
             for g,obj in enumerate(objects)}
    matched = {}
    def augment(g, seen):
        for p in edges[g]:
            if p in seen:
                continue
            seen.add(p)
            if p not in matched or augment(matched[p],seen):
                matched[p]=g
                return True
        return False
    for g in edges:
        augment(g,set())
    reverse = {g:p for p,g in matched.items()}
    by_type = {}
    partial = gold.get("annotation_coverage") == "partial_regions"
    for kind in sorted({_kind(o) for o in objects+predicted}):
        ng, np = sum(_kind(o)==kind for o in objects),sum(_kind(o)==kind for o in predicted)
        hit = sum(_kind(objects[g])==kind for g in reverse)
        precision,recall = (None if partial else hit/np if np else 0.0),hit/ng if ng else None
        by_type[kind] = {"gold":ng,"predicted":np,"matched":hit,"precision":precision,"recall":recall,
                         "f1":None if partial else 2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0.0}
    order = gold.get("reading_order", [])
    positions = {objects[g]["id"]:predicted[p].get("reading_order",p) for g,p in reverse.items()}
    pairs = [(a,b) for index,a in enumerate(order) for b in order[index+1:]]
    correct = sum(a in positions and b in positions and positions[a] < positions[b] for a,b in pairs)
    equations = [g for g,obj in enumerate(objects) if _kind(obj)=="equation" and obj.get("text")]
    norm=lambda value: re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))
    exact = sum(g in reverse and norm(objects[g]["text"]) == norm(predicted[reverse[g]].get("text")) for g in equations)
    return {"iou_threshold":iou_threshold,"annotation_coverage":gold.get("annotation_coverage","assumed_exhaustive_fixture"),"by_type":by_type,
            "reading_order":{"gold_pairs":len(pairs),"correct_pairs":correct,"accuracy":correct/len(pairs) if pairs else None},
            "equation_literal_match":{"eligible":len(equations),"correct":exact,"rate":exact/len(equations) if equations else None},
            "limitations":"Known-object recall/order and literal formula matching, not semantic equation equivalence or table-cell accuracy. Partial region annotations cannot score precision/F1."}


def evaluate_structure(golden, component, *, allow_draft=False, execution="component"):
    report = _base_report(golden,"structure",allow_draft,execution)
    for item in golden.items:
        row = _row(item)
        try:
            gold = item.task_metadata["structure_gold"]
            output = component({"id":item.id,"source_id":gold["source_id"],"page":gold["page"]})
            row["objects"] = structure_scores(output["objects"],gold)
            row["trace"] = _safe_trace(output.get("trace",{}))
        except Exception as exc:
            row.update(status="error",error={"type":type(exc).__name__,"message":"structure component failed"})
        report["items"].append(row)
    return _summary(report)
