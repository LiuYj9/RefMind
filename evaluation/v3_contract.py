"""V3 evaluation contracts. Validation is offline and never calls application code."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from .dataset import GoldenItem, GoldenSet, canonical_split, normalize_text, verify_corpus

SPLITS = {"dev", "id_holdout", "ood_holdout"}
TRACKS = {"local", "external", "dialogue", "structure"}


def _strings(value, name, *, nonempty=True):
    if (not isinstance(value, list) or (nonempty and not value)
            or any(not isinstance(v, str) or not v.strip() for v in value)
            or len(value) != len(set(value))):
        raise ValueError(f"{name} must contain unique nonempty strings")
    return tuple(value)


def _split(raw):
    value = raw.get("question_split")
    if value not in SPLITS:
        raise ValueError("v3 question_split must be dev, id_holdout or ood_holdout")
    if "split" in raw and canonical_split(raw["split"]) != value:
        raise ValueError("conflicting split and question_split")
    return value


def _review(raw):
    review = raw.get("human_review", {"status": "pending"})
    if not isinstance(review, dict) or review.get("status") not in {"pending", "approved", "rejected"}:
        raise ValueError("invalid human_review")
    status = raw.get("annotation_status", "draft")
    if status == "human_approved" and (review.get("status") != "approved"
            or not review.get("reviewer") or not review.get("reviewed_at")):
        raise ValueError("human_approved requires a named, dated approval")
    if review.get("status") == "approved" and status != "human_approved":
        raise ValueError("approval status and annotation_status disagree")
    return status, review


def _evidence(raw, sources, split):
    if not isinstance(raw, list):
        raise ValueError("evidence must be a list")
    result, ids = [], set()
    for ev in raw:
        if not isinstance(ev, dict) or not ev.get("id") or ev["id"] in ids:
            raise ValueError("evidence IDs must be unique")
        source = sources.get(ev.get("source_id"))
        if source is None or source["question_split"] != split:
            raise ValueError("positive evidence must belong to the question partition")
        if ev.get("filename", Path(source["path"]).name) != Path(source["path"]).name:
            raise ValueError("evidence filename disagrees with source_id")
        page = ev.get("page")
        if type(page) is not int or not 1 <= page <= source["pages"]:
            raise ValueError("invalid evidence page")
        kind = ev.get("evidence_source", "pdf_text")
        if kind == "pdf_text":
            if not isinstance(ev.get("quote"), str) or len(normalize_text(ev["quote"])) < 8:
                raise ValueError("text evidence needs a source quote")
        elif kind == "visual":
            bbox = ev.get("bbox")
            if (not isinstance(bbox, list) or len(bbox) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)
                    or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]
                    or ev.get("bbox_coordinate_system") not in {"pdf_points", "normalized_1000"}
                    or not re.fullmatch(r"[0-9a-f]{64}", ev.get("render_sha256", ""))
                    or not ev.get("observation")):
                raise ValueError("visual evidence needs a pinned region, coordinate system and observation")
        else:
            raise ValueError("invalid evidence_source")
        ids.add(ev["id"])
        result.append({**ev, "filename": Path(source["path"]).name})
    return tuple(result)


def _slots(raw, evidence):
    if not isinstance(raw, list):
        raise ValueError("fact_slots must be a list")
    result, ids, evidence_ids = [], set(), {ev["id"] for ev in evidence}
    for slot in raw:
        if not isinstance(slot, dict) or not slot.get("id") or slot["id"] in ids:
            raise ValueError("fact slot IDs must be unique")
        if type(slot.get("required", True)) is not bool:
            raise ValueError("slot required must be boolean")
        alternatives = slot.get("evidence_sets")
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError("each fact slot needs OR alternatives of AND evidence IDs")
        for group in alternatives:
            refs = _strings(group, "evidence set")
            if not set(refs) <= evidence_ids:
                raise ValueError("fact slot references unknown evidence")
        ids.add(slot["id"])
        result.append(dict(slot))
    return tuple(result)


def load_v3(data, raw_bytes, root, *, verify_files=False):
    audit=data.get("independence_audit",{})
    if not isinstance(audit,dict) or not isinstance(audit.get("events",[]),list):
        raise ValueError("independence_audit must contain explicit event metadata")
    for event in audit.get("events",[]):
        if (not isinstance(event,dict) or event.get("affected_question_split") not in {"id_holdout","ood_holdout"}
            or event.get("eligible_for_blind_acceptance") is not False):
            raise ValueError("exposure audit cannot restore blind eligibility")
        _strings(event.get("affected_family_ids"),"affected_family_ids")
    manifest, source_ids, names, hashes, family_splits = [], set(), set(), set(), {}
    if not isinstance(data.get("corpus"), list):
        raise ValueError("v3 corpus must be a list")
    for raw in data["corpus"]:
        if not isinstance(raw, dict):
            raise ValueError("v3 sources must be objects")
        sid, rel = raw.get("id"), raw.get("path")
        if not isinstance(sid, str) or not sid or sid in source_ids:
            raise ValueError("source IDs must be unique")
        if not isinstance(rel, str) or not (root / rel).resolve().is_relative_to(root):
            raise ValueError("corpus path escapes project root")
        name, digest = Path(rel).name.casefold(), raw.get("sha256", "")
        if name in names or digest in hashes or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("duplicate filename/bytes or invalid source SHA256")
        if type(raw.get("pages")) is not int or raw["pages"] < 1 or not raw.get("family"):
            raise ValueError("source pages and family are required")
        split = _split(raw)
        family_splits.setdefault(raw["family"], set()).add(split)
        manifest.append({**raw, "question_split": split})
        source_ids.add(sid); names.add(name); hashes.add(digest)
    if any(len(s) != 1 for s in family_splits.values()):
        raise ValueError("document family leaked across question partitions")
    sources = {s["id"]: s for s in manifest}
    items, seen_ids, seen_questions, query_families = [], set(), set(), {}
    if not isinstance(data.get("items"), list) or not data["items"]:
        raise ValueError("v3 dataset needs actual items, not quota placeholders")
    tolerance = data.get("page_tolerance", 0)
    if type(tolerance) is not int or tolerance < 0:
        raise ValueError("invalid page_tolerance")
    for raw in data["items"]:
        if not isinstance(raw, dict):
            raise ValueError("items must be objects")
        iid, question, truth = raw.get("id"), raw.get("question"), raw.get("ground_truth")
        if not isinstance(iid, str) or not iid or iid in seen_ids:
            raise ValueError("item ID missing or duplicated")
        if not isinstance(question, str) or not question.strip() or normalize_text(question) in seen_questions:
            raise ValueError("question missing or duplicated; variants are not new primary questions")
        if not isinstance(truth, str) or not truth.strip():
            raise ValueError("ground_truth missing")
        split, track = _split(raw), raw.get("track", "local")
        if track not in TRACKS or not raw.get("domain"):
            raise ValueError("track and domain are required")
        families = _strings(raw.get("family_ids"), "family_ids")
        for family in families:
            query_families.setdefault(family, set()).add(split)
            if family in family_splits and family_splits[family] != {split}:
                raise ValueError("question family crosses source partition")
        scope = _strings(raw.get("corpus_scope", []), "corpus_scope", nonempty=track in {"local", "structure"})
        if not set(scope) <= source_ids:
            raise ValueError("corpus_scope references unknown source")
        if split == "dev" and any(sources[s]["question_split"] != "dev" for s in scope):
            raise ValueError("development corpus_scope cannot expose held-out source families")
        evidence = _evidence(raw.get("evidence", []), sources, split)
        if not {e["source_id"] for e in evidence} <= set(scope):
            raise ValueError("gold evidence absent from explicit corpus_scope")
        if not {sources[e["source_id"]]["family"] for e in evidence} <= set(families):
            raise ValueError("evidence family missing from family_ids")
        refusal = raw.get("expect_refusal", False)
        if type(refusal) is not bool or (refusal and evidence):
            raise ValueError("invalid refusal/evidence contract")
        if track == "local" and not refusal and not evidence:
            raise ValueError("answerable local item needs source evidence")
        if refusal and not raw.get("negative_reason"):
            raise ValueError("negative items need an explicit reason and reviewed scope")
        slots = _slots(raw.get("fact_slots", []), evidence)
        if refusal and slots:
            raise ValueError("refusal has no positive fact slots")
        status, review = _review(raw)
        extra = dict(raw.get("task_metadata", {}))
        for key in ("negative_reason", "variants", "external_gold", "turns", "structure_gold", "source_group"):
            if key in raw:
                extra[key] = raw[key]
        if track == "dialogue":
            turns=extra.get("turns")
            if (not isinstance(turns,list) or len(turns)!=4 or any(not isinstance(turn,dict)
                    or not isinstance(turn.get("question"),str) or not turn["question"].strip()
                    or not isinstance(turn.get("expected"),dict) or not turn["expected"] for turn in turns)):
                raise ValueError("dialogue scenarios contain exactly four questions with explicit expected checks")
            for turn in turns:
                alternatives=turn["expected"].get("evidence_sets",[])
                for group in alternatives:
                    if not set(_strings(group,"turn evidence set")) <= {ev["id"] for ev in evidence}:
                        raise ValueError("turn references unknown source evidence")
        if track == "external" and (not isinstance(extra.get("external_gold"), dict) or not extra["external_gold"]):
            raise ValueError("external task needs external_gold, never fake PDF page evidence")
        if track == "structure":
            gold=extra.get("structure_gold")
            if (not isinstance(gold,dict) or gold.get("source_id") not in scope
                    or type(gold.get("page")) is not int or not 1<=gold["page"]<=sources[gold["source_id"]]["pages"]
                    or not isinstance(gold.get("objects"),list) or not gold["objects"]):
                raise ValueError("structure task needs a real source page and annotated objects")
        items.append(GoldenItem(iid, question, truth,
            tuple(dict.fromkeys((e["filename"], e["page"]) for e in evidence)),
            refusal, split, raw.get("category", "uncategorized"), evidence,
            track, raw["domain"], families, scope, status, review, slots, extra))
        seen_ids.add(iid); seen_questions.add(normalize_text(question))
    if any(len(s) > 1 for s in query_families.values()):
        raise ValueError("query/evidence family leaked across partitions")
    golden = GoldenSet(tuple(s["path"] for s in manifest), tuple(items), tolerance,
                       tuple(manifest), hashlib.sha256(raw_bytes).hexdigest(), 3, audit)
    if verify_files:
        verify_corpus(golden, root)
    return golden
