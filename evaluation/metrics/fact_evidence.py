"""Required fact slots are AND; each slot accepts OR sets of AND source evidence.

These metrics measure evidence availability, never answer correctness. A page hit
alone cannot prove a numerical fact or its experimental conditions survived packing.
"""
from __future__ import annotations

import json
import re
import unicodedata


def _text(value):
    # Preserve case: mA and MA are scientifically different units.
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))


def _list(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    return value if isinstance(value, list) else []


def _positive_int(value):
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
        return result if result > 0 and result == float(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def evidence_hit(document, evidence):
    meta = document.get("metadata", {})
    name = str(meta.get("filename", "")).replace("\\", "/").rsplit("/", 1)[-1].casefold()
    if name != evidence["filename"].casefold():
        return False
    page = evidence["page"]
    locations = _list(meta.get("source_locations"))
    if not locations:
        locations = [{"page_start": meta.get("page_start", meta.get("page")),
                      "page_end": meta.get("page_end", meta.get("page_start", meta.get("page")))}]
    valid_page = any((start := _positive_int(loc.get("page_start", loc.get("page")))) is not None
                     and (end := _positive_int(loc.get("page_end", start))) is not None
                     and start <= page <= end for loc in locations if isinstance(loc, dict))
    if not valid_page:
        return False
    if evidence.get("evidence_source", "pdf_text") == "visual":
        # Require explicit source-region identity; merely returning the same page is insufficient.
        block = evidence.get("block_id")
        blocks = _list(meta.get("block_ids")) + [meta.get("block_id")]
        return bool(block and block in blocks and meta.get("image_path"))
    quote = _text(evidence.get("quote"))
    return bool(quote and quote in _text(document.get("page_content")))


def evaluate_fact_evidence(documents, evidence, slots, k_values=(1, 3, 5, 10)):
    required = [slot for slot in slots if slot.get("required", True)]
    if not required:
        return None
    by_id = {entry["id"]: entry for entry in evidence}
    results = {}
    for k in k_values:
        hits = {eid for eid, entry in by_id.items()
                if any(evidence_hit(document, entry) for document in documents[:k])}
        covered = {slot["id"]: any(all(eid in hits for eid in group)
                       for group in slot["evidence_sets"]) for slot in required}
        results[f"@{k}"] = {"covered_slots": sum(covered.values()), "required_slots": len(required),
                            "slot_coverage": sum(covered.values()) / len(required),
                            "all_required_covered": all(covered.values()), "slots": covered}
    return results
