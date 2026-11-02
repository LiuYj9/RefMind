"""Evidence-page metrics: missing pages never count; repeated pages gain no credit."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

Evidence = tuple[str, int]


def _norm_name(name: str | None) -> str:
    return str(name or "").strip().replace("\\", "/").split("/")[-1].casefold()


def _normalize(evidence: Iterable[Evidence]) -> set[Evidence]:
    return {(_norm_name(n), int(p)) for n, p in evidence}


def _matches(meta: dict[str, Any], evidence: set[Evidence], tolerance: int) -> set[Evidence]:
    start = meta.get("page_start", meta.get("page"))
    end = meta.get("page_end", start)
    try:
        if isinstance(start, bool) or isinstance(end, bool):
            return set()
        if any(isinstance(v, float) and not v.is_integer() for v in (start, end)):
            return set()
        start, end = int(start), int(end)
    except (TypeError, ValueError, OverflowError):
        return set()
    if start < 1 or end < start:
        return set()
    name = _norm_name(meta.get("filename"))
    return {(n, p) for n, p in evidence if n == name and start-tolerance <= p <= end+tolerance}


def is_relevant(meta, evidence, page_tolerance=0):
    return bool(_matches(meta, _normalize(evidence), page_tolerance))


def _relevance_flags(retrieved_meta, evidence, page_tolerance):
    seen, flags = set(), []
    for meta in retrieved_meta:
        fresh = _matches(meta, evidence, page_tolerance) - seen
        flags.append(int(bool(fresh)))
        # Binary gain per rank; all covered pages become seen, so a duplicate range
        # cannot earn a second gain by being assigned to another page.
        if fresh:
            seen.update(fresh)
    return flags


def recall_at_k(retrieved_meta, evidence, k, page_tolerance=0):
    evidence = _normalize(evidence)
    hits = set()
    for meta in retrieved_meta[:k]:
        hits.update(_matches(meta, evidence, page_tolerance))
    return len(hits)/len(evidence) if evidence else 0.0


def mrr(retrieved_meta, evidence, page_tolerance=0):
    for rank, flag in enumerate(_relevance_flags(retrieved_meta, _normalize(evidence), page_tolerance), 1):
        if flag:
            return 1.0/rank
    return 0.0


def ndcg_at_k(retrieved_meta, evidence, k, page_tolerance=0):
    evidence = _normalize(evidence)
    flags = _relevance_flags(retrieved_meta[:k], evidence, page_tolerance)
    dcg = sum(flag/math.log2(i+2) for i, flag in enumerate(flags))
    ideal = sum(1/math.log2(i+2) for i in range(min(len(evidence), k)))
    return dcg/ideal if ideal else 0.0


@dataclass
class RetrievalScores:
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)
    num_evidence: int = 0
    num_retrieved: int = 0

    def as_dict(self):
        return {"recall": {f"@{k}": round(v, 4) for k, v in self.recall.items()},
                "mrr": round(self.mrr, 4), "ndcg": {f"@{k}": round(v, 4) for k, v in self.ndcg.items()},
                "num_evidence": self.num_evidence, "num_retrieved": self.num_retrieved}


def evaluate_retrieval(retrieved_meta, evidence, k_values=(1, 3, 5, 10), page_tolerance=0):
    ks = tuple(k_values)
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("K values must be positive integers")
    if type(page_tolerance) is not int or page_tolerance < 0:
        raise ValueError("page_tolerance must be nonnegative")
    ev = _normalize(evidence)
    return RetrievalScores({k: recall_at_k(retrieved_meta, ev, k, page_tolerance) for k in ks},
                           mrr(retrieved_meta, ev, page_tolerance),
                           {k: ndcg_at_k(retrieved_meta, ev, k, page_tolerance) for k in ks},
                           len(ev), len(retrieved_meta))
