"""检索侧指标：Recall@K、MRR、nDCG@K。

检索结果的 (filename, page) 命中标注证据集合（允许 page_tolerance 容差）即算相关。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

# 一条证据 = (文件名, 页码)
Evidence = tuple[str, int]


def _norm_name(name: str | None) -> str:
    """归一化文件名，忽略路径与大小写差异。"""
    if not name:
        return ""
    return str(name).strip().replace("\\", "/").split("/")[-1].lower()


def is_relevant(
    meta: dict[str, Any],
    evidence: set[Evidence],
    page_tolerance: int = 0,
) -> bool:
    """判断单个检索结果（按其 metadata）是否命中证据集合。"""
    name = _norm_name(meta.get("filename"))
    try:
        page = int(meta.get("page"))
    except (TypeError, ValueError):
        page = None

    for ev_name, ev_page in evidence:
        if _norm_name(ev_name) != name:
            continue
        if page is None:
            # 无页码信息时退化为"文件名命中即算相关"
            return True
        if abs(page - ev_page) <= page_tolerance:
            return True
    return False


def _relevance_flags(
    retrieved_meta: list[dict[str, Any]],
    evidence: set[Evidence],
    page_tolerance: int,
) -> list[int]:
    """把有序检索结果映射为 0/1 相关性序列。"""
    return [
        1 if is_relevant(meta, evidence, page_tolerance) else 0
        for meta in retrieved_meta
    ]


def recall_at_k(
    retrieved_meta: list[dict[str, Any]],
    evidence: set[Evidence],
    k: int,
    page_tolerance: int = 0,
) -> float:
    """Recall@K：前 K 个结果覆盖的证据页比例（按页去重）。"""
    if not evidence:
        return 0.0
    hit_pages: set[Evidence] = set()
    for meta in retrieved_meta[:k]:
        name = _norm_name(meta.get("filename"))
        try:
            page = int(meta.get("page"))
        except (TypeError, ValueError):
            continue
        for ev_name, ev_page in evidence:
            if _norm_name(ev_name) == name and abs(page - ev_page) <= page_tolerance:
                hit_pages.add((ev_name, ev_page))
    return len(hit_pages) / len(evidence)


def mrr(
    retrieved_meta: list[dict[str, Any]],
    evidence: set[Evidence],
    page_tolerance: int = 0,
) -> float:
    """MRR：首个相关结果排名的倒数；无相关结果则为 0。"""
    flags = _relevance_flags(retrieved_meta, evidence, page_tolerance)
    for idx, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(
    retrieved_meta: list[dict[str, Any]],
    evidence: set[Evidence],
    k: int,
    page_tolerance: int = 0,
) -> float:
    """nDCG@K：二值相关性下的归一化折损累积增益。"""
    flags = _relevance_flags(retrieved_meta, evidence, page_tolerance)[:k]
    dcg = sum(flag / math.log2(i + 2) for i, flag in enumerate(flags))
    # 理想情形：所有相关项排在最前；理想相关项数上限为证据页数
    ideal_hits = min(len(evidence), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RetrievalScores:
    """单个问题的检索指标结果。"""

    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)
    num_evidence: int = 0
    num_retrieved: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "recall": {f"@{k}": round(v, 4) for k, v in self.recall.items()},
            "mrr": round(self.mrr, 4),
            "ndcg": {f"@{k}": round(v, 4) for k, v in self.ndcg.items()},
            "num_evidence": self.num_evidence,
            "num_retrieved": self.num_retrieved,
        }


def evaluate_retrieval(
    retrieved_meta: list[dict[str, Any]],
    evidence: Iterable[Evidence],
    k_values: Iterable[int] = (1, 3, 5, 10),
    page_tolerance: int = 0,
) -> RetrievalScores:
    """计算单个问题的全部检索指标。"""
    ev_set: set[Evidence] = {(_norm_name(n), int(p)) for n, p in evidence}
    scores = RetrievalScores(
        num_evidence=len(ev_set),
        num_retrieved=len(retrieved_meta),
    )
    for k in k_values:
        scores.recall[k] = recall_at_k(retrieved_meta, ev_set, k, page_tolerance)
        scores.ndcg[k] = ndcg_at_k(retrieved_meta, ev_set, k, page_tolerance)
    scores.mrr = mrr(retrieved_meta, ev_set, page_tolerance)
    return scores
