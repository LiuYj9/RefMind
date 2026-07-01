"""评测编排：建库入库 → 检索/问答 → 打分 → 汇总，入口 run_evaluation。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional

from refmind import storage
from refmind.config import settings
from refmind.rag import (
    answer_question,
    build_retriever,
    get_retriever,
    invalidate_retriever,
)
from refmind.rag.graph import NO_CONTEXT_REPLY
from refmind.services import ingest_pdf, remove_group

from .metrics import GenerationScores, RetrievalScores, evaluate_generation, evaluate_retrieval


# golden set 载入
@dataclass
class GoldenItem:
    """单条标注样本。"""

    id: str
    question: str
    ground_truth: str
    evidence: list[tuple[str, int]]
    expect_refusal: bool = False


@dataclass
class GoldenSet:
    """完整 golden set。"""

    corpus: list[str]
    items: list[GoldenItem]
    page_tolerance: int = 0


def load_golden_set(path: str | Path) -> GoldenSet:
    """从 JSON 载入 golden set。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[GoldenItem] = []
    for raw in data.get("items", []):
        evidence = [
            (e["filename"], int(e["page"])) for e in raw.get("evidence", [])
        ]
        items.append(
            GoldenItem(
                id=str(raw["id"]),
                question=raw["question"],
                ground_truth=raw.get("ground_truth", ""),
                evidence=evidence,
                expect_refusal=bool(raw.get("expect_refusal", False)),
            )
        )
    return GoldenSet(
        corpus=data.get("corpus", []),
        items=items,
        page_tolerance=int(data.get("page_tolerance", 0)),
    )


# 结果结构
@dataclass
class ItemResult:
    """单个问题的完整评测结果。"""

    id: str
    question: str
    answer: str
    expect_refusal: bool
    is_refusal: bool
    retrieval: Optional[RetrievalScores] = None
    generation: Optional[GenerationScores] = None
    latency_s: float = 0.0
    retrieved: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "expect_refusal": self.expect_refusal,
            "is_refusal": self.is_refusal,
            "latency_s": round(self.latency_s, 3),
            "retrieval": self.retrieval.as_dict() if self.retrieval else None,
            "generation": self.generation.as_dict() if self.generation else None,
            "retrieved": self.retrieved,
        }


def _is_refusal(answer: str) -> bool:
    """粗略判断答案是否为"拒答/未找到"话术。"""
    if not answer:
        return True
    markers = [NO_CONTEXT_REPLY, "无法回答", "未找到相关", "暂未找到"]
    return any(m in answer for m in markers)


def _build_retriever_k(group_id: int, k: int):
    """构建一个 top-k 指定的临时检索器（评测检索指标用，取召回原始排序）。"""
    return build_retriever(group_id, k=k)


# 主流程
def run_evaluation(
    golden_path: str | Path,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    production_top_k: int = 5,
    judge: bool = True,
    keep_group: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """执行完整评测，返回结果字典（含逐题明细与汇总）。"""
    settings.ensure_dirs()
    storage.init_db()

    golden = load_golden_set(golden_path)
    if not golden.items:
        raise ValueError("golden set 为空，请先标注 items。")

    project_root = settings.project_root
    max_k = max(k_values)

    # 生产问答最终保留的上下文条数（answer_question 会读取该配置）
    settings.retrieval_top_k = production_top_k
    settings.rerank_top_n = production_top_k

    group = storage.create_group(f"eval_{int(time.time())}")
    log(f"已创建评测文献库：group_id={group.id}")

    try:
        # 1) 入库 corpus
        for rel in golden.corpus:
            pdf_path = (project_root / rel).resolve()
            if not pdf_path.exists():
                raise FileNotFoundError(
                    f"找不到语料 PDF：{pdf_path}。"
                    f"若使用内置合成语料，请先运行 evaluation/make_sample_corpus.py。"
                )
            log(f"入库：{pdf_path.name}")
            ingest_pdf(group.id, pdf_path, pdf_path.name, make_summary=False)

        # 2) 检索器：一个用于检索指标（max_k），生产缓存用于问答（production_top_k）
        eval_retriever = _build_retriever_k(group.id, max_k)
        if eval_retriever is None:
            raise RuntimeError("检索器构建失败：文献库可能为空。")
        invalidate_retriever(group.id)  # 让 answer_question 按生产 top-k 重建

        # 3) 逐题评测
        results: list[ItemResult] = []
        for item in golden.items:
            log(f"评测 [{item.id}] {item.question[:40]} ...")

            retrieved_docs = eval_retriever.invoke(item.question)
            retrieved_meta = [d.metadata or {} for d in retrieved_docs]

            t0 = time.perf_counter()
            qa = answer_question(item.question, group.id)
            latency = time.perf_counter() - t0

            answer = qa.get("answer", "")
            ctx_docs = qa.get("documents") or []
            contexts = [d.page_content for d in ctx_docs]

            res = ItemResult(
                id=item.id,
                question=item.question,
                answer=answer,
                expect_refusal=item.expect_refusal,
                is_refusal=_is_refusal(answer),
                latency_s=latency,
                retrieved=[
                    {
                        "filename": m.get("filename"),
                        "page": m.get("page"),
                    }
                    for m in retrieved_meta[:max_k]
                ],
            )

            # 检索指标：仅对有证据标注的题计算
            if item.evidence:
                res.retrieval = evaluate_retrieval(
                    retrieved_meta,
                    item.evidence,
                    k_values=k_values,
                    page_tolerance=golden.page_tolerance,
                )

            # 生成指标：拒答题不算四项指标（用 refusal 正确性衡量）
            if judge and not item.expect_refusal:
                res.generation = evaluate_generation(
                    question=item.question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=item.ground_truth,
                )

            results.append(res)

        summary = _aggregate(results, k_values)
        return {
            "golden_path": str(golden_path),
            "config": {
                "k_values": list(k_values),
                "production_top_k": production_top_k,
                "judge": judge,
                "page_tolerance": golden.page_tolerance,
            },
            "summary": summary,
            "items": [r.as_dict() for r in results],
        }
    finally:
        if keep_group:
            log(f"保留评测文献库 group_id={group.id}")
        else:
            remove_group(group.id)
            log(f"已清理评测文献库 group_id={group.id}")


def _mean_or_none(values: list[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def _aggregate(results: list[ItemResult], k_values: tuple[int, ...]) -> dict[str, Any]:
    """汇总逐题指标为整体指标。"""
    answerable = [r for r in results if not r.expect_refusal]
    refusal_items = [r for r in results if r.expect_refusal]

    # 检索侧
    retrieval_avg: dict[str, Any] = {"recall": {}, "ndcg": {}, "mrr": None}
    with_retrieval = [r for r in answerable if r.retrieval is not None]
    if with_retrieval:
        for k in k_values:
            retrieval_avg["recall"][f"@{k}"] = _mean_or_none(
                [r.retrieval.recall.get(k) for r in with_retrieval]
            )
            retrieval_avg["ndcg"][f"@{k}"] = _mean_or_none(
                [r.retrieval.ndcg.get(k) for r in with_retrieval]
            )
        retrieval_avg["mrr"] = _mean_or_none([r.retrieval.mrr for r in with_retrieval])

    # 生成侧
    with_gen = [r for r in answerable if r.generation is not None]
    generation_avg = {
        "faithfulness": _mean_or_none([r.generation.faithfulness for r in with_gen]),
        "answer_relevance": _mean_or_none(
            [r.generation.answer_relevance for r in with_gen]
        ),
        "context_precision": _mean_or_none(
            [r.generation.context_precision for r in with_gen]
        ),
        "context_recall": _mean_or_none(
            [r.generation.context_recall for r in with_gen]
        ),
    }

    # 拒答正确率：应拒答且确实拒答
    refusal_accuracy = None
    if refusal_items:
        correct = sum(1 for r in refusal_items if r.is_refusal)
        refusal_accuracy = round(correct / len(refusal_items), 4)

    latencies = [r.latency_s for r in results]
    latencies_sorted = sorted(latencies)

    def _pct(p: float) -> Optional[float]:
        if not latencies_sorted:
            return None
        idx = min(len(latencies_sorted) - 1, int(round(p * (len(latencies_sorted) - 1))))
        return round(latencies_sorted[idx], 3)

    return {
        "num_items": len(results),
        "num_answerable": len(answerable),
        "num_refusal": len(refusal_items),
        "retrieval": retrieval_avg,
        "generation": generation_avg,
        "refusal_accuracy": refusal_accuracy,
        "latency": {
            "mean_s": round(mean(latencies), 3) if latencies else None,
            "p50_s": _pct(0.50),
            "p95_s": _pct(0.95),
            "p99_s": _pct(0.99),
        },
    }
