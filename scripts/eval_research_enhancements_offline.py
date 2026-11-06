"""Run 18 synthetic mechanism fixtures in four configurations, with no network.

This is a regression/trace demonstrator, NOT a real-corpus quality benchmark.
Hybrid fusion, source merging, expansion, extractive views and EvidenceBundle are
production functions. Recall lists, planning and rerank scores are injected.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from refmind.config import settings
from refmind.rag import retrieval, research_pipeline, research_queries
from refmind.rag.evidence_bundle import pack_evidence
from refmind.rag.extractive_views import build_extractive_views, validate_extractive_view

CONFIGURATIONS = {"baseline": (False, False), "multiquery_only": (True, False),
                  "compression_only": (False, True), "multiquery_compression": (True, True)}
CASES = [
    ("M01", "keyword", "铜屏蔽如何影响损耗？", "method"),
    ("M02", "bilingual", "铜屏蔽降低交流损耗 AC loss 的依据？", "method"),
    ("M03", "complementary", "铜屏蔽的方法和适用条件？", "method"),
    ("M04", "source_identity", "两篇研究分别使用什么方法？", "comparison"),
    ("M05", "numeric", "77 K 下铜屏蔽的影响？", "method"),
    ("M06", "negation", "是否可以推断所有工况都下降？", "method"),
    ("M07", "table", "表中列出的损耗分量有哪些？", "method"),
    ("M08", "equation", "公式中损耗如何定义？", "method"),
    ("M09", "summary", "概述论文的方法、条件与边界。", "summary"),
    ("M10", "comparison", "比较两个选定研究的方法与边界。", "comparison"),
    ("M11", "duplicate", "铜屏蔽如何减小涡流？", "method"),
    ("M12", "empty", "这些合成片段有没有电解液配方？", "method"),
    ("M13", "branch_error", "铜屏蔽有哪些条件和局限？", "method"),
    ("M14", "planner_error", "解释铜屏蔽的方法和适用条件。", "method"),
    ("M15", "empty_scope", "解释我选定的文献。", "method"),
    ("M16", "version_change", "解释检索期间更新的文献。", "method"),
    ("M17", "scorer_error", "铜屏蔽降低损耗的依据？", "method"),
    ("M18", "budget", "在有限上下文中比较两篇研究。", "comparison"),
]

BACKGROUND = ("本段介绍研究讨论的起点。下面补充一般背景说明。"
    "这里回顾写作习惯与编辑流程，介绍章节组织、图例编排和读者导航，帮助读者了解一般的论文结构。"
    "另一段详细介绍行文安排，包括引言如何展开、论述如何过渡、附录怎样组织以及结尾如何回顾，主要作用是向读者说明文章的结构。"
    "接下来还包含一段较长的历史回顾和一般性说明，内容只是帮助读者了解文章的组织过程，与正在提取的具体研究结论之间保持相当距离。"
    "下面进入相关结论。铜屏蔽在研究工况下减小了线圈感生涡流。这一说明与前句相连。"
    "最后附带编辑性的阅读提示，重新概述文章采用的写作顺序和过渡安排，帮助读者从结构层面了解整篇文章的行文组织。")


class FixedRecall(BaseRetriever):
    mapping: dict
    def _get_relevant_documents(self, query, *, run_manager):
        return deepcopy(self.mapping.get(query, []))


def fixture_documents(kind):
    def doc(source, chunk, text, content_type="paragraph", index=1):
        return Document(page_content=text, metadata={"group_id": 7, "doc_id": source,
            "chunk_id": chunk, "chunk_index": index, "version": "v1", "file_sha256": f"hash{source}",
            "filename": f"synthetic-{source}.pdf", "paper_title": f"合成机制材料{source}",
            "page": 1, "section": "Methods", "content_type": content_type,
            "_retrieval_identity": f"7:{source}:v1:{chunk}"})
    main = BACKGROUND
    if kind == "numeric":
        main += "样品在77 K与50 Hz下测量，结果为3.2%。"
    if kind == "negation":
        main += "这并不意味着所有工况下系统总损耗都会下降。"
    ds = [doc(1, "e1", main), doc(2, "e1", "另一研究分析不同对象，不提供跨工况最优结论。"),
          doc(1, "context", "本实验仅讨论限定工况；不能推断全部系统损耗或全部运行范围。", index=2),
          doc(1, "noise", "本文讨论编辑排版和章节安排。", index=8)]
    if kind == "table":
        ds[0] = doc(1, "table", "| 项目 | 值 |\n| --- | --- |\n| 线圈损耗 | 3 W |\n| 屏蔽损耗 | 1 W |", "table")
    if kind == "equation":
        ds[0] = doc(1, "equation", "$$P=\\int_V \\mathbf{J}\\cdot\\mathbf{E}\\,dV$$", "equation")
    return ds


def sentence_scores(query, sentences):
    return [.95 if "铜屏蔽" in value else .02 for value in sentences]
sentence_scores.score_type = "deterministic_offline_fixture"


def run_case(case, multiquery, compression):
    case_id, kind, question, task_type = case
    docs = fixture_documents(kind)
    originals = deepcopy(docs)
    rows = {i: SimpleNamespace(id=i, group_id=7, status="ready", index_version="v1",
        file_sha256=f"hash{i}", paper_title=f"合成机制材料{i}", filename=f"synthetic-{i}.pdf") for i in (1, 2)}
    branch = question + " 适用范围"
    keyword = {question: [docs[3], docs[0]], branch: [docs[0], docs[2], docs[1]]}
    vector = {question: [docs[0], docs[1]], branch: [docs[2], docs[0]]}
    if kind == "empty":
        keyword = vector = {}
    if kind == "duplicate":
        keyword[question].append(docs[0])
    fused = retrieval._ObservedEnsembleRetriever(retrievers=[FixedRecall(mapping=keyword), FixedRecall(mapping=vector)],
        weights=[.5, .5], id_key="_retrieval_identity")
    def recall(query):
        if kind == "branch_error" and query == branch:
            raise RuntimeError("injected_offline_branch_error")
        if kind == "version_change":
            rows[1].index_version = "v2"
        return fused.invoke(query)
    def rerank(query, candidates, **kwargs):
        ordered = sorted(candidates, key=lambda d: (d.metadata["chunk_id"] != docs[0].metadata["chunk_id"],
                                                   d.metadata["doc_id"], d.metadata["chunk_id"]))
        return [Document(page_content=d.page_content, metadata={**d.metadata,
            "rerank_backend": "deterministic_offline_fixture", "rerank_score": 1. / rank,
            "score_type": "fixture_order_only_not_model_score"}) for rank, d in enumerate(ordered, 1)]
    selected = [] if kind == "empty_scope" else [1, 2]
    with ExitStack() as stack:
        for name, value in {"research_multi_query_enabled": multiquery, "research_max_queries": 3,
                            "research_query_workers": 2, "research_query_timeout": 1.}.items():
            stack.enter_context(patch.object(settings, name, value))
        stack.enter_context(patch.object(research_pipeline.storage, "get_document", side_effect=rows.get))
        stack.enter_context(patch.object(research_pipeline.storage, "list_documents", return_value=list(rows.values())))
        stack.enter_context(patch.object(retrieval, "get_retriever", return_value=Mock(invoke=recall)))
        stack.enter_context(patch("refmind.rag.document_processor.load_group_documents", return_value=deepcopy(docs)))
        stack.enter_context(patch.object(research_pipeline, "rerank", side_effect=rerank))
        plan = {"queries": [question, branch], "source": "injected_offline_fixture"}
        if kind == "planner_error":
            plan = research_queries.plan_research_queries(question, planner_llm=Mock(invoke=Mock(side_effect=RuntimeError("offline"))))
        result = research_pipeline.retrieve_research_candidates(question, 7, selected_doc_ids=selected,
            allowed_doc_ids=[1, 2], task_type=task_type, query_plan=plan)
        ranked = research_pipeline.run_research_rerank(question, result["documents"])
        records = []
        viewed = ranked
        if compression:
            scoring = sentence_scores
            if kind == "scorer_error":
                def scoring(query, sentences):
                    raise RuntimeError("injected_offline_sentence_error")
                scoring.score_type = "deterministic_offline_fixture_error"
            views = build_extractive_views(question, ranked, scorer=scoring, task_kind=task_type)
            viewed, records = views.documents, views.records
        packed = pack_evidence(question, viewed, task_type=task_type,
            evidence_tokens=350 if kind == "budget" else 4000,
            required_source_ids=[f"library:7:{doc_id}" for doc_id in selected], counter=lambda text: len(text.encode("utf-8")))
    checked_spans = all(validate_extractive_view(before.page_content, after.page_content, record)
                        for before, after, record in zip(ranked, viewed, records))
    identity_before = [(d.metadata["doc_id"], d.metadata["chunk_id"]) for d in ranked]
    identity_after = [(d.metadata["doc_id"], d.metadata["chunk_id"]) for d in viewed]
    protected_unchanged = all(a.page_content == b.page_content for a, b in zip(ranked, viewed)
                              if a.metadata.get("content_type") in {"table", "equation"})
    checks = {"source_documents_unchanged": docs == originals, "view_identity_preserved": identity_before == identity_after,
              "span_reconstruction_valid": checked_spans, "tables_equations_unchanged": protected_unchanged,
              "query_count_bounded": len(result["queries"]) <= (3 if multiquery else 1),
              "original_query_first": result["queries"][0] == question,
              "allowed_sources_only": all(d.metadata["doc_id"] in selected for d in result["documents"])}
    return {"id": case_id, "kind": kind, "question": question, "checks": checks,
        "passed": all(checks.values()), "retrieval_failed": result["retrieval_failed"],
        "degraded": result["degraded"], "warnings": result["warnings"],
        "candidate_count": len(ranked), "packed_count": len(packed.documents),
        "before_chars": sum(len(d.page_content) for d in ranked), "after_chars": sum(len(d.page_content) for d in viewed),
        "compression_records": records, "trace": result["trace"], "packing": packed.trace,
        "quality_metrics": None, "quality_status": "not_measured_synthetic_mechanism_fixture"}


def serializable(value):
    if isinstance(value, Document):
        return {"text": value.page_content, "metadata": serializable(value.metadata)}
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def run():
    report = {"schema": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "offline_synthetic_mechanism_ablation", "online_model_calls": 0,
        "real_corpus_quality_status": "pending_source_review_and_real_run",
        "limitations": ["合成文本不是论文原文", "召回排名、查询规划和重排分数采用注入fixture",
            "实际运行排名融合、身份合并、扩邻、压缩视图和上下文预算函数", "不能据此报告论文库检索或回答质量提升"],
        "configurations": {}}
    for name, flags in CONFIGURATIONS.items():
        rows = [run_case(case, *flags) for case in CASES]
        report["configurations"][name] = {"multiquery": flags[0], "compression": flags[1],
            "mechanism_cases_passed": sum(row["passed"] for row in rows), "case_count": len(rows), "cases": rows}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serializable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {name: f'{value["mechanism_cases_passed"]}/{value["case_count"]}' for name, value in report["configurations"].items()}
    print(json.dumps({"offline_mechanism_checks": summary, "real_corpus_quality": "not_run", "output": str(args.output)}, ensure_ascii=False))
    return 0 if all(value["mechanism_cases_passed"] == value["case_count"] for value in report["configurations"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
