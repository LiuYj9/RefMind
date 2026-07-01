"""RAG 评测命令行入口。

常用：
    python evaluation/run_eval.py               # 完整评测
    python evaluation/run_eval.py --no-judge    # 只跑检索侧，不调用裁判
    python evaluation/run_eval.py --golden x.json --k 1 3 5 10

报告写到 evaluation/reports/，同时输出 JSON 与 Markdown。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许从项目根目录导入 refmind
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# 避免 Windows GBK 控制台输出中文/emoji 报错
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from evaluation.evaluator import run_evaluation  # noqa: E402


def _ensure_sample_corpus(golden_path: Path) -> None:
    """若 golden set 引用了内置合成语料却尚未生成，则自动生成。"""
    try:
        data = json.loads(golden_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return
    corpus = data.get("corpus", [])
    needs_sample = any("RefMindNet.pdf" in c for c in corpus)
    target = _PROJECT_ROOT / "evaluation" / "corpus" / "RefMindNet.pdf"
    if needs_sample and not target.exists():
        from evaluation.make_sample_corpus import build_sample_pdf

        print("首次运行：生成内置合成语料 RefMindNet.pdf ...")
        build_sample_pdf(target)


def _fmt(value) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _to_markdown(report: dict) -> str:
    s = report["summary"]
    cfg = report["config"]
    lines: list[str] = []
    lines.append("# RefMind RAG 评测报告\n")
    lines.append(f"- golden set：`{report['golden_path']}`")
    lines.append(f"- 题目数：{s['num_items']}（可答 {s['num_answerable']} / 拒答 {s['num_refusal']}）")
    lines.append(f"- K 取值：{cfg['k_values']} ｜ 生产 top-k：{cfg['production_top_k']} ｜ LLM 裁判：{cfg['judge']}")
    lines.append("")

    lines.append("## 检索侧指标（均值）\n")
    r = s["retrieval"]
    lines.append("| 指标 | " + " | ".join(f"@{k}" for k in cfg["k_values"]) + " |")
    lines.append("|---|" + "---|" * len(cfg["k_values"]))
    lines.append(
        "| Recall | "
        + " | ".join(_fmt(r["recall"].get(f"@{k}")) for k in cfg["k_values"])
        + " |"
    )
    lines.append(
        "| nDCG | "
        + " | ".join(_fmt(r["ndcg"].get(f"@{k}")) for k in cfg["k_values"])
        + " |"
    )
    lines.append(f"\n- MRR：{_fmt(r['mrr'])}\n")

    lines.append("## 生成侧指标（均值，LLM-as-judge）\n")
    g = s["generation"]
    lines.append("| Faithfulness | Answer Relevance | Context Precision | Context Recall |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| {_fmt(g['faithfulness'])} | {_fmt(g['answer_relevance'])} "
        f"| {_fmt(g['context_precision'])} | {_fmt(g['context_recall'])} |"
    )
    lines.append("")

    lines.append("## 其他\n")
    lines.append(f"- 拒答正确率：{_fmt(s['refusal_accuracy'])}")
    lat = s["latency"]
    lines.append(
        f"- 端到端问答延迟：mean {_fmt(lat['mean_s'])}s ｜ P50 {_fmt(lat['p50_s'])}s "
        f"｜ P95 {_fmt(lat['p95_s'])}s ｜ P99 {_fmt(lat['p99_s'])}s"
    )
    lines.append("")

    lines.append("## 逐题明细\n")
    lines.append("| ID | 拒答期望/实际 | Recall@" + str(max(cfg["k_values"])) + " | MRR | Faith. | Ans.Rel | Ctx.Prec | Ctx.Rec | 延迟(s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    maxk = max(cfg["k_values"])
    for it in report["items"]:
        ret = it["retrieval"] or {}
        gen = it["generation"] or {}
        recall = (ret.get("recall") or {}).get(f"@{maxk}")
        lines.append(
            f"| {it['id']} "
            f"| {it['expect_refusal']}/{it['is_refusal']} "
            f"| {_fmt(recall)} "
            f"| {_fmt(ret.get('mrr'))} "
            f"| {_fmt(gen.get('faithfulness'))} "
            f"| {_fmt(gen.get('answer_relevance'))} "
            f"| {_fmt(gen.get('context_precision'))} "
            f"| {_fmt(gen.get('context_recall'))} "
            f"| {it['latency_s']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="RefMind RAG 评测")
    parser.add_argument(
        "--golden",
        default=str(_PROJECT_ROOT / "evaluation" / "golden_set.json"),
        help="golden set JSON 路径",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[1, 3, 5, 10], help="Recall/nDCG 的 K 取值"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="生产问答使用的检索 top-k"
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="跳过 LLM 裁判，仅评测检索侧"
    )
    parser.add_argument(
        "--keep-group", action="store_true", help="评测后保留临时文献库（默认清理）"
    )
    args = parser.parse_args()

    golden_path = Path(args.golden).resolve()
    _ensure_sample_corpus(golden_path)

    report = run_evaluation(
        golden_path=golden_path,
        k_values=tuple(args.k),
        production_top_k=args.top_k,
        judge=not args.no_judge,
        keep_group=args.keep_group,
    )

    reports_dir = _PROJECT_ROOT / "evaluation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"report_{stamp}.json"
    md_path = reports_dir / f"report_{stamp}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    print("\n" + _to_markdown(report))
    print(f"\n报告已保存：\n  {json_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
