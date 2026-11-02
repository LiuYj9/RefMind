"""Safe evaluation CLI. Default is offline validation; paid operations require opt-in."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from importlib.metadata import PackageNotFoundError, version
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from evaluation.dataset import load_golden_set
from evaluation.evaluator import evaluate_items


def _workspace(path: Path, *, create: bool) -> Path:
    path = path.resolve()
    parent = (PROJECT_ROOT / "evaluation" / "workspaces").resolve()
    if path == parent or not path.is_relative_to(parent):
        raise ValueError("workspace must be a child of evaluation/workspaces")
    marker = path / ".refmind-eval"
    if create:
        path.mkdir(parents=True, exist_ok=False)
        marker.write_text("RefMind evaluation-owned storage\n", encoding="utf-8")
    elif not marker.is_file():
        raise ValueError("not a prepared evaluation workspace")
    return path


def worker_environment(workspace: Path) -> dict:
    env = os.environ.copy()
    for key, relative in {"DATABASE_PATH": "db/evaluation.db", "UPLOAD_DIR": "uploads",
                          "PARSED_DIR": "parsed", "DOCSTORE_DIR": "docstore",
                          "CHROMA_PERSIST_DIR": "chroma"}.items():
        env[key] = str(workspace / relative)
    env.update(REFMIND_EVAL_WORKSPACE=str(workspace), IMAGE_SUMMARY_ENABLED="false",
               LONG_TERM_MEMORY_ENABLED="false", ACADEMIC_SEARCH_ENABLED="false",
               REFMIND_PLUGIN_MODULES="", REFMIND_MCP_SERVERS="[]",
               PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    return env


def run_backend(request: dict, workspace: Path) -> dict:
    # Child lifetime bounds settings mutations and model/retriever caches to this run.
    with tempfile.TemporaryDirectory(prefix="eval-result-") as tmp:
        result = Path(tmp) / "result.json"
        request = {**request, "result_path": str(result)}
        try:
            proc = subprocess.run([sys.executable, "-B", "-m", "evaluation.worker"],
                                  input=json.dumps(request), text=True, encoding="utf-8",
                                  cwd=PROJECT_ROOT, env=worker_environment(workspace),
                                  capture_output=True, timeout=1800)
        except subprocess.TimeoutExpired:
            return {"schema_version": 2, "mode": request["mode"], "status": "failed",
                    "error": {"type": "TimeoutExpired", "message": "isolated worker exceeded 1800 seconds"}}
        if proc.stderr:
            import re
            diagnostic = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", proc.stderr)
            (workspace / "worker.stderr.log").write_text(diagnostic, encoding="utf-8")
        if not result.exists():
            raise RuntimeError(f"evaluation worker exited {proc.returncode} without a result")
        report = json.loads(result.read_text(encoding="utf-8"))
        if proc.stderr:
            report["diagnostic_log"] = "worker.stderr.log"
        if proc.returncode and report.get("status") == "completed":
            raise RuntimeError("worker exit status inconsistent with report")
        return report


def offline_bm25(golden, k_values):
    """Diagnostic page-level lexical baseline, explicitly NOT production hybrid RAG."""
    import fitz
    import jieba
    import unicodedata
    from rank_bm25 import BM25Okapi
    def tokens(text):
        result = []
        for t in jieba.cut(unicodedata.normalize("NFKC", text)):
            t = t.strip()
            if t and not all(unicodedata.category(c).startswith("P") for c in t):
                result.append(t.casefold())
        return result
    splits = {i.split for i in golden.items}
    docs = []
    for source in golden.manifest:
        if source.get("split", "dev") not in splits:
            continue
        p = PROJECT_ROOT / source["path"]
        with fitz.open(p) as pdf:
            for page, pg in enumerate(pdf, 1):
                text = pg.get_text()
                if text.strip():
                    docs.append({"page_content": text, "metadata": {"filename": p.name, "page": page}})
    model = BM25Okapi([tokens(d["page_content"]) for d in docs])
    def retrieve(question):
        scores = model.get_scores(tokens(question))
        ranked = sorted(range(len(docs)), key=lambda i: (-scores[i], i))
        return [docs[i] for i in ranked[:max(k_values)] if scores[i] > 0]
    report = evaluate_items(golden, mode="retrieval", component=retrieve, k_values=k_values)
    report["retrieval_stage"] = "diagnostic_bm25_pdf_pages"
    report["limitations"] = ["Not production hybrid retrieval; no vector search, reranking or answer generation.",
                             "Evidence labels were agent-authored and source-checked, pending human review."]
    report["api_usage"] = {"calls": 0, "cost_usd": 0}
    return report


def markdown(report):
    lines = ["# RefMind evaluation report", "", f"- Run: {report['run_id']}",
             f"- Mode: {report['mode']}", f"- Status: {report['status']}",
             f"- Time: {report['created_at']}", f"- Dataset SHA256: {report.get('dataset_hash', 'N/A')}"]
    if "summary" in report:
        lines += ["", "## Summary", "", "```json", json.dumps(report["summary"], ensure_ascii=False, indent=2), "```"]
    if report.get("limitations"):
        lines += ["", "## Limitations", ""] + ["- " + s for s in report["limitations"]]
    if "items" in report:
        lines += ["", "## Items", "", "| ID | Status | Recall@10 | MRR | Seconds |", "|---|---|---|---|---|"]
        for row in report["items"]:
            ret = row.get("retrieval") or {}
            lines.append(f"| {row['id']} | {row['status']} | {(ret.get('recall') or {}).get('@10', 'N/A')} | {ret.get('mrr', 'N/A')} | {row.get('latency_s', 'N/A')} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "prepare", "retrieval", "answer", "judge"))
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "evaluation/golden_set.json")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--unlock-holdout", action="store_true")
    parser.add_argument("--backend", choices=("production", "bm25"), default="production")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--input", type=Path, help="Saved answer report; required for judge")
    parser.add_argument("--allow-api", action="store_true", help="Explicitly allow billable model calls")
    parser.add_argument("--no-judge", action="store_true", help="Deprecated alias for retrieval mode")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evaluation/reports")
    args = parser.parse_args(argv)
    mode = args.mode or ("retrieval" if args.no_judge else "validate")
    try:
        if args.no_judge and mode != "retrieval":
            raise ValueError("--no-judge is only compatible with retrieval")
        if any(k < 1 for k in args.k):
            raise ValueError("K values must be positive")
        if args.backend == "bm25" and mode not in {"validate", "retrieval"}:
            raise ValueError("bm25 backend supports retrieval only")
        paid = mode in {"prepare", "answer", "judge"} or (mode == "retrieval" and args.backend == "production")
        if paid and not args.allow_api:
            raise ValueError("this mode may call paid APIs; --allow-api is required")
        if mode != "judge":
            golden = load_golden_set(args.golden, verify_files=True)
            # Validation checks all labels, but never runs a query or exposes holdout questions.
            if mode == "validate":
                report = {"schema_version": 2, "mode": mode, "status": "completed", "dataset_hash": golden.dataset_hash,
                          "summary": {"documents": len(golden.corpus), "items": len(golden.items),
                                      "splits": dict(Counter(i.split for i in golden.items)),
                                      "categories": dict(Counter(i.category for i in golden.items))},
                          "api_usage": {"calls": 0, "cost_usd": 0}}
            else:
                selected = golden.select(args.split, args.unlock_holdout)
        if mode == "retrieval" and args.backend == "bm25":
            report = offline_bm25(selected, tuple(args.k))
        elif paid:
            if mode == "judge" and (args.input is None or not args.input.is_file()):
                raise ValueError("judge requires --input answer_report.json")
            if mode in {"retrieval", "answer"} and args.workspace is None:
                raise ValueError("production evaluation requires an explicitly prepared --workspace")
            path = args.workspace or PROJECT_ROOT / "evaluation/workspaces" / uuid.uuid4().hex
            workspace = _workspace(path, create=mode in {"prepare", "judge"})
            report = run_backend({"mode": mode, "golden": str(args.golden.resolve()), "split": args.split,
                                  "unlock_holdout": args.unlock_holdout, "k_values": args.k,
                                  "input": str(args.input.resolve()) if args.input else None}, workspace)
            report["workspace"] = str(workspace)
        source_files = ("evaluation/dataset.py", "evaluation/evaluator.py", "evaluation/run_eval.py",
                        "evaluation/worker.py", "evaluation/metrics/retrieval.py", "evaluation/metrics/generation.py")
        packages = {}
        for package in ("langchain", "langchain-core", "langchain-openai", "chromadb", "pymupdf", "rank-bm25", "jieba"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        report.update(run_id=uuid.uuid4().hex, created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                      source_hashes={p: hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in source_files},
                      environment={"python": platform.python_version(), "platform": platform.platform(), "packages": packages})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.output_dir / (mode + "_" + report["run_id"])
        stem.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        stem.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
        print(json.dumps({"status": report["status"], "report": str(stem.with_suffix('.json'))}, ensure_ascii=False))
        return 0 if report["status"] == "completed" else 1
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
