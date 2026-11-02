"""Private subprocess backend; its storage is redirected BEFORE application import."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path


def fingerprint(golden, settings):
    config = {name: getattr(settings, name) for name in (
        "embedding_model", "api_base", "chunk_size", "chunk_overlap", "layout_chunk_max_chars",
        "use_fallback_parser", "mineru_backend", "mineru_method")}
    return {"dataset_hash": golden.dataset_hash, "index_config": config}


def run(request):
    from .dataset import load_golden_set
    from .evaluator import evaluate_items, judge_answers
    # The parent only launches this with an evaluation-owned workspace.
    workspace = Path(os.environ["REFMIND_EVAL_WORKSPACE"]).resolve()
    if not (workspace / ".refmind-eval").is_file():
        raise ValueError("evaluation workspace marker missing")
    from refmind.config import settings
    for name in ("database_path", "upload_dir", "parsed_dir", "docstore_dir", "chroma_persist_dir"):
        if not Path(getattr(settings, name)).resolve().is_relative_to(workspace):
            raise ValueError("application storage escaped evaluation workspace")
    mode = request["mode"]
    if mode == "judge":
        from .metrics.generation import evaluate_generation
        source = json.loads(Path(request["input"]).read_text(encoding="utf-8"))
        return judge_answers(source, judge=evaluate_generation)
    golden = load_golden_set(request["golden"], verify_files=True)
    selected = golden.select(request["split"], request.get("unlock_holdout", False))
    snapshot_path = workspace / "snapshot.json"
    expected = fingerprint(golden, settings)
    if mode == "prepare":
        if snapshot_path.exists():
            raise ValueError("workspace already prepared; use a new workspace")
        from refmind import storage
        from refmind.services import ingest_pdf
        storage.init_db()
        group = storage.create_group("evaluation")
        try:
            for source in golden.manifest:
                if source.get("split", "dev") != request["split"] and request["split"] != "all":
                    continue
                p = settings.project_root / source["path"]
                ingest_pdf(group.id, p, p.name, make_summary=False)
            snapshot = {**expected, "group_id": group.id, "split": request["split"]}
            snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"schema_version": 2, "mode": "prepare", "status": "completed", "snapshot": snapshot}
        except Exception:
            from refmind.services import remove_group
            remove_group(group.id)
            raise
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if any(snapshot.get(k) != v for k, v in expected.items()) or snapshot.get("split") != request["split"]:
        raise ValueError("snapshot corpus/config/split mismatch; explicitly prepare a new workspace")
    from refmind import storage
    storage.init_db()
    group_id = snapshot["group_id"]
    if mode == "retrieval":
        from refmind.rag import build_retriever
        retriever = build_retriever(group_id, k=max(request["k_values"]))
        if retriever is None:
            raise ValueError("prepared index has no documents")
        component = retriever.invoke
    else:
        from refmind.rag import answer_question
        from refmind.rag.graph import NO_CONTEXT_REPLY, ACADEMIC_NO_CONTEXT_REPLY
        def component(question):
            result = answer_question(question, group_id)
            result["is_refusal"] = bool(result.get("rejection_stage")) or result.get("answer") in (
                NO_CONTEXT_REPLY, ACADEMIC_NO_CONTEXT_REPLY)
            return result
    report = evaluate_items(selected, mode=mode, component=component, k_values=tuple(request["k_values"]))
    report["index_snapshot"] = snapshot
    report["model_config"] = {name: getattr(settings, name) for name in (
        "llm_model", "embedding_model", "rerank_model", "rerank_enabled", "multi_agent_enabled",
        "recall_top_k", "rerank_top_n", "context_max_chars")}
    return report


def main():
    request = json.load(sys.stdin)
    output = Path(request["result_path"])
    try:
        result = run(request)
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        result = {"schema_version": 2, "mode": request["mode"], "status": "failed",
                  "error": {"type": type(exc).__name__, "message": "worker failed; see local stderr"}}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
