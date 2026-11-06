"""Private subprocess backend; its storage is redirected BEFORE application import."""
from __future__ import annotations
import json
import os
import sys
import hashlib
from pathlib import Path


def fingerprint(golden, settings):
    config = {name: getattr(settings, name) for name in (
        "embedding_model", "api_base", "chunk_size", "chunk_overlap", "layout_chunk_max_chars",
        "use_fallback_parser", "mineru_backend", "mineru_method")}
    config["embedding_provider"] = os.getenv("EMBEDDING_PROVIDER", "dashscope")
    if config["embedding_provider"] == "openai": config["embedding_model"] = "text-embedding-3-small"
    corpus = [{k: source.get(k) for k in ("id", "path", "sha256", "family", "question_split", "split")}
              for source in golden.scoped_manifest()]
    index_files=("refmind/parsing/pdf_parser.py","refmind/rag/layout_chunker.py","refmind/rag/document_processor.py")
    implementation={path:hashlib.sha256((settings.project_root/path).read_bytes()).hexdigest() for path in index_files}
    return {"dataset_hash": golden.dataset_hash, "index_config": config,"index_implementation":implementation,
            "corpus_scope_hash": hashlib.sha256(json.dumps(corpus, sort_keys=True).encode()).hexdigest()}


def configure_text_only(graph_module):
    if os.getenv("REFMIND_EVAL_TEXT_ONLY") == "true":
        from refmind.llm import image_understanding

        def no_visual_content(_documents):
            return []

        # Generation uses an imported alias. Reviewers resolve the module attribute at call time.
        graph_module.build_visual_content = no_visual_content
        image_understanding.build_visual_content = no_visual_content


def dialogue_component(group_id,source_doc_ids,*,pipeline="legacy"):
    """One isolated SQLite session per scenario; input actions are separate from gold checks."""
    from refmind import storage
    from refmind.rag import answer_question
    from refmind.rag.memory import RelevantMemory
    from langchain_core.messages import HumanMessage,AIMessage
    memories={}
    histories={}
    reverse={int(value):key for key,value in source_doc_ids.items()}
    def component(request,state):
        scenario=request["scenario_id"]
        if scenario not in memories:
            session=storage.create_session(group_id,"evaluation:"+scenario)
            memories[scenario]=RelevantMemory(session.id,group_id=group_id,user_id="evaluation:"+scenario)
            histories[scenario]=[]
        actions=request.get("input") or {}
        if pipeline=="research":
            selected,selected_explicit=None,False
            if "selected_source_ids" in actions:
                selected,selected_explicit=actions["selected_source_ids"],True
            elif state.get("selected_source_ids_explicit") is True:
                selected,selected_explicit=state.get("selected_source_ids"),True
        else:
            selected=actions.get("selected_source_ids",state.get("selected_source_ids",request["corpus_scope"]))
            selected_explicit=True
        if (selected_explicit and not isinstance(selected,list)) or (selected is not None and not set(selected)<=set(request["corpus_scope"])):
            raise ValueError("dialogue action selected a source outside scenario corpus_scope")
        mode=actions.get("retrieval_mode",state.get("retrieval_mode","library"))
        if mode!="library":
            raise ValueError("local dialogue worker cannot silently enable external discovery")
        memory=memories[scenario]
        if pipeline=="research":
            from refmind.rag.graph import answer_research_question
            scope_kwargs={"allowed_doc_ids":[source_doc_ids[sid] for sid in request["corpus_scope"]]}
            if selected_explicit:
                scope_kwargs["selected_doc_ids"]=[source_doc_ids[sid] for sid in selected]
            result=answer_research_question(request["question"],group_id,
                **scope_kwargs,history=histories[scenario],user_id="evaluation:"+scenario)
        else:
            result=answer_question(request["question"],group_id,memory,
                                  selected_doc_ids=[source_doc_ids[sid] for sid in selected])
        histories[scenario].extend([HumanMessage(content=request["question"]),
                                    AIMessage(content=result.get("answer",""))])
        memory.add_user_message(request["question"])
        memory.add_ai_message(result.get("answer",""))
        docs=[]
        for document in result.get("documents",[]):
            metadata=dict(document.metadata or {})
            try:
                metadata["source_id"]=reverse.get(int(metadata.get("doc_id")))
            except (TypeError,ValueError):
                pass
            docs.append({"page_content":document.page_content,"metadata":metadata})
        updated_state={"selected_source_ids":selected,"retrieval_mode":mode,
                       "session_id":memory.session_id,"turn_count":state.get("turn_count",0)+1}
        if pipeline=="research":
            updated_state["selected_source_ids_explicit"]=selected_explicit
        return {**result,"documents":docs,"state":updated_state}
    return component


def run(request):
    from .dataset import load_golden_set, verify_corpus
    from .evaluator import evaluate_items, judge_answers
    # The parent only launches this with an evaluation-owned workspace.
    workspace = Path(os.environ["REFMIND_EVAL_WORKSPACE"]).resolve()
    if not (workspace / ".refmind-eval").is_file():
        raise ValueError("evaluation workspace marker missing")
    from refmind.config import settings
    from .runtime import effective_overrides
    effective=effective_overrides(settings,request.get("runtime_overrides",{}))
    for name in ("database_path", "upload_dir", "parsed_dir", "docstore_dir", "chroma_persist_dir"):
        if not Path(getattr(settings, name)).resolve().is_relative_to(workspace):
            raise ValueError("application storage escaped evaluation workspace")
    mode = request["mode"]
    pipeline=request.get("pipeline","legacy")
    if pipeline not in {"legacy","research"}:
        raise ValueError("unknown evaluation pipeline")
    if mode == "judge":
        os.environ["OPENAI_CALL_ROLE"] = "judge"
        from .metrics.generation import evaluate_generation
        source = json.loads(Path(request["input"]).read_text(encoding="utf-8"))
        saved_gate=source.get("review_gate", {})
        if request.get("independent_acceptance") and not saved_gate.get("independent_eligible"):
            raise ValueError("saved answer report is not eligible for independent acceptance")
        if not saved_gate.get("official") and not (saved_gate.get("human_review_complete") and saved_gate.get("label")=="historical_regression"):
            if not request.get("allow_draft") or any(row.get("question_split", row.get("split", "dev")) != "dev" for row in source.get("items", [])):
                raise ValueError("unreviewed saved answers require explicit dev-only draft diagnostics")
        return judge_answers(source, judge=evaluate_generation)
    golden = load_golden_set(request["golden"])
    selected = golden.select(request["split"], request.get("unlock_holdout", False))
    review_gate = selected.require_review(allow_draft=request.get("allow_draft", False),
                                         independent_acceptance=request.get("independent_acceptance",False))
    verify_corpus(selected, settings.project_root, scoped=True)
    allowed_track="dialogue" if mode=="dialogue" or mode=="prepare" and all(i.track=="dialogue" for i in selected.items) else "local"
    if any(item.track != allowed_track for item in selected.items):
        raise ValueError("this worker is for local RAG, not external/dialogue/structure tasks")
    snapshot_path = workspace / "snapshot.json"
    expected = fingerprint(selected, settings)
    split_names = sorted({item.question_split for item in selected.items})
    if mode == "prepare":
        if snapshot_path.exists():
            raise ValueError("workspace already prepared; use a new workspace")
        from refmind import storage
        from refmind.services import ingest_pdf
        storage.init_db()
        group = storage.create_group("evaluation")
        source_doc_ids = {}
        try:
            for source in selected.scoped_manifest():
                p = settings.project_root / source["path"]
                result = ingest_pdf(group.id, p, p.name, make_summary=False)
                source_doc_ids[source.get("id", source["path"])] = result.id
            snapshot = {**expected, "group_id": group.id, "question_splits": split_names,
                        "source_doc_ids": source_doc_ids}
            snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"schema_version": golden.schema_version, "mode": "prepare", "status": "completed",
                    "review_gate": review_gate, "effective_overrides":effective,"snapshot": snapshot}
        except Exception:
            from refmind.services import remove_group
            remove_group(group.id)
            raise
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if any(snapshot.get(k) != v for k, v in expected.items()) or snapshot.get("question_splits") != split_names:
        raise ValueError("snapshot corpus/config/split mismatch; explicitly prepare a new workspace")
    from refmind import storage
    storage.init_db()
    group_id = snapshot["group_id"]
    if mode=="dialogue":
        from .suites import evaluate_dialogues
        from refmind.rag import graph as graph_module
        configure_text_only(graph_module)
        report=evaluate_dialogues(selected,dialogue_component(group_id,snapshot["source_doc_ids"],pipeline=pipeline),
            allow_draft=request.get("allow_draft",False),execution="production_isolated_dialogue")
        report["index_snapshot"]=snapshot
        report["pipeline"]=pipeline
        report["effective_overrides"]=effective
        return report
    def document_scope(item):
        if golden.schema_version < 3:
            return None
        mapping = snapshot.get("source_doc_ids", {})
        if not set(item.corpus_scope) <= set(mapping):
            raise ValueError("prepared workspace is missing a question's explicit corpus scope")
        return tuple(mapping[sid] for sid in item.corpus_scope)
    if mode == "retrieval":
        from refmind.rag import build_retriever
        from refmind.agents.orchestration import canonicalize_retrieval_query
        retrievers = {}
        def component(item):
            scope = document_scope(item)
            if pipeline=="research":
                from refmind.rag.graph import retrieve_research_evidence
                return retrieve_research_evidence(item.question,group_id,
                    allowed_doc_ids=list(scope) if scope is not None else None)
            if scope not in retrievers:
                kwargs = {"selected_doc_ids": scope} if scope is not None else {}
                retrievers[scope] = build_retriever(group_id,
                    k=max(max(request["k_values"]), settings.recall_top_k), **kwargs)
            retriever = retrievers[scope]
            return retriever.invoke(canonicalize_retrieval_query(item.question)) if retriever is not None else []
    else:
        from refmind.rag import answer_question
        from refmind.rag import graph as graph_module
        configure_text_only(graph_module)
        from refmind.rag.graph import NO_CONTEXT_REPLY, ACADEMIC_NO_CONTEXT_REPLY
        def component(item):
            scope = document_scope(item)
            scope_argument="allowed_doc_ids" if pipeline=="research" else "selected_doc_ids"
            kwargs = {scope_argument: list(scope)} if scope is not None else {}
            if pipeline=="research":
                from refmind.rag.graph import answer_research_question
                result=answer_research_question(item.question,group_id,**kwargs)
            else:
                result = answer_question(item.question, group_id, **kwargs)
            from .classification import classify_refusal
            result["is_refusal"] = (result["status"]=="insufficient") if result.get("status") in {"answered","scoped","insufficient"} else True if result.get("rejection_stage") else classify_refusal(
                result.get("answer"), (NO_CONTEXT_REPLY, ACADEMIC_NO_CONTEXT_REPLY))
            return result
    report = evaluate_items(selected, mode=mode, component=component, k_values=tuple(request["k_values"]),
                            allow_draft=request.get("allow_draft", False), component_accepts_item=True)
    report["review_gate"] = review_gate
    report["pipeline"]=pipeline
    report["effective_overrides"]=effective
    if mode=="retrieval" and pipeline=="research":
        report["retrieval_stage"]="research_reranked_candidates_with_raw_stage_trace"
    report["index_snapshot"] = snapshot
    report["model_config"] = {name: getattr(settings, name) for name in (
        "llm_model", "embedding_model", "rerank_model", "rerank_enabled", "multi_agent_enabled",
        "recall_top_k", "retrieval_top_k", "rerank_top_n", "context_max_chars")}
    if os.getenv("CHAT_PROVIDER") == "openai": report["model_config"]["llm_model"] = "gpt-5.4-mini-2026-03-17"
    if os.getenv("EMBEDDING_PROVIDER") == "openai": report["model_config"]["embedding_model"] = "text-embedding-3-small"
    report["model_config"]["text_only"] = os.getenv("REFMIND_EVAL_TEXT_ONLY") == "true"
    report["model_config"]["chat_provider"] = os.getenv("CHAT_PROVIDER", "dashscope")
    report["model_config"]["embedding_provider"] = os.getenv("EMBEDDING_PROVIDER", "dashscope")
    for name in ("context_compression_enabled","context_max_tokens","model_context_tokens","answer_token_reserve","refusal_confirmation_enabled","context_body_priority_enabled"):
        report["model_config"][name]=getattr(settings,name,None)
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
