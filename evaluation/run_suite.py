"""Run frozen external, four-turn dialogue or structural evaluation without model APIs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid
import copy

from .dataset import load_golden_set
from .suites import evaluate_external,evaluate_dialogues,evaluate_structure
from .runtime import parse_overrides
from .fingerprints import implementation_hashes


def load_replay(path, golden, track):
    raw = path.read_bytes()
    replay = json.loads(raw.decode("utf-8"))
    if replay.get("track") != track or replay.get("dataset_sha256") != golden.dataset_hash:
        raise ValueError("replay track/dataset hash mismatch")
    cases = replay.get("cases")
    if not isinstance(cases,dict):
        raise ValueError("replay cases must be keyed by item ID")
    if not {item.id for item in golden.items} <= set(cases):
        raise ValueError("replay is missing selected cases; never synthesize a passing result")
    return replay,hashlib.sha256(raw).hexdigest()


def provider_replay_component(cases):
    """Exercise the production provider parsers/filter/merge with frozen HTTP payloads."""
    def component(request):
        from unittest.mock import patch
        from refmind.services import paper_discovery as discovery
        from urllib.parse import urlparse
        case=cases[request["id"]]
        payloads=case["provider_payloads"]
        def fetch(url,**kwargs):
            host=urlparse(url).hostname or ""
            provider={"api.crossref.org":"crossref","api.semanticscholar.org":"semantic_scholar",
                      "api.openalex.org":"openalex"}.get(host)
            if provider not in payloads:
                raise ValueError("frozen replay has no response for this provider")
            payload=payloads[provider]
            if isinstance(payload,dict) and payload.get("replay_error"):
                raise RuntimeError("recorded provider failure")
            return copy.deepcopy(payload)
        parameters=dict(case.get("parameters",{}))
        filters=discovery.SearchFilters(**parameters.pop("filters",{}))
        parameters.setdefault("provider","crossref")
        with patch.object(discovery,"_request_json",side_effect=fetch), \
             patch.object(discovery.settings,"academic_search_enabled",True), \
             patch.object(discovery.settings,"openalex_api_key","replay" if "openalex" in payloads else ""), \
             patch.object(discovery.settings,"semantic_scholar_api_key",""), \
             patch.object(discovery.settings,"crossref_mailto",""):
            result=discovery.discover_papers(request["query"],filters=filters,**parameters).to_dict()
        downstream=copy.deepcopy(case.get("recorded_downstream",{}))
        return {**downstream,**result,"status":"discovery_failed" if result["failed"] else "discovered",
                "trace":{"discovery":{"execution":"production_provider_adapter_with_frozen_payloads",
                           "providers":result.get("providers"),"provider_status":result.get("provider_status")},
                         "downstream_execution":"recorded_only" if downstream else "not_run"}}
    return component


def parser_component(golden,project_root,parser="pymupdf"):
    cache={}
    sources={source["id"]:source for source in golden.scoped_manifest()}
    def component(request):
        import fitz
        from unittest.mock import patch
        from refmind.parsing import pdf_parser
        sid=request["source_id"]
        if sid not in cache:
            path=project_root/sources[sid]["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sources[sid]["sha256"]:
                raise ValueError("structure source bytes changed")
            with patch.object(pdf_parser.settings,"use_fallback_parser",parser=="pymupdf"):
                cache[sid]=pdf_parser.parse_pdf(path)
        parsed=cache[sid]
        objects=[{**block,"bbox_coordinate_system":block.get("bbox_coordinate_system",
                         "normalized_1000" if parsed.get("parser")=="mineru" else "pdf_points")}
                 for block in parsed.get("blocks",[]) if block.get("page")==request["page"]]
        return {"objects":objects,"trace":{"parser":parsed.get("parser"),"requested_parser":parser}}
    return component


def run(golden,replay,track,*,allow_draft=False,backend="recorded",project_root=None,parser="pymupdf"):
    if backend=="parser":
        if track!="structure":
            raise ValueError("parser backend is only valid for structure track")
        return evaluate_structure(golden,parser_component(golden,project_root,parser),
                                  allow_draft=allow_draft,execution="production_parser")
    cases=replay["cases"]
    if backend=="provider_replay":
        if track!="external":
            raise ValueError("provider replay is only valid for external track")
        return evaluate_external(golden,provider_replay_component(cases),allow_draft=allow_draft,
                                 execution="production_discovery_frozen_http_recorded_downstream")
    if track=="dialogue":
        def component(request,state):
            return cases[request["scenario_id"]]["actual_turns"][request["turn_index"]]
        return evaluate_dialogues(golden,component,allow_draft=allow_draft,execution="recorded_replay")
    component=lambda request:cases[request["id"]]["actual"]
    function={"external":evaluate_external,"structure":evaluate_structure}[track]
    return function(golden,component,allow_draft=allow_draft,execution="recorded_replay")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track",choices=("external","dialogue","structure"),required=True)
    parser.add_argument("--golden",type=Path,required=True)
    parser.add_argument("--replay",type=Path)
    parser.add_argument("--backend",choices=("recorded","provider_replay","parser","production"),default="recorded")
    parser.add_argument("--parser",choices=("pymupdf","mineru"),default="pymupdf")
    parser.add_argument("--allow-api",action="store_true")
    parser.add_argument("--workspace",type=Path)
    parser.add_argument("--pipeline",choices=("legacy","research"),default="research")
    parser.add_argument("--set",dest="runtime_overrides",action="append",default=[],metavar="KEY=VALUE")
    parser.add_argument("--split",choices=("dev","id_holdout","ood_holdout","all"),default="dev")
    parser.add_argument("--allow-draft",action="store_true")
    parser.add_argument("--unlock-holdout",action="store_true")
    parser.add_argument("--independent-acceptance",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"reports")
    args=parser.parse_args(argv)
    overrides=parse_overrides(args.runtime_overrides)
    if overrides and args.backend!="production":
        parser.error("runtime RAG overrides apply only to isolated production dialogue")
    if args.backend=="production" and (not args.allow_api or args.workspace is None or args.track!="dialogue"):
        parser.error("production dialogue requires --allow-api and an explicitly prepared --workspace")
    golden=load_golden_set(args.golden).select(args.split,args.unlock_holdout)
    gate=golden.require_review(allow_draft=args.allow_draft,independent_acceptance=args.independent_acceptance)
    if args.backend not in {"parser","production"} and args.replay is None:
        parser.error("recorded/provider replay requires --replay")
    replay,digest=load_replay(args.replay,golden,args.track) if args.replay else ({},None)
    if args.backend=="production":
        from .run_eval import run_backend,_workspace
        workspace=_workspace(args.workspace,create=False)
        report=run_backend({"mode":"dialogue","golden":str(args.golden.resolve()),"split":args.split,
            "unlock_holdout":args.unlock_holdout,"allow_draft":args.allow_draft,"independent_acceptance":args.independent_acceptance,"k_values":[1,3,5,10],"pipeline":args.pipeline,
            "runtime_overrides":overrides},workspace)
        report.setdefault("execution","production_isolated_dialogue")
    else:
        report=run(golden,replay,args.track,allow_draft=args.allow_draft,backend=args.backend,
                   project_root=Path(__file__).resolve().parents[1],parser=args.parser)
    report.update(run_id=uuid.uuid4().hex,created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  replay_sha256=digest,network_calls=0 if args.backend not in {"production","parser"} or args.backend=="parser" and args.parser=="pymupdf" else None,
                  model_api_calls=None if args.backend=="production" else 0)
    report["review_gate"]=gate
    report["source_hashes"]=implementation_hashes()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/(args.track+"_"+report["run_id"]+".json")
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"execution":report["execution"],"report":str(path)}))
    return 0 if report["status"]=="completed" else 1


if __name__=="__main__":
    raise SystemExit(main())
