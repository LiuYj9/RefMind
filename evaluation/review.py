"""Export review packets and apply explicit human decisions with immutable audit files."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

from .construction import CORE_CATEGORIES
from .dataset import load_golden_set


def apply_decisions(dataset_path,decisions_path,output,*,project_root):
    if dataset_path.resolve()==output.resolve() or output.exists():
        raise ValueError("review creates a new dataset version, never overwrites the input or an existing version")
    raw_bytes=dataset_path.read_bytes()
    decisions_bytes=decisions_path.read_bytes()
    raw=json.loads(raw_bytes.decode("utf-8"))
    decisions=json.loads(decisions_bytes.decode("utf-8"))
    if raw.get("schema_version")!=3 or decisions.get("dataset_sha256")!=hashlib.sha256(raw_bytes).hexdigest():
        raise ValueError("human decisions must identify the exact v3 dataset hash")
    reviewer,reviewed_at=decisions.get("reviewer"),decisions.get("reviewed_at")
    if not isinstance(reviewer,str) or not reviewer.strip() or not reviewed_at:
        raise ValueError("named human reviewer and review date are required")
    changes=decisions.get("decisions")
    if not isinstance(changes,list) or not changes:
        raise ValueError("explicit per-item decisions required")
    items={item["id"]:item for item in raw["items"]}
    seen=set(); audit=[]
    for decision in changes:
        iid=decision.get("id")
        if iid not in items or iid in seen or decision.get("status") not in {"approved","rejected","pending"}:
            raise ValueError("invalid, unknown or duplicate item decision")
        seen.add(iid); item=items[iid]; before=copy.deepcopy(item)
        patches=decision.get("patches",{})
        if not isinstance(patches,dict) or not set(patches)<={"question","ground_truth","category","evidence","fact_slots","negative_reason","external_gold","turns","structure_gold"}:
            raise ValueError("human review cannot silently move question/source partitions or identities")
        item.update(patches)
        if decision["status"]=="approved":
            if item.get("track","local")=="local" and (item.get("category") not in CORE_CATEGORIES or
                    not item.get("expect_refusal") and not item.get("fact_slots")):
                raise ValueError("local approval requires a final category and fact slots for answerable items")
            item["annotation_status"]="human_approved"
        else:
            item["annotation_status"]="human_rejected" if decision["status"]=="rejected" else "draft_pending_human_review"
        item["human_review"]={"status":decision["status"],"reviewer":reviewer,"reviewed_at":reviewed_at,
                              "comment":str(decision.get("comment",""))}
        audit.append({"id":iid,"before_sha256":hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
                      "after_sha256":hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest(),
                      "status":decision["status"],"patched_fields":sorted(patches)})
    encoded=(json.dumps(raw,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
    # Validate before writing the durable reviewed version.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="refmind-review-") as tmp:
        candidate=Path(tmp)/"candidate.json";candidate.write_bytes(encoded)
        load_golden_set(candidate,project_root=project_root)
    output.parent.mkdir(parents=True,exist_ok=True)
    audit_path=output.with_suffix(output.suffix+".review-audit.json")
    if audit_path.exists():
        raise ValueError("review audit already exists")
    audit_data={"source_dataset_sha256":hashlib.sha256(raw_bytes).hexdigest(),
                "decision_file_sha256":hashlib.sha256(decisions_bytes).hexdigest(),
                "output_dataset_sha256":hashlib.sha256(encoded).hexdigest(),
                "reviewer":reviewer,"reviewed_at":reviewed_at,"items":audit}
    output.write_bytes(encoded)
    audit_path.write_text(json.dumps(audit_data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"output":str(output),"audit":str(audit_path),"decisions":len(audit)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("export","apply"))
    parser.add_argument("--golden",type=Path,required=True)
    parser.add_argument("--decisions",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--split",choices=("dev","id_holdout","ood_holdout","all"),default="dev")
    parser.add_argument("--unlock-holdout",action="store_true")
    parser.add_argument("--assigned-reviewer",help="Assignment only; does not assert completion or domain expertise")
    parser.add_argument("--task-table",type=Path,help="Optional UTF-8 CSV checklist; decisions are imported from JSON")
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parents[1]
    if args.action=="apply":
        if args.decisions is None:
            parser.error("apply requires --decisions")
        result=apply_decisions(args.golden,args.decisions,args.output,project_root=root)
    else:
        golden=load_golden_set(args.golden).select(args.split,args.unlock_holdout)
        raw=json.loads(args.golden.read_text(encoding="utf-8"))
        ids={item.id for item in golden.items}
        selected=[item for item in raw["items"] if item["id"] in ids]
        result={"dataset_sha256":golden.dataset_hash,"assigned_reviewer":args.assigned_reviewer,
                "reviewer":args.assigned_reviewer,"reviewed_at":None,
                "independence_audit":golden.independence_audit,
                "instructions":"Assignment is not approval or domain expertise. Inspect the original source, question, answer, evidence, units/conditions, category, split and fact slots. Fill reviewed_at and explicit per-item status/comment/patches. Keep unreviewed items pending. Import this JSON with evaluation.review apply; CSV is a checklist only.",
                "sources":raw.get("corpus",raw.get("sources",[])),"items":selected,
                "decisions":[{"id":item["id"],"status":"pending","comment":"","patches":{}} for item in selected]}
        if args.output.exists():
            raise ValueError("review packet already exists")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        if args.task_table:
            if args.task_table.exists():
                raise ValueError("review checklist already exists")
            args.task_table.parent.mkdir(parents=True,exist_ok=True)
            with args.task_table.open("w",encoding="utf-8-sig",newline="") as stream:
                columns=["id","assigned_reviewer","track","domain","question_split","question","status","source_verified","answer_verified","evidence_verified","units_conditions_verified","category_verified","fact_slots_verified","comment"]
                writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
                for item in selected:
                    writer.writerow({"id":item["id"],"assigned_reviewer":args.assigned_reviewer or "",
                                     "track":item.get("track","local"),"domain":item.get("domain",""),
                                     "question_split":item.get("question_split",""),"question":item["question"],"status":"pending"})
        result={"packet":str(args.output),"task_table":str(args.task_table) if args.task_table else None,"assigned_reviewer":args.assigned_reviewer,"items":len(ids),"approved":0}
    print(json.dumps(result,ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
