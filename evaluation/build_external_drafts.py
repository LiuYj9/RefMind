"""Build actual development discovery/read tasks from already pinned public papers.

Boundary judgments stay unknown until human review. This creates task drafts, not
imaginary live search results or 120 completed external gold annotations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from .dataset import load_golden_set
from .public_sources import fetch


def build(root,max_families=24):
    paths=[root/"evaluation/datasets/public_materials_hts_dev_v3.json",
           root/"evaluation/datasets/public_qasper_dev_v3_160.json"]
    sources={};first_questions={}
    for path in paths:
        data=json.loads(path.read_text(encoding="utf-8"))
        for source in data["corpus"]:
            sources[source["id"]]=source
        for item in data["items"]:
            sid=item["corpus_scope"][0]
            first_questions.setdefault(sid,item)
    selected=[sid for sid in sources if sid in first_questions][:max_families]
    items=[]
    for sid in selected:
        source=sources[sid];original=first_questions[sid]
        family=source["family"];work=family.casefold()
        aliases={}
        if work.startswith("arxiv:"):
            aliases["10.48550/arxiv."+work.split(":",1)[1]]=work
        if sid=="public_llzo":
            aliases["10.1103/physrevlett.109.205702"]=work
        for kind in ("lookup","abstract_boundary","fulltext_read"):
            if kind=="lookup":
                question=f"精确定位论文《{source['title']}》，核对文献身份{family}并返回其公开来源，不把题录称为已读全文。"
                truth=f"目标论文身份为{family}，题名为{source['title']}；公开版本链接{source['source_url']}。"
                evidence=[]
            elif kind=="abstract_boundary":
                question=f"检索《{source['title']}》后，仅凭摘要能否回答以下问题？请明确证据层级，不足时指出需要全文核对：{original['question']}"
                truth="摘要支撑程度待独立人审；全文参考答案为："+original["ground_truth"]
                evidence=[]
            else:
                question=f"读取《{source['title']}》的公开全文版本{source['version']}，基于原文页证据回答：{original['question']}"
                truth=original["ground_truth"]
                evidence=copy.deepcopy(original["evidence"])
            gold={"task_type":kind,"required_work_ids":[work],"qrels":{work:2},"aliases":aliases,
                  "fulltext_sha256":source["sha256"],"fulltext_source_url":source["source_url"],
                  "version":source["version"],"abstract_support":"pending_human_review",
                  "qrels_scope":"known_target_work_only_not_exhaustive_web_relevance"}
            if kind=="lookup":
                gold["expected_status"]="discovered"
            items.append({"id":"PUBLIC-EXTERNAL-"+sid+"-"+kind,"track":"external","question":question,"ground_truth":truth,
                "domain":source["domain"],"question_split":"dev","family_ids":[family],"corpus_scope":[sid],
                "category":kind,"expect_refusal":False,"annotation_status":"source_checked_draft","human_review":{"status":"pending"},
                "evidence":evidence,"fact_slots":copy.deepcopy(original.get("fact_slots",[])) if kind=="fulltext_read" else [],
                "external_gold":gold,"task_metadata":{"source_local_question_id":original["id"],
                  "provenance":"Real pinned public PDF and source question; no live result or human support judgment fabricated."}})
    result={"schema_version":3,"corpus":[sources[sid] for sid in selected],"items":items}
    path=root/"evaluation/datasets/public_external_dev_v3.json"
    if path.exists():
        raise ValueError("external task drafts already exist")
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    load_golden_set(path,project_root=root,verify_files=True)
    return result


def capture_lookup_replay(root,dataset):
    lookup=[item for item in dataset["items"] if item["category"]=="lookup" and item["corpus_scope"][0] in {"public_llzo","public_robotic_hts"}]
    selected={sid for item in lookup for sid in item["corpus_scope"]}
    subset={"schema_version":3,"corpus":[s for s in dataset["corpus"] if s["id"] in selected],"items":lookup}
    path=root/"evaluation/workspaces/public-source-v3/external_lookup_subset.json"
    path.write_text(json.dumps(subset,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    cases={}
    for item in lookup:
        doi="10.1103/PhysRevLett.109.205702" if item["corpus_scope"]==["public_llzo"] else "10.1088/1361-6668/ae2362"
        url="https://api.crossref.org/works/"+quote(doi,safe="")
        body=fetch(url,root/("evaluation/workspaces/public-source-v3/crossref_"+hashlib.sha256(url.encode()).hexdigest()[:16]+".json"))
        cases[item["id"]]={"parameters":{"doi":doi,"provider":"crossref"},"provider_payloads":{"crossref":json.loads(body)},
            "capture":{"url":url,"response_sha256":hashlib.sha256(body).hexdigest(),"kind":"actual_public_crossref_doi_response"}}
    replay={"track":"external","dataset_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"cases":cases}
    replay_path=path.with_name("external_lookup_replay.json")
    replay_path.write_text(json.dumps(replay,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"subset":str(path),"replay":str(replay_path),"cases":len(cases)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-families",type=int,default=24)
    parser.add_argument("--capture-lookup",action="store_true")
    args=parser.parse_args(argv);root=args.root.resolve()
    data=build(root,args.max_families)
    result={"external_dev_drafts":len(data["items"]),"families":len(data["corpus"]),"human_approved":0,
            "boundary_judgments":"pending_not_synthetic_labels"}
    if args.capture_lookup:
        result["replay"]=capture_lookup_replay(root,data)
    print(json.dumps(result))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
