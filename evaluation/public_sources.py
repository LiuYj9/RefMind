"""Explicit, bounded public-source collection and licensed QASPER draft import.

Downloads require --download. No model APIs, pickle loading, archive extraction,
holdout QA import or changes to daily user libraries are performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time
import unicodedata
from urllib.parse import urlencode,urlparse
from urllib.request import Request,urlopen,HTTPRedirectHandler,build_opener

QASPER_URL="https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
QASPER_LICENSE="https://creativecommons.org/licenses/by/4.0/"


class PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        check_url(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def check_url(url):
    host=urlparse(url).hostname or ""
    allowed=(host in {"api.crossref.org","qasper-dataset.s3.us-west-2.amazonaws.com","arxiv.org","export.arxiv.org","publikationen.bibliothek.kit.edu"}
             or host.endswith(".arxiv.org"))
    if not allowed or urlparse(url).scheme!="https" or urlparse(url).username:
        raise ValueError("public collection only requests fixed HTTPS scholarly source hosts")


def fetch(url,cache,*,limit=40_000_000):
    check_url(url)
    if cache.exists():
        return cache.read_bytes()
    opener=build_opener(PublicRedirects())
    last=None
    for attempt in range(3):
        try:
            with opener.open(Request(url,headers={"User-Agent":"RefMind-public-evaluation/3.0","Accept":"*/*"}),timeout=45) as response:
                body=response.read(limit+1)
            if len(body)>limit:
                raise ValueError("public source exceeds collection size bound")
            cache.parent.mkdir(parents=True,exist_ok=True)
            cache.write_bytes(body)
            return body
        except Exception as exc:
            last=exc
            if attempt<2:
                time.sleep(1+attempt)
    raise RuntimeError(f"public fetch failed: {type(last).__name__}") from last


def _normalize(text):
    return re.sub(r"\s+","",unicodedata.normalize("NFKC",str(text))).casefold()


def _map_page(text):
    normalized=[];positions=[]
    for index,char in enumerate(text):
        for piece in unicodedata.normalize("NFKC",char).casefold():
            if not piece.isspace():
                normalized.append(piece);positions.append(index)
    return "".join(normalized),positions


def locate_quote(paragraph,page_texts):
    """Exact normalized substrings only; return actual PDF text, never inferred OCR."""
    normalized=_normalize(paragraph)
    if len(normalized)<40:
        return None
    for length in dict.fromkeys(min(n,len(normalized)) for n in (160,80,40)):
        starts=list(range(0,min(len(normalized)-length+1,3000),max(20,length//2)))
        for page_number,text in enumerate(page_texts,1):
            page,positions=_map_page(text)
            for start in starts:
                needle=normalized[start:start+length]
                where=page.find(needle)
                if where>=0:
                    left,right=positions[where],positions[where+len(needle)-1]+1
                    raw=text[max(0,left-100):min(len(text),right+150)]
                    return page_number,raw
    return None


def collect_candidates(root,cache,qasper,*,download,dev_papers=12):
    selected=sorted(qasper)[:dev_papers+4]
    candidates=[]
    for index,arxiv_id in enumerate(selected):
        paper=qasper[arxiv_id]
        candidates.append({"candidate_id":"arxiv:"+arxiv_id,"family_candidate":"arxiv:"+arxiv_id,
            "domain":"computer_science_ml","proposed_question_split":"dev" if index<dev_papers else "id_holdout",
            "title":paper["title"],"source_url":"https://arxiv.org/abs/"+arxiv_id,
            "public_pdf_url":"https://arxiv.org/pdf/"+arxiv_id,"version":"not_yet_pinned",
            "qa_dataset":"QASPER v0.3 train","qa_license":QASPER_LICENSE,
            "pdf_license":"check_individual_arxiv_record","annotation_status":"metadata_checked_candidate_pending_family_review"})
    domains=[("superconductivity_physics","high temperature superconducting quench coils",6,5),
             ("materials_energy","battery electrode materials energy",16,12),
             ("mechanical_thermofluids","heat transfer fluid flow mechanical",6,0),
             ("chemical_process","chemical process reaction engineering",6,0)]
    seen=set()
    for domain,query,count,dev_count in domains:
        url="https://api.crossref.org/works?"+urlencode({"query":query,"rows":100,
            "filter":"type:journal-article,has-license:true,has-full-text:true"})
        payload=fetch(url,cache/(domain+"_"+hashlib.sha256(url.encode()).hexdigest()[:12]+"_crossref.json"))
        entries=json.loads(payload)["message"]["items"]
        chosen=[]
        for entry in entries:
            doi=str(entry.get("DOI","")).casefold()
            licenses=[record.get("URL","") for record in entry.get("license",[]) if "creativecommons.org/" in record.get("URL","")]
            if not doi or doi in seen or not licenses or not entry.get("title"):
                continue
            seen.add(doi)
            links=[link["URL"] for link in entry.get("link",[]) if link.get("content-type")=="application/pdf" and link.get("URL","").startswith("https://")]
            position=len(chosen)
            partition=("dev" if position<dev_count else "id_holdout") if dev_count else "ood_holdout"
            chosen.append({"candidate_id":"doi:"+doi,"family_candidate":"doi:"+doi,"doi":doi,
                "title":entry["title"][0],"domain":domain,"proposed_question_split":partition,
                "source_url":entry.get("URL","https://doi.org/"+doi),"public_pdf_urls":links,
                "license_urls":licenses,"version":"publisher_metadata_version_unverified",
                "record_sha256":hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest(),
                "annotation_status":"metadata_checked_candidate_pending_domain_family_review"})
            if len(chosen)==count:
                break
        if len(chosen)<count:
            extra_url="https://api.crossref.org/works?"+urlencode({"query":query,"rows":100,
                "filter":"type:journal-article,license.url:https://creativecommons.org/licenses/by/4.0/"})
            extra_payload=fetch(extra_url,cache/(domain+"_"+hashlib.sha256(extra_url.encode()).hexdigest()[:12]+"_crossref.json"))
            for entry in json.loads(extra_payload)["message"]["items"]:
                doi=str(entry.get("DOI","")).casefold()
                if not doi or doi in seen or not entry.get("title"):
                    continue
                seen.add(doi);position=len(chosen)
                licenses=[record.get("URL","") for record in entry.get("license",[]) if "creativecommons.org/" in record.get("URL","")]
                if not licenses:
                    continue
                chosen.append({"candidate_id":"doi:"+doi,"family_candidate":"doi:"+doi,"doi":doi,
                    "title":entry["title"][0],"domain":domain,
                    "proposed_question_split":("dev" if position<dev_count else "id_holdout") if dev_count else "ood_holdout",
                    "source_url":entry.get("URL","https://doi.org/"+doi),"public_pdf_urls":[],"license_urls":licenses,
                    "version":"publisher_metadata_version_unverified",
                    "record_sha256":hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest(),
                    "annotation_status":"metadata_checked_candidate_pending_domain_family_review"})
                if len(chosen)==count:
                    break
        candidates.extend(chosen)
    return candidates


def import_qasper(root,cache,qasper,candidates,*,max_papers=12,max_questions=None):
    import fitz
    corpus=[];items=[];failures=[]
    for candidate in [c for c in candidates if c["domain"]=="computer_science_ml" and c["proposed_question_split"]=="dev"][:max_papers]:
        arxiv_id=candidate["candidate_id"].split(":",1)[1]
        paper=qasper[arxiv_id]
        path=cache/"pdfs"/(arxiv_id+".pdf")
        try:
            body=fetch(candidate["public_pdf_url"],path)
            if not body.lstrip().startswith(b"%PDF-"):
                raise ValueError("source is not PDF")
            with fitz.open(path) as pdf:
                page_texts=[page.get_text() for page in pdf]
                pages=len(pdf)
            digest=hashlib.sha256(body).hexdigest()
            version_match=re.search(r"arXiv:\s*"+re.escape(arxiv_id)+r"(v\d+)"," ".join(page_texts[:2]))
            version=arxiv_id+(version_match.group(1) if version_match else ":version_unknown_bytes_pinned")
            source_id="qasper_"+arxiv_id.replace(".","_")
            source={"id":source_id,"path":str(path.relative_to(root)).replace("\\","/"),"title":paper["title"],
                "sha256":digest,"pages":pages,"family":"arxiv:"+arxiv_id,"question_split":"dev",
                "domain":"computer_science_ml","source_url":candidate["public_pdf_url"],"version":version,
                "license":candidate["pdf_license"],"source_status":"downloaded_bytes_and_excerpt_checked_pending_human_review"}
            imported=[]
            for qa in paper["qas"]:
                annotations=qa.get("answers",[])
                annotation=next((a.get("answer",a) for a in annotations if not a.get("answer",a).get("unanswerable")
                    and a.get("answer",a).get("evidence")),None)
                if annotation is None:
                    continue
                spans=annotation.get("extractive_spans") or []
                truth="; ".join(spans) or annotation.get("free_form_answer") or (
                    "Yes" if annotation.get("yes_no") is True else "No" if annotation.get("yes_no") is False else "")
                if not truth:
                    continue
                evidence=[];complete=True
                source_evidence=annotation.get("highlighted_evidence") or annotation["evidence"]
                for evidence_index,paragraph in enumerate(source_evidence):
                    if paragraph.startswith("FLOAT SELECTED"):
                        complete=False;break
                    match=locate_quote(paragraph,page_texts)
                    if match is None:
                        complete=False;break
                    page_number,quote=match
                    evidence.append({"id":f"e{evidence_index+1}","source_id":source_id,"filename":path.name,
                        "page":page_number,"quote":quote,"evidence_source":"pdf_text"})
                if not complete or not evidence:
                    continue
                question=qa["question"]
                lower=question.casefold()
                category="numerical" if re.search(r"\d",truth) and any(x in lower for x in ("how many","how much","number","size","percentage")) else (
                    "summary" if any(x in lower for x in ("main findings","main results","what are the results","conclusions")) else "method")
                imported.append({"id":"QASPER-"+qa["question_id"],"question":question,"ground_truth":truth,
                    "question_split":"dev","track":"local","domain":"computer_science_ml",
                    "family_ids":[source["family"]],"corpus_scope":[source_id],"category":category,
                    "expect_refusal":False,"annotation_status":"source_checked_draft","human_review":{"status":"pending"},
                    "evidence":evidence,"fact_slots":[],"task_metadata":{"provenance":{"dataset":"QASPER",
                        "version":"0.3","original_split":"train","question_id":qa["question_id"],
                        "dataset_license":QASPER_LICENSE,"citation":"Dasigi et al., NAACL 2021, https://aclanthology.org/2021.naacl-main.365/",
                        "public_benchmark_may_be_in_pretraining":True},"scope_mode":"selected_paper",
                        "support_status":"annotated_paragraph_excerpt_located_exactly_in_downloaded_pdf_pending_semantic_review",
                        "fact_slot_status":"pending_human_decomposition","category_status":"proposed_pending_human_review"}})
            if imported:
                corpus.append(source);items.extend(imported)
                candidate.update(file_sha256=digest,local_path=source["path"],version=version,matched_draft_questions=len(imported))
            else:
                failures.append({"candidate_id":candidate["candidate_id"],"reason":"no_annotated_evidence_exactly_matched_downloaded_version"})
        except Exception as exc:
            failures.append({"candidate_id":candidate["candidate_id"],"reason":type(exc).__name__})
        print(json.dumps({"progress":"qasper_pdf_source_checked","papers_with_questions":len(corpus),
                          "unique_draft_questions":len({_normalize(item['question']) for item in items})}),flush=True)
        if max_questions and len({_normalize(item['question']) for item in items})>=max_questions:
            break
    # Same short wording scoped to different papers is not expanded into duplicate primary items.
    unique=[];questions=set()
    for item in items:
        key=_normalize(item["question"])
        if key not in questions:
            questions.add(key);unique.append(item)
    if max_questions:
        unique=unique[:max_questions]
    used={sid for item in unique for sid in item["corpus_scope"]}
    return {"schema_version":3,"annotation_status":"source_checked_draft_pending_human_review",
            "description":"Licensed public QASPER train questions, exact PDF excerpt location checked; no RefMind human approval.",
            "page_tolerance":0,"corpus":[s for s in corpus if s["id"] in used],"items":unique},failures


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--max-papers",type=int,default=12)
    parser.add_argument("--max-questions",type=int)
    parser.add_argument("--collection-tag",default="v3")
    args=parser.parse_args(argv)
    if not args.download:
        parser.error("explicit --download is required for public network collection")
    root=args.root.resolve();cache=root/"evaluation/workspaces/public-source-v3"
    archive=fetch(QASPER_URL,cache/"qasper-train-dev-v0.3.tgz")
    with tarfile.open(fileobj=io.BytesIO(archive),mode="r:gz") as tar:
        # Read one named regular member; never extract arbitrary paths or execute dataset code.
        member=next(member for member in tar.getmembers() if member.name.endswith("qasper-train-v0.3.json") and member.isfile())
        raw=tar.extractfile(member).read()
        qasper=json.loads(raw)
    candidates=collect_candidates(root,cache,qasper,download=True,dev_papers=args.max_papers)
    dataset,failures=import_qasper(root,cache,qasper,candidates,max_papers=args.max_papers,max_questions=args.max_questions)
    if not re.fullmatch(r"[a-zA-Z0-9_]+",args.collection_tag):
        raise ValueError("collection tag must be a simple filename label")
    output=root/("evaluation/datasets/public_qasper_dev_"+args.collection_tag+".json")
    metadata=root/("evaluation/datasets/public_source_candidates_"+args.collection_tag+".json")
    for path in (output,metadata):
        if path.exists():
            raise ValueError("public draft already exists; choose a new collection version instead of overwriting")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dataset,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    metadata.write_text(json.dumps({"schema_version":3,"kind":"public_source_candidates","collected_at":datetime.now(timezone.utc).isoformat(),
        "qasper_source":QASPER_URL,"qasper_archive_sha256":hashlib.sha256(archive).hexdigest(),
        "qa_license":QASPER_LICENSE,"family_independence":"pending_human_review","candidates":candidates,
        "download_or_match_failures":failures},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    from .dataset import load_golden_set
    if dataset["items"]:
        load_golden_set(output,project_root=root,verify_files=True)
    print(json.dumps({"candidate_families":len(candidates),"downloaded_matched_families":len(dataset["corpus"]),
                      "actual_new_draft_questions":len(dataset["items"]),"human_approved":0,"failures":len(failures)}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
