"""Read verified PDF page 8 and generate honest extraction/fallback teaching records."""
from pathlib import Path
import argparse, datetime, hashlib, json, sys
import fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from langchain_core.documents import Document
from refmind.rag.extractive_views import (
    build_extractive_views, validate_extractive_view, _sentence_ranges,
    _NEGATION, _CONDITION, _OBJECT, _UNIT,
)
from tests.test_extractive_views import TEXT as SYNTHETIC_TEXT, focused

def sha(data):
    return hashlib.sha256(data).hexdigest()

def fixture_score(query,sentences):
    results=[]
    for sentence in sentences:
        text=" ".join(sentence.split()).casefold()
        if "eddy current induced on the ni hts coil is reduced by copper shields" in text:
            results.append(.95)
        elif "increases the total losses" in text:
            results.append(.91)
        else:
            results.append(.02)
    return results
fixture_score.score_type="teaching_fixed_sentence_scores_not_model_run"

def guards(text):
    out=[]
    if any(c.isdigit() for c in text):out.append("包含数值，不能脱离原句删除")
    for regex,label in [(_UNIT,"包含单位或量纲"),(_NEGATION,"包含否定、例外或范围限制"),
                        (_CONDITION,"包含条件、比较关系或转折"),(_OBJECT,"包含研究对象、工况或损耗关系")]:
        if regex.search(text):out.append(label)
    return out

def locate_chars(page,block):
    target=next(b for b in page.get_text("rawdict")["blocks"]
                if b.get("type")==0 and max(abs(a-c) for a,c in zip(b["bbox"],block[:4]))<.01)
    text="";positions=[]
    for line_index,line in enumerate(target["lines"]):
        for span in line["spans"]:
            for char in span["chars"]:
                for character in char["c"]:
                    positions.append((len(text),line_index,list(char["bbox"])))
                    text+=character
        text+="\n"
    assert text==block[4],"Raw glyph text differs from source block"
    return positions

def rects(positions,start,end):
    lines={}
    for offset,line,box in positions:
        if start<=offset<end:
            if line not in lines:lines[line]=box[:]
            else:
                old=lines[line];old[0]=min(old[0],box[0]);old[1]=min(old[1],box[1])
                old[2]=max(old[2],box[2]);old[3]=max(old[3],box[3])
    return list(lines.values())

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--pdf",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();pdf_path=Path(args.pdf);output=Path(args.output)
    pdf=fitz.open(pdf_path);page=pdf[7];blocks=page.get_text("blocks")
    cases=[]
    for index,title in [(2,"铜屏蔽降低线圈损耗：保留数值、对象与工况"),(9,"线圈损耗降低并不保证系统总损耗降低")]:
        block=blocks[index];source=block[4];positions=locate_chars(page,block)
        document=Document(page_content=source,metadata={
            "content_type":"paragraph","group_id":"teaching","doc_id":"A","chunk_id":f"teaching:A:pdf8:pdfblock{index}",
            "chunk_index":index,"version":"page-excerpt-fixture-v1","filename":pdf_path.name,"page":8,"page_start":8,"page_end":8,
            "bbox":list(block[:4]),"bbox_coordinate_system":"pdf_points",
            "source_kind":"verified_physical_page_excerpt_not_production_chunk",
        })
        result=build_extractive_views("铜屏蔽是否能使所有工况下的系统总损耗下降？",[document],scorer=fixture_score)
        record=result.records[0];view=result.documents[0].page_content
        assert validate_extractive_view(source,view,record)
        sentences=[]
        for ordinal,(start,end) in enumerate(_sentence_ranges(source)):
            selected=any(s["start"]<=start and s["end"]>=end for s in record["selected_spans"])
            reasons=guards(source[start:end])
            if record["sentence_scores"] and record["sentence_scores"][ordinal]>=.35:
                reasons.insert(0,"教学评分：直接涉及本例问题（不是模型实测）")
            if selected and not reasons:reasons.append("保留相邻语境，或因整块回退保留")
            sentences.append({"index":ordinal,"start":start,"end":end,"text":source[start:end],
                              "selected":selected,"reasons":reasons,"pdf_rectangles":rects(positions,start,end)})
        cases.append({"id":f"real_pdf8_{index}","title":title,"source_text":source,"view_text":view,
                      "record":record,"sentences":sentences,"pdf_physical_page":8,"page_size_points":[page.rect.width,page.rect.height],
                      "bbox_coordinate_system":"pdf_points","page_rotation":page.rotation,"block_bbox":list(block[:4]),
                      "production_chunk_claimed":False,
                      "scope_note":"此为物理第8页的可核对正文摘录。段落在页内或跨栏处延续；不冒充生产索引chunk，不声称覆盖论文所有工况。",
                      "expected_teaching_message":"相关性分数低并不等于可删；条件、数值、对象和反向结论受到保护，无安全收益则保留原块。"})
    synthetic=Document(page_content=SYNTHETIC_TEXT,metadata={"content_type":"paragraph","group_id":"teaching",
                       "doc_id":"synthetic","chunk_id":"synthetic:mechanism-only","filename":"synthetic-teaching-text","version":"v1"})
    positive=build_extractive_views("铜屏蔽 降损",[synthetic],scorer=focused)
    assert positive.records[0]["status"]=="extracted"
    assert validate_extractive_view(SYNTHETIC_TEXT,positive.documents[0].page_content,positive.records[0])
    payload={"schema":1,"createdUTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "basis":"Actual function execution on physical-PDF excerpts and a separately labelled synthetic mechanism fixture",
             "model_run":False,"embedding_api_called":False,"source_pdf":str(pdf_path),
             "source_pdf_sha256":sha(pdf_path.read_bytes()),"module_sha256":sha((ROOT/"refmind/rag/extractive_views.py").read_bytes()),
             "original_source_modified":False,"real_cases":cases,
             "synthetic_positive_case":{"label":"合成机制教学文本；不是论文摘录或模型实测","source_coordinates_available":False,
                "source_text":SYNTHETIC_TEXT,"view_text":positive.documents[0].page_content,
                "record":positive.records[0],"metadata":positive.documents[0].metadata},
             "suggested_short_display":[
                {"text":"排名靠后 ≠ 可以删除","basis":"原句含对象、数值、条件或反向结论时仍保留"},
                {"text":"线圈损耗降低，不等于系统总损耗始终降低","basis":"PDF第8页右栏含 increases the total losses 的原句"},
                {"text":"没有安全压缩收益 → 保留原块","basis":"真实段落执行结果 unchanged_no_gain"},
                {"text":"省略仅发生在副本，保留区间仍指向原句","basis":"另行明确标注的合成机制教学例"}
             ]}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),"utf-8")
    print(json.dumps({"output":str(output),"real_statuses":[c["record"]["status"] for c in cases],
                      "real_sentence_counts":[len(c["sentences"]) for c in cases],
                      "synthetic_status":positive.records[0]["status"]},ensure_ascii=False))
if __name__=="__main__":main()
