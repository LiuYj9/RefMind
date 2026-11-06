"""Scientific aliases and controlled section/neighbor coverage."""
from __future__ import annotations
import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation

from langchain_core.documents import Document

_MEASUREMENT = re.compile(r"(?<![\w.])([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(mAh\s*/\s*g|MPa|MHz|kHz|Hz|nΩ|µΩ|μΩ|mΩ|Ω|mA|µA|μA|mV|°C|℃|K|T|A|V|%)(?![\w])")


def scientific_aliases(text: str) -> list[str]:
    aliases = []
    for match in _MEASUREMENT.finditer(text):
        try:
            number = Decimal(match.group(1)).normalize()
            if len(number.as_tuple().digits) > 100 or abs(number.as_tuple().exponent) > 100:
                continue
            value = format(number, "f")
        except InvalidOperation:
            continue
        unit = re.sub(r"\s+", "", match.group(2)).replace("μ", "µ").replace("℃", "°C")
        aliases.append("measurement:" + value + ":" + unit)
    for symbol in re.findall(r"\b[A-Za-zΑ-Ωα-ω]+_(?:\{[^{}\s]+\}|[A-Za-z0-9]+)", text):
        aliases.append("symbol:" + symbol.replace("{", "").replace("}", ""))
    return aliases


def retrieve_for_task(group_id: int, query: str, *, selected_doc_ids: tuple[int, ...] | None = None,
                      task_type: str = "method", expand_neighbors: bool = True) -> list[Document]:
    from .retrieval import get_retriever
    retriever = get_retriever(group_id, selected_doc_ids=selected_doc_ids)
    result = list(retriever.invoke(query) or []) if retriever is not None else []
    return expand_research_candidates(group_id, result, selected_doc_ids=selected_doc_ids,
                                      task_type=task_type, expand_neighbors=expand_neighbors)


def expand_research_candidates(group_id: int, result: list[Document], *,
                               selected_doc_ids: tuple[int, ...] | None = None,
                               task_type: str = "method", expand_neighbors: bool = True,
                               document_filter=None) -> list[Document]:
    """Expand a merged direct pool once; an optional request snapshot fences all pools."""
    from .retrieval import filter_ready_documents
    from .document_processor import load_group_documents
    def scoped_documents(ids):
        documents = load_group_documents(group_id, selected_doc_ids=ids)
        return document_filter(documents) if document_filter else documents
    result = document_filter(result) if document_filter else list(result)
    # Capture direct hits before neighbor expansion and first-occurrence dedup.
    # A later direct hit can already have appeared as an earlier hit's neighbor.
    def identity(document):
        meta = document.metadata
        return (meta.get("doc_id"), meta.get("version"), meta.get("chunk_id"), document.page_content)

    direct = {}
    origins = {}

    def origin(document):
        return origins.setdefault(identity(document), {
            "direct_rank": None, "neighbor_of": [], "section_coverage": False})

    for rank, document in enumerate(result, 1):
        if identity(document) not in direct:
            direct[identity(document)] = document
            origin(document)["direct_rank"] = rank
    if task_type == "summary":
        scoped = scoped_documents(selected_doc_ids)
        seen_sections = set()
        coverage, per_paper = [], {}
        for doc in sorted(scoped, key=lambda d: (int(d.metadata.get("doc_id", -1)), int(d.metadata.get("chunk_index", 0)))):
            meta = doc.metadata
            section = str(meta.get("section_path") or meta.get("section") or "正文")
            if re.search(r"^(references|bibliography|参考文献)$", section, re.I):
                continue
            key = (meta.get("doc_id"), section)
            if key not in seen_sections:
                per_paper.setdefault(meta.get("doc_id"), []).append(doc)
                seen_sections.add(key)
        # Round-robin section coverage prevents the first long paper consuming every slot.
        while per_paper and len(coverage) < 80:
            for paper_id in list(per_paper):
                coverage.append(per_paper[paper_id].pop(0))
                if not per_paper[paper_id]:
                    del per_paper[paper_id]
                if len(coverage) >= 80:
                    break
        for document in coverage:
            origin(document)["section_coverage"] = True
        result = [*coverage, *result]
    elif expand_neighbors and result:
        ids = tuple(sorted({int(d.metadata["doc_id"]) for d in result if int(d.metadata.get("doc_id", -1)) > 0}))
        scoped = scoped_documents(ids) if ids else []
        positions = {(d.metadata.get("doc_id"), d.metadata.get("version"), int(d.metadata.get("chunk_index", -999))): d for d in scoped}
        expanded = []
        for doc in result:
            expanded.append(doc)
            meta = doc.metadata
            if meta.get("content_type") in {"table", "equation", "figure", "figure_caption"}:
                continue
            index = int(meta.get("chunk_index", -999))
            for offset in (-1, 1):
                neighbor = positions.get((meta.get("doc_id"), meta.get("version"), index + offset))
                if neighbor is not None and (neighbor.metadata.get("section_path") or neighbor.metadata.get("section")) == (meta.get("section_path") or meta.get("section")):
                    parent = {"doc_id": meta.get("doc_id"), "version": meta.get("version"),
                              "chunk_id": meta.get("chunk_id"), "chunk_index": index,
                              "direct_rank": origin(doc)["direct_rank"], "offset": offset}
                    parents = origin(neighbor)["neighbor_of"]
                    if parent not in parents:
                        parents.append(parent)
                    expanded.append(Document(page_content=neighbor.page_content, metadata={**neighbor.metadata, "retrieval_neighbor": True}))
        result = expanded
    output, seen = [], set()
    ready = filter_ready_documents(group_id, result, selected_doc_ids)
    for doc in (document_filter(ready) if document_filter else ready):
        key = identity(doc)
        if key not in seen:
            provenance = origin(doc)
            metadata = {**doc.metadata,
                        "retrieval_origin": {**provenance,
                            "neighbor_of": [dict(parent) for parent in provenance["neighbor_of"]]},
                        "retrieval_neighbor": bool(provenance["neighbor_of"]) and key not in direct}
            # Only this request's direct hit owns channel/fusion observations.
            # The first occurrence may be a pool neighbor or summary-coverage block.
            for field in ("_retrieval_identity", "retrieval_channels", "retrieval_fusion", "retrieval_runs"):
                metadata.pop(field, None)
                if key in direct and field in direct[key].metadata:
                    metadata[field] = deepcopy(direct[key].metadata[field])
            output.append(Document(page_content=doc.page_content, metadata=metadata))
            seen.add(key)
    return output
