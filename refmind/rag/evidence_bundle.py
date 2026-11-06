"""Immutable, source-aware evidence packing and conservative query resolution."""
from __future__ import annotations

import os
import hashlib
import re
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Callable

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ..citations import build_citation_label


@dataclass
class TaskDescription:
    original_question: str
    resolved_query: str
    task_type: str
    conditions: list[str] = field(default_factory=list)
    source_titles: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    resolution: str = "original"
    source_ids: list[str] = field(default_factory=list)
    required_source_ids: list[str] = field(default_factory=list)
    target_resolution: str = "unresolved_allowed_scope"

    def to_dict(self):
        return asdict(self)


def paired_history(history: list[BaseMessage], max_pairs: int = 6) -> list[BaseMessage]:
    pairs, pending = [], None
    for message in history:
        if isinstance(message, HumanMessage):
            pending = message
        elif isinstance(message, AIMessage) and pending is not None:
            pairs.append((pending, message))
            pending = None
    return [message for pair in pairs[-max_pairs:] for message in pair]


def _source_ids(values, name: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must be a list of nonempty source identities")
    return list(dict.fromkeys(value.strip() for value in values))


def required_targets(description: dict, fallback_source_ids: list[str]) -> list[str]:
    """Resolve only declared requirements; an explicit [] never inherits candidates."""
    allowed = _source_ids(description.get("source_ids", fallback_source_ids), "source_ids")
    if "required_source_ids" not in description:
        return allowed or _source_ids(fallback_source_ids, "fallback_source_ids")
    if description["required_source_ids"] is None:
        raise ValueError("required_source_ids must be a list; use [] for unresolved targets")
    required = _source_ids(description["required_source_ids"], "required_source_ids")
    if not set(required) <= set(allowed):
        raise ValueError("required_source_ids must be a subset of the allowed source_ids")
    if required and description.get("target_resolution") == "unresolved_allowed_scope":
        raise ValueError("unresolved_allowed_scope cannot declare resolved required sources")
    return required


def describe_task(question: str, history: list[BaseMessage] | None = None, source_titles: list[str] | None = None, *, source_ids: list[str] | None = None, required_source_ids: list[str] | None = None, target_resolution: str | None = None) -> TaskDescription:
    kinds = [("comparison", r"比较|对比|相比|compare|versus|\bvs\b"),
             ("summary", r"总结|综述|概括|局限|summari|overview|limitations"),
             ("table", r"表\s*\d|表格|\btable\b"), ("figure", r"图\s*\d|曲线|图像|\bfig(?:ure)?\b|plot"),
             ("equation", r"公式|方程|符号|equation|formula"),
             ("numeric", r"多少|数值|温度|电流|比容量|how much|how many|value|accuracy|F1")]
    kind = next((name for name, pattern in kinds if re.search(pattern, question, re.I)), "method")
    conditions = re.findall(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?\s*(?:K\b|°C|℃|T\b|A\b|Hz\b|MPa\b|mAh\s*/\s*g|%|V\b)", question, re.I)
    titles = list(dict.fromkeys(source_titles or []))
    identities = _source_ids(source_ids, "source_ids")
    required = identities if required_source_ids is None else _source_ids(required_source_ids, "required_source_ids")
    resolution = target_resolution or ("selected_sources" if required else "unresolved_allowed_scope")
    required_targets({"source_ids": identities, "required_source_ids": required,
                      "target_resolution": resolution}, [])
    task = TaskDescription(question, question, kind, conditions, titles, source_ids=identities,
                           required_source_ids=required, target_resolution=resolution)
    source_count = len(required) if required else len(identities) if source_ids is not None else len(titles)
    pronoun = re.search(r"该论文|这篇|上篇|它(?:们)?|该方法|上述|前者|后者|\b(?:it|its|that paper|this method|former|latter)\b", question, re.I)
    if pronoun:
        if source_count == 1:
            display = titles[0] if titles and (len(titles) == 1 or required == identities) else (required or identities)[0]
            task.resolved_query = f"{question}\n明确文献：{display}"
            task.resolution = "selected_single_paper"
        else:
            previous = paired_history(history or [], 1)
            if not previous or re.search(r"前者|后者|former|latter", question, re.I):
                task.needs_clarification = True
                task.resolution = "ambiguous_reference"
            elif not conditions:
                task.resolved_query = f"{question}\n用于解析指代的上一轮提问（不作论文证据）：{previous[0].content}"
                task.resolution = "previous_user_question"
            else:
                # Do not append stale numeric conditions to an explicitly changed query.
                task.needs_clarification = True
                task.resolution = "changed_conditions_ambiguous_paper"
    return task


def source_identity(doc: Document) -> str:
    """Stable selected-source identity; titles never determine source equality."""
    m = doc.metadata or {}
    if m.get("resource_id") or m.get("asset_id"):
        return "asset:" + str(m.get("resource_id") or m["asset_id"])
    if m.get("evidence_origin") == "academic_search":
        paper_id = m.get("paper_id") or m.get("evidence_id") or m.get("source_id")
        if paper_id:
            return "abstract:" + str(paper_id)
        if m.get("doi"):
            return "abstract:doi:" + str(m["doi"]).strip().casefold()
    elif m.get("doc_id") not in {None, -1, ""}:
        return f"library:{m.get('group_id')}:{m['doc_id']}"
    if m.get("source_id") or m.get("filename") or m.get("source"):
        return "legacy:" + str(m.get("source_id") or m.get("filename") or m["source"])
    # Old anonymous evidence has no declared paper identity. Keep it distinct by
    # content rather than silently merging different papers with the same title.
    return "unidentified:" + hashlib.sha256(doc.page_content.encode()).hexdigest()


def source_key(doc: Document) -> str:
    identity = source_identity(doc)
    if identity.startswith("library:"):
        return identity + ":" + str(doc.metadata.get("version", ""))
    return identity


@lru_cache(maxsize=1)
def _openai_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("o200k_base")


def token_counter() -> tuple[Callable[[str], int], str]:
    if os.getenv("CHAT_PROVIDER", "dashscope") == "openai":
        try:
            encoding = _openai_tokenizer()
            return lambda text: len(encoding.encode(text, disallowed_special=())), "o200k_base"
        except Exception:
            pass
    # A byte bound avoids pretending an OpenAI tokenizer matches another provider.
    return lambda text: len(text.encode("utf-8")), "utf8_byte_upper_bound"


@dataclass
class EvidenceBundle:
    documents: list[Document]
    history: list[BaseMessage]
    trace: dict
    warnings: list[str]


def _table_view(doc: Document, query: str, remaining: int, count: Callable[[str], int], *,
                citation_index: int = 1) -> Document | None:
    """Only Markdown tables with explicit headers; never infer cells/rowspans."""
    text = doc.page_content
    if "<table" in text.lower():
        return None
    lines = text.splitlines()
    separator = next((i for i, line in enumerate(lines) if i > 0 and re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*", line)), None)
    if separator is None:
        return None
    end = separator + 1
    while end < len(lines) and "|" in lines[end] and lines[end].strip():
        end += 1
    prefix, suffix = lines[:separator + 1], lines[end:]
    data = list(enumerate(lines[separator + 1:end], separator + 2))
    words = set(re.findall(r"[\w.]+", query.casefold()))
    ranked = sorted(data, key=lambda pair: (-len(words & set(re.findall(r"[\w.]+", pair[1].casefold()))), pair[0]))
    selected = []
    for row in ranked:
        trial = sorted([*selected, row])
        content = "\n".join(["（以下仅展示选中原始行；其余行未纳入本轮证据）", *prefix, *(value for _, value in trial), *suffix])
        candidate = Document(page_content=content, metadata=doc.metadata)
        if count(build_citation_label(candidate, citation_index) + "\n" + content) <= remaining:
            selected = trial
    if not selected:
        return None
    content = "\n".join(["（以下仅展示选中原始行；其余行未纳入本轮证据）", *prefix, *(value for _, value in selected), *suffix])
    metadata = {**doc.metadata, "evidence_view": "explicit_table_rows", "source_rows": [i for i, _ in selected], "source_text": text, "coverage_limit": "只覆盖所列原始表格行，不能推断整表最优或全体结论"}
    return Document(page_content=content, metadata=metadata)


def pack_evidence(question: str, documents: list[Document], *, history: list[BaseMessage] | None = None,
                  system_prompt: str = "", task_type: str = "method", evidence_tokens: int = 4000,
                  model_context_tokens: int = 32768, output_reserve: int = 4096,
                  counter: Callable[[str], int] | None = None, required_source_ids: list[str] | None = None,
                  prioritize_body: bool = False) -> EvidenceBundle:
    count, estimator = (counter, "injected") if counter else token_counter()
    history = paired_history(history or [], 6)
    fixed = count(system_prompt + question) + 512
    while history and fixed + sum(count(str(m.content)) + 8 for m in history) + output_reserve + evidence_tokens > model_context_tokens:
        history = history[2:]
    overhead = fixed + sum(count(str(m.content)) + 8 for m in history)
    available = max(0, min(evidence_tokens, model_context_tokens - output_reserve - overhead))
    groups: OrderedDict[str, deque] = OrderedDict()
    seen = set()
    for doc in documents:
        key = (source_key(doc), doc.metadata.get("chunk_id"), doc.page_content)
        if key in seen:
            continue
        seen.add(key)
        # Broad summaries take one chunk per available section before extras.
        group = source_key(doc)
        if task_type == "summary":
            group += ":" + str(doc.metadata.get("section_path") or doc.metadata.get("section") or "正文")
        groups.setdefault(group, deque()).append(doc)
    ordered = []
    if task_type in {"comparison", "summary"}:
        while any(groups.values()):
            for items in groups.values():
                if items:
                    ordered.append(items.popleft())
    else:
        ordered = [doc for items in groups.values() for doc in items]
        # Keep original reranking order for ordinary questions.
        ordered.sort(key=lambda d: next(i for i, item in enumerate(documents) if item is d))
    priority_trace = {"enabled": prioritize_body, "policy": "reference_demotion_B_safe", "moves": []}
    if prioritize_body:
        from .evidence_priority import prioritize_evidence, evidence_kind
        original_order = list(ordered)
        ordered = prioritize_evidence(question, ordered, mode="references", strategy="B_safe")
        original_positions = {id(document): index for index, document in enumerate(original_order, 1)}
        for index, document in enumerate(ordered, 1):
            previous = original_positions[id(document)]
            if index != previous:
                priority_trace["moves"].append({"before": previous, "after": index,
                    "source": source_identity(document), "chunk_id": document.metadata.get("chunk_id"),
                    "reason": evidence_kind(document, strategy="B_safe")["reason"]})
    packed, omitted, warnings = [], [], []
    used = 0
    used_images = 0
    for doc in ordered:
        citation_index = len(packed) + 1
        text_cost = count(build_citation_label(doc, citation_index) + "\n" + doc.page_content) + 8
        image_cost = 4096 if doc.metadata.get("image_path") else 0
        remaining = min(available - used,
                        model_context_tokens - overhead - used - used_images - image_cost - output_reserve)
        if text_cost > remaining and doc.metadata.get("content_type") == "table":
            view = _table_view(doc, question, remaining - 8, count, citation_index=citation_index)
            if view is not None:
                doc = view
                text_cost = count(build_citation_label(doc, citation_index) + "\n" + doc.page_content) + 8
        if text_cost > remaining:
            omitted.append({"source": source_key(doc), "chunk_id": doc.metadata.get("chunk_id"), "reason": "budget_cannot_fit_whole_evidence"})
            continue
        metadata = {**doc.metadata, "_evidence_pack_complete": True}
        packed.append(Document(page_content=doc.page_content, metadata=metadata))
        used += text_cost
        used_images += image_cost
    missing_sources = sorted({source_key(d) for d in documents} - {source_key(d) for d in packed})
    if omitted:
        warnings.append(f"证据预算未容纳 {len(omitted)} 个完整片段；未截断公式或原始结构")
    missing_required = None if required_source_ids is None else sorted(
        set(_source_ids(required_source_ids, "required_source_ids")) - {source_identity(d) for d in packed})
    if (missing_sources if missing_required is None else missing_required) and task_type in {"comparison", "summary"}:
        warnings.append("部分目标文献未进入最终上下文，不能据此给出完整比较或综述")
    return EvidenceBundle(packed, history, {"token_estimator": estimator, "evidence_budget": available,
        "evidence_tokens": used, "image_token_reserve": used_images, "prompt_history_tokens": overhead, "output_reserve": output_reserve,
        "model_context_tokens": model_context_tokens, "omitted": omitted, "missing_sources": missing_sources,
        "missing_required_source_ids": missing_required, "evidence_priority": priority_trace}, warnings)


def enforce_generation_budget(documents: list[Document], history: list[BaseMessage], *,
                              measure: Callable[[list[Document], list[BaseMessage]], dict],
                              evidence_tokens: int, model_context_tokens: int,
                              output_reserve: int) -> EvidenceBundle:
    """Fit the actual post-review payload; remove whole blocks, never just edit counts."""
    history = list(paired_history(history, 6))
    dropped_pairs = 0
    while history:
        totals = measure(documents, history)
        if totals["prompt_tokens"] + totals.get("image_token_reserve", 0) + output_reserve <= model_context_tokens:
            break
        history = history[2:]
        dropped_pairs += 1
    selected, omitted = [], []
    for document in documents:
        trial = [*selected, document]
        totals = measure(trial, history)
        if (totals["evidence_tokens"] > evidence_tokens or
            totals["prompt_tokens"] + totals.get("image_token_reserve", 0) + output_reserve > model_context_tokens):
            omitted.append({"source": source_key(document), "chunk_id": document.metadata.get("chunk_id"),
                            "reason": "post_review_payload_exceeds_budget"})
            continue
        selected.append(document)
    actual = measure(selected, history)
    missing = sorted({source_key(doc) for doc in documents} - {source_key(doc) for doc in selected})
    warnings = []
    if omitted:
        warnings.append(f"审查后完整提示预算排除 {len(omitted)} 个片段；结构未被截断")
    if dropped_pairs:
        warnings.append(f"审查后完整提示预算移除 {dropped_pairs} 组较早会话")
    trace = {**actual, "evidence_budget": evidence_tokens, "model_context_tokens": model_context_tokens,
             "output_reserve": output_reserve, "omitted": omitted, "missing_sources": missing,
             "dropped_history_pairs": dropped_pairs, "post_review_checked": True,
             "prompt_history_tokens": actual["prompt_tokens"] - actual["evidence_tokens"]}
    return EvidenceBundle(selected, history, trace, warnings)
