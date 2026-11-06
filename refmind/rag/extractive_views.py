"""Request-local, lossless extractive views. Scores are not answer confidence."""
from __future__ import annotations
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import re
from typing import Any, Callable, Sequence
from langchain_core.documents import Document
from .evidence_bundle import source_identity

SentenceScorer = Callable[[str, Sequence[str]], Sequence[float]]
OMISSION_MARKER = "\n[……此处省略原文；非连续摘录……]\n"
HARD_BLOCK_LIMIT = 12
HARD_SENTENCE_LIMIT = 24
MIN_SEED_SCORE = 0.35
MAX_OMITTED_SCORE = 0.25
MIN_SAVED_CHARS = 40
MIN_SAVED_FRACTION = 0.10
_STRUCTURED = re.compile(r"<(?:table|tr|td)\b|\$\$|\\begin\{|^\s*\x60{3}|^\s*\|.*\|", re.I | re.M)
_NEGATION = re.compile(
    r"\b(?:no|not|never|without|neither|nor|cannot|can't|doesn't|didn't|isn't|aren't|"
    r"wasn't|weren't|fails?|failed|except|unless|only)\b|不|未|无|没有|并非|仅|除外", re.I)
_CONDITION = re.compile(
    r"\b(?:if|when|while|under|within|between|provided|assuming|whereas|however|"
    r"although|compared|relative|than|depends?|condition|range|fixed|given)\b|"
    r"(?:^|[，,；;])\s*当|若|如果|条件|工况|范围|前提|情况下|限定|相比|相对|但是|然而|其中|式中", re.I)
_OBJECT = re.compile(
    r"\b(?:samples?|specimens?|coils?|windings?|conductors?|magnets?|materials?|"
    r"systems?|devices?|models?|datasets?|patients?|participants?|experiments?|"
    r"simulations?|copper|shields?|temperature|current|frequency|field)\b|"
    r"样品|样本|线圈|绕组|导体|磁体|材料|系统|设备|模型|数据集|研究对象|"
    r"实验|试验|仿真|铜|屏蔽|温度|电流|频率|磁场", re.I)
_UNIT = re.compile(
    r"\b(?:kelvin|tesla|ampere|hertz|watt|joule|MPa|mAh|kHz|MHz|Hz|mT|kA|"
    r"mA|mV|kV|mW|kW|mm|cm|nm|kg)\b|[°℃%‰μµ]|"
    r"摄氏|开尔文|安培|赫兹|特斯拉|焦耳|瓦特", re.I)
_ANAPHORA = re.compile(
    r"^\s*[\"'“‘(（]*(?:this|that|these|those|it|its|they|their|such|"
    r"the former|the latter|the above)\b|^\s*(?:这|该|其|它|上述|前者|后者)", re.I)
_UNRESOLVED_REFERENCE = re.compile(r"\b(?:former|latter|above-mentioned)\b|前者|后者|上述多个", re.I)
_WORDS = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|[\u3400-\u9fff]+")
_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "be",
         "for", "what", "which", "how", "can", "does", "with", "on", "this", "that"}
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "fig.", "eq.", "dr.", "prof.", "vs.")


@dataclass
class ExtractiveViewResult:
    documents: list[Document]
    records: list[dict[str, Any]]
    coverage_limit: dict[str, Any]
    scorer: str
    score_type: str

    def audit_dict(self) -> dict[str, Any]:
        return {"records": deepcopy(self.records), "coverage_limit": deepcopy(self.coverage_limit),
                "scorer": self.scorer, "score_type": self.score_type,
                "score_purpose": "sentence_selection_only_not_answer_confidence"}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _WORDS.findall(text.casefold()):
        if re.search(r"[\u3400-\u9fff]", token):
            tokens.update(token[i:i + 2] for i in range(len(token) - 1))
        elif token not in _STOP:
            tokens.add(token)
    return tokens


def _lexical_scores(query: str, sentences: Sequence[str]) -> list[float]:
    terms = _tokens(query)
    if len(terms) < 2:
        return [0.0] * len(sentences)
    return [len(terms & _tokens(sentence)) / len(terms) for sentence in sentences]


_lexical_scores.score_type = "offline_lexical_overlap"  # type: ignore[attr-defined]


def make_embedding_scorer(embeddings: Any = None) -> SentenceScorer:
    """Explicit service opt-in. Construction is offline; failures retain source blocks."""
    model = embeddings

    @lru_cache(maxsize=1)
    def encode_query(query: str) -> tuple[float, ...]:
        nonlocal model
        if model is None:
            from ..llm import get_embedding_model
            model = get_embedding_model()
        return tuple(float(value) for value in model.embed_query(query))

    def score(query: str, sentences: Sequence[str]) -> list[float]:
        query_vector = encode_query(query)
        vectors = model.embed_documents(list(sentences))
        if len(vectors) != len(sentences):
            raise ValueError("embedding_count_mismatch")
        result: list[float] = []
        for vector in vectors:
            values = tuple(float(value) for value in vector)
            if not values or len(values) != len(query_vector):
                raise ValueError("embedding_dimension_mismatch")
            if not all(math.isfinite(value) for value in (*query_vector, *values)):
                raise ValueError("nonfinite_embedding")
            left = math.sqrt(sum(value * value for value in query_vector))
            right = math.sqrt(sum(value * value for value in values))
            if not left or not right or not math.isfinite(left) or not math.isfinite(right):
                raise ValueError("invalid_embedding_norm")
            cosine = sum(a * b for a, b in zip(query_vector, values)) / (left * right)
            result.append(max(-1.0, min(1.0, cosine)))
        return result

    score.score_type = "sentence_query_cosine"  # type: ignore[attr-defined]
    score.__name__ = "embedding_sentence_scorer"
    return score


def _sentence_ranges(text: str) -> list[tuple[int, int]]:
    """Partition original Unicode code points without normalizing/reconstructing text."""
    ends: list[int] = []
    for match in re.finditer(r"[。！？!?]+[\"'”’）)]*|\.[\"'”’）)]*(?=\s+[\"'“‘(（]*[A-Z])|\n[ \t]*\n", text):
        end = match.end()
        if match.group().startswith(".") and any(
            text[:end].casefold().endswith(abbreviation) for abbreviation in _ABBREVIATIONS
        ):
            continue
        while end < len(text) and text[end].isspace():
            end += 1
        if not ends or end > ends[-1]:
            ends.append(end)
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    ranges: list[tuple[int, int]] = []
    start = 0
    for end in ends:
        if end <= start:
            continue
        if text[start:end].strip():
            ranges.append((start, end))
        elif ranges:
            ranges[-1] = (ranges[-1][0], end)
        else:
            continue
        start = end
    return ranges


def _plain(doc: Document) -> bool:
    metadata = doc.metadata or {}
    return (metadata.get("content_type") in (None, "", "text", "paragraph")
            and metadata.get("is_atomic") is not True and not _STRUCTURED.search(doc.page_content))


def _protected(sentence: str) -> bool:
    return bool(re.search(r"\d", sentence) or _UNIT.search(sentence)
                or _NEGATION.search(sentence) or _CONDITION.search(sentence) or _OBJECT.search(sentence))


def _spans(text: str, ranges: Sequence[tuple[int, int]]) -> list[dict[str, Any]]:
    return [{"start": start, "end": end, "quote": text[start:end], "text": text[start:end]} for start, end in ranges]


def _merge(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start == merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _complement(length: int, selected: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    omitted: list[tuple[int, int]] = []
    position = 0
    for start, end in selected:
        if start > position:
            omitted.append((position, start))
        position = end
    if position < length:
        omitted.append((position, length))
    return omitted


def _render(text: str, selected: Sequence[tuple[int, int]]) -> str:
    pieces: list[str] = []
    position = 0
    for start, end in selected:
        if start > position:
            pieces.append(OMISSION_MARKER)
        pieces.append(text[start:end])
        position = end
    if position < len(text):
        pieces.append(OMISSION_MARKER)
    return "".join(pieces)


def validate_extractive_view(source: str, view: str, record: dict[str, Any]) -> bool:
    """Validate hashes, exact half-open quotes, ordering and a complete span partition."""
    try:
        if record["offset_unit"] != "unicode_code_point" or record["interval"] != "half_open":
            return False
        if record["source_chunk_hash"] != _hash(source) or record["view_hash"] != _hash(view):
            return False
        groups: list[list[tuple[int, int]]] = []
        for key in ("selected_spans", "omitted_spans"):
            previous = -1
            ranges: list[tuple[int, int]] = []
            for span in record[key]:
                start, end = span["start"], span["end"]
                if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
                    return False
                if not 0 <= start < end <= len(source) or start < previous:
                    return False
                if source[start:end] != span["quote"] or source[start:end] != span["text"]:
                    return False
                previous = end
                ranges.append((start, end))
            groups.append(ranges)
        position = 0
        for start, end in sorted([*groups[0], *groups[1]]):
            if start != position:
                return False
            position = end
        return position == len(source) and _render(source, groups[0]) == view
    except (KeyError, TypeError, ValueError, UnicodeError):
        return False


def _selection_indices(documents: Sequence[Document], candidates: Sequence[int],
                       limit: int, round_robin: bool) -> list[int]:
    if not round_robin:
        return list(candidates[:limit])
    queues: OrderedDict[str, deque[int]] = OrderedDict()
    for index in candidates:
        queues.setdefault(source_identity(documents[index]), deque()).append(index)
    selected: list[int] = []
    while queues and len(selected) < limit:
        for identity in list(queues):
            selected.append(queues[identity].popleft())
            if not queues[identity]:
                del queues[identity]
            if len(selected) == limit:
                break
    return selected


def build_extractive_views(
    query: str, documents: Sequence[Document], *, scorer: SentenceScorer | None = None,
    task_kind: str = "method", max_blocks: int = HARD_BLOCK_LIMIT,
    max_sentences: int = HARD_SENTENCE_LIMIT,
) -> ExtractiveViewResult:
    """Return same-order copies, never deduplicate, write back, or character-truncate.

    Default scoring is offline lexical overlap. Production semantic selection
    explicitly passes make_embedding_scorer(...). Limits bound processing, not
    evidence coverage; all unprocessed chunks remain intact.
    """
    if (isinstance(max_blocks, bool) or isinstance(max_sentences, bool)
            or not isinstance(max_blocks, int) or not isinstance(max_sentences, int)):
        raise ValueError("limits_must_be_integers")
    block_limit = min(HARD_BLOCK_LIMIT, max(0, max_blocks))
    sentence_limit = min(HARD_SENTENCE_LIMIT, max(1, max_sentences))
    scoring = scorer if scorer is not None else _lexical_scores
    score_type = str(getattr(scoring, "score_type", "injected_sentence_score"))
    scorer_name = str(getattr(scoring, "__name__", type(scoring).__name__))
    candidates = [i for i, doc in enumerate(documents) if doc.page_content.strip() and _plain(doc)]
    selected_indices = _selection_indices(documents, candidates, block_limit, task_kind in {"comparison", "summary"})
    processing = set(selected_indices)
    coverage: dict[str, Any] = {
        "max_blocks": block_limit, "max_sentences_per_block": sentence_limit,
        "input_blocks": len(documents), "ordinary_text_blocks": len(candidates),
        "selected_processing_indices": selected_indices,
        "not_processed_due_to_block_limit": [i for i in candidates if i not in processing],
        "sentence_limit_indices": [], "source_round_robin": task_kind in {"comparison", "summary"},
        "scored_blocks": 0, "scored_sentences": 0, "scoring_attempts": 0, "failed_scoring_blocks": [],
        "full_evidence_coverage_claimed": False, "unprocessed_chunks_retained": True,
    }
    outputs: list[Document] = []
    records: list[dict[str, Any]] = []
    scoring_failure: str | None = None
    for index, document in enumerate(documents):
        source = document.page_content
        selected_ranges = [(0, len(source))] if source else []
        status = "unchanged_empty" if not source.strip() else "unchanged_not_plain_text"
        reason = status
        scores: list[float] = []
        ranges: list[tuple[int, int]] = []
        guard_indices: list[int] = []
        if index in candidates and index not in processing:
            status = reason = "fallback_block_limit"
        elif index in processing:
            ranges = _sentence_ranges(source)
            if len(ranges) > sentence_limit:
                coverage["sentence_limit_indices"].append(index)
                status = reason = "fallback_sentence_limit"
            elif not query.strip():
                status = reason = "fallback_empty_query"
            elif len(ranges) < 3:
                status = reason = "unchanged_too_few_sentences"
            elif scoring_failure is not None:
                status, reason = "fallback_scoring_unavailable", scoring_failure
            else:
                sentences = [source[start:end] for start, end in ranges]
                try:
                    coverage["scoring_attempts"] += 1
                    raw_scores = list(scoring(query, sentences))
                    coverage["scored_blocks"] += 1
                    coverage["scored_sentences"] += len(sentences)
                    if len(raw_scores) != len(sentences) or any(isinstance(value, bool) for value in raw_scores):
                        raise ValueError("score_shape_mismatch")
                    scores = [float(value) for value in raw_scores]
                    if any(not math.isfinite(value) or not -1 <= value <= 1 for value in scores):
                        raise ValueError("invalid_score")
                except Exception as error:
                    scores = []
                    status, reason = "fallback_scoring_error", type(error).__name__
                    scoring_failure = reason
                    coverage["failed_scoring_blocks"].append(index)
                else:
                    highest = max(scores)
                    if highest < MIN_SEED_SCORE:
                        status = reason = "fallback_ambiguous_relevance"
                    else:
                        seeds = {i for i, value in enumerate(scores)
                                 if value >= max(MIN_SEED_SCORE, highest - 0.12)}
                        guards = {i for i, sentence in enumerate(sentences) if _protected(sentence)}
                        guard_indices = sorted(guards)
                        keep = {0, *seeds, *guards}
                        for i in list(keep):
                            keep.update(j for j in (i - 1, i + 1) if 0 <= j < len(ranges))
                        for i in sorted(keep):
                            previous = i
                            while previous > 0 and _ANAPHORA.search(sentences[previous]):
                                previous -= 1
                                keep.add(previous)
                        omitted_indices = set(range(len(ranges))) - keep
                        ambiguous = any(scores[i] > MAX_OMITTED_SCORE for i in omitted_indices)
                        ambiguous = ambiguous or any(
                            _UNRESOLVED_REFERENCE.search(sentences[i]) and any(j not in keep for j in range(i))
                            for i in keep)
                        if ambiguous:
                            status = reason = "fallback_ambiguous_context"
                        else:
                            proposed = _merge([ranges[i] for i in sorted(keep)])
                            rendered = _render(source, proposed)
                            saved = len(source) - len(rendered)
                            if not omitted_indices or saved < max(MIN_SAVED_CHARS, len(source) * MIN_SAVED_FRACTION):
                                status = reason = "unchanged_no_gain"
                            else:
                                selected_ranges, status, reason = proposed, "extracted", "extracted"
        view = _render(source, selected_ranges)
        record: dict[str, Any] = {
            "schema": 1, "input_index": index, "status": status, "reason": reason,
            "source_identity": source_identity(document),
            "identity": {key: deepcopy(document.metadata.get(key)) for key in
                         ("group_id", "doc_id", "chunk_id", "chunk_index", "version", "filename")},
            "source_chunk_hash": _hash(source), "view_hash": _hash(view),
            "offset_unit": "unicode_code_point", "interval": "half_open",
            "selected_spans": _spans(source, selected_ranges),
            "omitted_spans": _spans(source, _complement(len(source), selected_ranges)),
            "scorer": scorer_name, "score_type": score_type,
            "score_purpose": "sentence_selection_only_not_answer_confidence",
            "sentence_scores": scores, "sentence_count": len(ranges) if ranges else None,
            "protected_sentence_indices": guard_indices,
            "source_chars": len(source), "view_chars": len(view),
            "coverage_limit": {"max_blocks": block_limit, "max_sentences_per_block": sentence_limit,
                               "processed": index in processing, "full_evidence_coverage_claimed": False},
            "guard_scope": "conservative_lexical_and_adjacent_context_not_semantic_proof",
        }
        if not validate_extractive_view(source, view, record):
            selected_ranges = [(0, len(source))] if source else []
            view = source
            record.update(status="fallback_validation_failed", reason="span_or_hash_validation_failed",
                          selected_spans=_spans(source, selected_ranges), omitted_spans=[],
                          view_hash=_hash(source), view_chars=len(source))
        output = deepcopy(document)
        output.page_content = view
        output.metadata = deepcopy(document.metadata or {})
        output.metadata["source_text"] = source
        output.metadata["extractive_view"] = deepcopy(record)
        if record["status"] == "extracted":
            view_identity = json.dumps({"identity": record["identity"], "source_chunk_hash": record["source_chunk_hash"],
                                        "view_hash": record["view_hash"], "selected_spans": record["selected_spans"]},
                                       ensure_ascii=False, sort_keys=True, default=str)
            output.metadata.update({"evidence_view": "extractive_text", "view_id": "ev:" + _hash(view_identity),
                                    "source_chunk_hash": record["source_chunk_hash"], "view_hash": record["view_hash"],
                                    "selected_spans": deepcopy(record["selected_spans"]),
                                    "omitted_spans": deepcopy(record["omitted_spans"]),
                                    "compression_status": record["status"],
                                    "coverage_limit": "仅覆盖所列原句；省略内容不能用于推断，适用条件以保留原句为准。",
                                    "offset_unit": "unicode_code_point", "interval": "half_open"})
        outputs.append(output)
        records.append(record)
    return ExtractiveViewResult(outputs, records, coverage, scorer_name, score_type)
