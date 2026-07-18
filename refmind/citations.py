"""论文证据引用：动态补齐旧索引 metadata，并生成可自解释的稳定标签。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode, urlparse

from langchain_core.documents import Document

from . import storage

_FRAGMENT_CITATION_RE = re.compile(
    r"\[\s*片段\s*(\d+)(?:\s*\|[^\]\r\n]*)?\s*]",
    re.I,
)
_SPACE_RE = re.compile(r"\s+")
_CITATION_TARGET_RE = re.compile(
    r"^g(?P<group_id>\d+)-d(?P<doc_id>\d+)-p(?P<page>\d+)-r(?P<paragraph>\d+)$"
)
_FULL_CITATION_RE = re.compile(
    r"\[文献(?P<library_index>\d+)《(?P<title>[^\]\r\n]+?)》，"
    r"(?P<location>[^\]\r\n]+)]"
)
_VISIBLE_COMPACT_LINK_RE = re.compile(
    r"\[\[(?P<number>\d+)]]\("
    r"(?:\?open_citation=[^)\s\"]+|https?://[^)\s\"]+|#参考来源)"
    r'(?:\s+"(?:\\.|[^"\\])*")?\)',
    re.I,
)
_LEGACY_CITATION_LINK_PREFIX_RE = re.compile(
    r"\[(?P<number>\d+)]"
    r"(?=\((?:\?open_citation=|https?://[^)\s]*[?&]open_citation=))",
    re.I,
)
_BARE_CITATION_RE = re.compile(
    r"(?<!\[)\[(?P<position>\d+)](?!\]|\()"
)
_LEGACY_REFERENCE_MARKER = "<!-- refmind-reference-list -->"
_REFERENCE_HEADING = "### 参考来源"
_REFERENCE_BLOCK_RE = re.compile(
    r"\n{2,}---\s*\n{2,}### 参考来源\s*(?:\n|$)"
)


def _remove_legacy_reference_marker(value: str) -> str:
    """清理旧版本写入回答的内部标记，避免 Markdown 渲染器将其显示出来。"""
    return str(value or "").replace(
        f"{_LEGACY_REFERENCE_MARKER}\n", ""
    ).replace(_LEGACY_REFERENCE_MARKER, "")


def _has_reference_block(value: str) -> bool:
    return _REFERENCE_BLOCK_RE.search(value) is not None


def _answer_body_without_references(value: str) -> str:
    """同时兼容旧标记和新无标记格式，返回供对话历史使用的正文。"""
    cleaned = _remove_legacy_reference_marker(value)
    match = _REFERENCE_BLOCK_RE.search(cleaned)
    return (cleaned[: match.start()] if match else cleaned).rstrip()


@dataclass(frozen=True)
class CitationTarget:
    """回答链接携带的最小定位信息；打开时仍需按文献库重新校验。"""

    group_id: int
    doc_id: int
    page: int
    paragraph: int


def _positive_int(value: object) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _paper_title(metadata: dict) -> str:
    value = metadata.get("paper_title")
    if value:
        title = str(value)
    else:
        title = Path(str(metadata.get("filename") or "未知论文")).stem
    # 防止题名换行或方括号破坏引用标签边界。
    return _SPACE_RE.sub(" ", title).strip().replace("[", "（").replace("]", "）")


def _is_external_academic(metadata: dict) -> bool:
    return str(metadata.get("evidence_origin") or "").casefold() == "academic_search"


def _external_evidence_label(metadata: dict) -> str:
    level = str(metadata.get("evidence_level") or "abstract").strip().casefold()
    return {
        "abstract": "摘要",
        "search_snippet": "检索片段",
        "full_text": "开放全文",
    }.get(level, "外部检索内容")


def _safe_external_url(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except ValueError:
        return ""
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        return ""
    return raw


def _paper_number(metadata: dict, fallback_position: int) -> str:
    if _is_external_academic(metadata):
        return str(max(1, fallback_position))
    library_index = _positive_int(metadata.get("library_index"))
    doc_id = _positive_int(metadata.get("doc_id"))
    if library_index is not None:
        return str(library_index)
    if doc_id is not None:
        return f"ID{doc_id}"
    return f"?{max(1, fallback_position)}"


def _citation_location_parts(metadata: dict) -> list[str]:
    if _is_external_academic(metadata):
        year = _positive_int(metadata.get("publication_year"))
        provider = _SPACE_RE.sub(
            " ", str(metadata.get("provider") or "开放学术索引")
        ).strip()
        return [
            str(year) if year is not None else "年份未知",
            provider,
            _external_evidence_label(metadata),
        ]

    page_start = _positive_int(
        metadata.get("page_start") or metadata.get("page")
    )
    page_end = _positive_int(metadata.get("page_end"))
    if page_start is None:
        page_text = "页码未知"
    elif page_end is not None and page_end > page_start:
        page_text = f"第{page_start}–{page_end}页"
    else:
        page_text = f"第{page_start}页"

    parts = [page_text]
    paper_title = _paper_title(metadata)
    section = str(
        metadata.get("section_path") or metadata.get("section") or ""
    ).strip()
    if (
        section
        and section != "正文"
        and section.casefold() != paper_title.casefold()
    ):
        parts.append(f"§{_SPACE_RE.sub(' ', section)}")
    paragraph = _positive_int(
        metadata.get("paragraph_index")
        or (
            int(metadata["chunk_index"]) + 1
            if str(metadata.get("chunk_index", "")).isdigit()
            else None
        )
    )
    if paragraph is not None:
        parts.append(f"段落{paragraph}")
    return parts


def _paper_key(document: Document, fallback_position: int) -> tuple[str, str]:
    metadata = document.metadata or {}
    if _is_external_academic(metadata):
        doi = str(metadata.get("doi") or "").strip().casefold()
        evidence_id = str(metadata.get("evidence_id") or "").strip().casefold()
        return "academic", doi or evidence_id or _paper_title(metadata).casefold()
    doc_id = _positive_int(metadata.get("doc_id"))
    if doc_id is not None:
        return "doc_id", str(doc_id)
    group_id = _positive_int(metadata.get("group_id"))
    library_index = _positive_int(metadata.get("library_index"))
    if library_index is not None:
        return "library", f"{group_id or 0}:{library_index}"
    source = str(metadata.get("filename") or metadata.get("source") or "").casefold()
    return "source", source or f"position:{fallback_position}"


def _escape_markdown_text(value: str) -> str:
    """转义题名/章节中的 Markdown 控制字符，避免破坏参考来源列表。"""
    return re.sub(r"([\\`*_{}\[\]()<>#+\-.!|])", r"\\\1", value)


def _escape_markdown_title(value: str) -> str:
    return _SPACE_RE.sub(" ", value).replace("\\", "\\\\").replace('"', '\\"')


def _external_paper_reference(metadata: dict) -> str:
    parts = [f"外部论文《{_paper_title(metadata)}》"]
    authors = [
        item.strip()
        for item in str(metadata.get("authors") or "").split(",")
        if item.strip()
    ]
    if authors:
        author_text = "、".join(authors[:3]) + (" 等" if len(authors) > 3 else "")
        parts.append(author_text)
    venue = _SPACE_RE.sub(" ", str(metadata.get("venue") or "")).strip()
    if venue:
        parts.append(venue)
    doi = _SPACE_RE.sub(" ", str(metadata.get("doi") or "")).strip()
    if doi:
        parts.append(f"DOI: {doi}")
    return "｜".join(parts)


def citation_target_for_document(document: Document) -> CitationTarget | None:
    metadata = document.metadata or {}
    group_id = _positive_int(metadata.get("group_id"))
    doc_id = _positive_int(metadata.get("doc_id"))
    if group_id is None or doc_id is None:
        return None
    page = _positive_int(metadata.get("page_start") or metadata.get("page")) or 0
    paragraph = _positive_int(metadata.get("paragraph_index")) or 0
    return CitationTarget(group_id, doc_id, page, paragraph)


def encode_citation_target(target: CitationTarget) -> str:
    return (
        f"g{max(0, int(target.group_id))}-d{max(0, int(target.doc_id))}-"
        f"p{max(0, int(target.page))}-r{max(0, int(target.paragraph))}"
    )


def decode_citation_target(value: object) -> CitationTarget | None:
    match = _CITATION_TARGET_RE.fullmatch(str(value or "").strip())
    if match is None:
        return None
    target = CitationTarget(**{key: int(raw) for key, raw in match.groupdict().items()})
    if target.group_id <= 0 or target.doc_id <= 0:
        return None
    return target


def build_citation_href(document: Document) -> str:
    metadata = document.metadata or {}
    if _is_external_academic(metadata):
        return _safe_external_url(metadata.get("external_url")) or "#参考来源"
    target = citation_target_for_document(document)
    if target is None:
        return "#参考来源"
    return "?" + urlencode({"open_citation": encode_citation_target(target)})


def enrich_citation_metadata(
    documents: Iterable[Document], group_id: int
) -> list[Document]:
    """用 SQLite 文档表补齐旧 Chroma 块的题名与库内序号。"""
    selected = list(documents)
    try:
        rows = storage.list_documents(group_id)
    except Exception:  # noqa: BLE001 - 引用增强失败不能让论文问答不可用
        rows = []
    by_id = {int(row.id): row for row in rows}

    enriched: list[Document] = []
    for document in selected:
        metadata = dict(document.metadata or {})
        doc_id = _positive_int(metadata.get("doc_id"))
        row = by_id.get(doc_id) if doc_id is not None else None
        if row is not None:
            metadata["paper_title"] = row.paper_title or Path(row.filename).stem
            metadata["library_index"] = int(row.library_index or 0)
            metadata["filename"] = row.filename
        if _positive_int(metadata.get("paragraph_index")) is None:
            chunk_index = metadata.get("chunk_index")
            try:
                metadata["paragraph_index"] = int(chunk_index) + 1
            except (TypeError, ValueError):
                pass
        enriched.append(
            Document(page_content=document.page_content, metadata=metadata)
        )
    return enriched


def build_citation_label(document: Document, fallback_position: int = 1) -> str:
    """生成形如“文献2《题名》，第5页，§3.1，段落13”的完整标签。"""
    metadata = document.metadata or {}
    paper_title = _paper_title(metadata)
    paper_number = _paper_number(metadata, fallback_position)
    if _is_external_academic(metadata):
        parts = [
            f"GS文献{paper_number}《{paper_title}》",
            *_citation_location_parts(metadata),
        ]
        return "[" + "，".join(parts) + "]"
    parts = [
        f"文献{paper_number}《{paper_title}》",
        *_citation_location_parts(metadata),
    ]
    return "[" + "，".join(parts) + "]"


def normalize_answer_citations(answer: str, documents: list[Document]) -> str:
    """把模型遗留的旧式片段标签确定性替换为完整论文定位。"""
    labels = {
        index: build_citation_label(document, index)
        for index, document in enumerate(documents, start=1)
    }

    def _replace(match: re.Match[str]) -> str:
        return labels.get(int(match.group(1)), match.group(0))

    return _FRAGMENT_CITATION_RE.sub(_replace, str(answer or ""))


@dataclass(frozen=True)
class _CitationEvidence:
    document: Document
    position: int
    label: str
    paper_key: tuple[str, str]
    paper_reference: str
    location: str


def _citation_markdown(number: int, evidence: _CitationEvidence) -> str:
    href = build_citation_href(evidence.document)
    title = _escape_markdown_title(evidence.label)
    # 外层 [] 是 Markdown 链接语法，内层 [] 才是用户实际看到的论文引用括号。
    return f'[[{number}]]({href} "{title}")'


def ensure_visible_citation_brackets(answer: str) -> str:
    """把旧式 `[2](url)` 升级为视觉上显示 `[2]` 的 `[[2]](url)`。"""
    return _LEGACY_CITATION_LINK_PREFIX_RE.sub(
        lambda match: f"[[{match.group('number')}]]",
        str(answer or ""),
    )


def finalize_answer_citations(answer: str, documents: list[Document]) -> str:
    """把内部精确标签压缩成论文式编号，并在回答末尾追加聚合参考来源。"""
    body = _remove_legacy_reference_marker(
        ensure_visible_citation_brackets(
            normalize_answer_citations(answer, documents)
        )
    ).strip()
    if not body or _has_reference_block(body) or not documents:
        return body

    evidence_items: list[_CitationEvidence] = []
    label_map: dict[str, _CitationEvidence] = {}
    for position, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        paper_reference = (
            _external_paper_reference(metadata)
            if _is_external_academic(metadata)
            else f"文献{_paper_number(metadata, position)}《{_paper_title(metadata)}》"
        )
        evidence = _CitationEvidence(
            document=document,
            position=position,
            label=build_citation_label(document, position),
            paper_key=_paper_key(document, position),
            paper_reference=paper_reference,
            location="，".join(_citation_location_parts(metadata)),
        )
        evidence_items.append(evidence)
        label_map.setdefault(evidence.label, evidence)

    # 审校模型偶尔会提前把内部标签缩写成 [N]。在 UI 编号前先按证据位置恢复，
    # 后续仍由“正文首次出现的论文”重新连续编号并绑定精确页码。
    def _restore_bare_citation(match: re.Match[str]) -> str:
        position = int(match.group("position"))
        if 1 <= position <= len(evidence_items):
            return evidence_items[position - 1].label
        return match.group(0)

    body = _BARE_CITATION_RE.sub(_restore_bare_citation, body)

    paper_numbers: dict[tuple[str, str], int] = {}
    used: dict[tuple[str, str], list[_CitationEvidence]] = {}
    used_labels: dict[tuple[str, str], set[str]] = {}

    def _number_for(evidence: _CitationEvidence) -> int:
        if evidence.paper_key not in paper_numbers:
            paper_numbers[evidence.paper_key] = len(paper_numbers) + 1
        return paper_numbers[evidence.paper_key]

    def _use(evidence: _CitationEvidence) -> int:
        number = _number_for(evidence)
        labels = used_labels.setdefault(evidence.paper_key, set())
        if evidence.label not in labels:
            used.setdefault(evidence.paper_key, []).append(evidence)
            labels.add(evidence.label)
        return number

    labels = sorted(label_map, key=len, reverse=True)
    if labels:
        pattern = re.compile("|".join(re.escape(label) for label in labels))

        def _replace_label(match: re.Match[str]) -> str:
            evidence = label_map[match.group(0)]
            return _citation_markdown(_use(evidence), evidence)

        body = pattern.sub(_replace_label, body)

    # 极少数模型会遗漏正文引用。此时仍把本轮实际送入模型的证据完整列在末尾，
    # 但不猜测它们应挂在正文哪一句后面。
    if not used:
        for evidence in evidence_items:
            _use(evidence)

    reference_lines: list[str] = []
    for paper_key, number in sorted(paper_numbers.items(), key=lambda item: item[1]):
        locations = used.get(paper_key) or []
        if not locations:
            continue
        first = locations[0]
        link = _citation_markdown(number, first)
        paper_reference = _escape_markdown_text(first.paper_reference)
        unique_locations = list(dict.fromkeys(item.location for item in locations))
        location_text = "；".join(
            _escape_markdown_text(location) for location in unique_locations
        )
        separator = "，" if len(unique_locations) == 1 else "："
        reference_lines.append(
            f"- {link} {paper_reference}{separator}{location_text}"
        )

    if not reference_lines:
        return body
    return (
        f"{body}\n\n---\n\n{_REFERENCE_HEADING}\n\n"
        + "\n".join(reference_lines)
    )


def finalize_stored_answer_citations(answer: str, group_id: int) -> str:
    """为升级前已保存的完整标签回答补上紧凑链接；无法映射时保持原文。"""
    body = _remove_legacy_reference_marker(
        ensure_visible_citation_brackets(str(answer or ""))
    )
    if _has_reference_block(body):
        return body
    matches = list(_FULL_CITATION_RE.finditer(body))
    if not matches:
        return body
    try:
        rows = storage.list_documents(group_id)
    except Exception:  # noqa: BLE001 - 历史消息展示不能依赖迁移成功
        rows = []
    by_index = {int(row.library_index or 0): row for row in rows}
    documents: list[Document] = []
    seen_labels: set[str] = set()
    for match in matches:
        label = match.group(0)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        library_index = int(match.group("library_index"))
        location = match.group("location")
        page_match = re.search(r"第(\d+)(?:–(\d+))?页", location)
        paragraph_match = re.search(r"段落(\d+)", location)
        section_match = re.search(r"(?:^|，)§([^，]+)", location)
        row = by_index.get(library_index)
        metadata = {
            "group_id": group_id,
            "doc_id": int(row.id) if row is not None else -1,
            "library_index": library_index,
            "paper_title": match.group("title"),
            "filename": row.filename if row is not None else "",
            "page": int(page_match.group(1)) if page_match else None,
            "page_end": (
                int(page_match.group(2))
                if page_match and page_match.group(2)
                else None
            ),
            "section": section_match.group(1) if section_match else "正文",
            "paragraph_index": (
                int(paragraph_match.group(1)) if paragraph_match else None
            ),
        }
        document = Document(page_content="", metadata=metadata)
        if build_citation_label(document, len(documents) + 1) == label:
            documents.append(document)
    return finalize_answer_citations(body, documents) if documents else body


def strip_citation_rendering(answer: str) -> str:
    """供下一轮模型历史使用：去掉 UI 链接参数和文末书目，保留正文数字引用。"""
    body = _answer_body_without_references(
        ensure_visible_citation_brackets(answer)
    )
    return _VISIBLE_COMPACT_LINK_RE.sub(
        lambda match: f"[{match.group('number')}]",
        body,
    )
