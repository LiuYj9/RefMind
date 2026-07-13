"""面向学术论文版面的结构化切分器。

解析器负责提供按阅读顺序排列的 block；本模块保留章节、页码和内容类型，
只在单个语义单元过长时才使用字符级递归切分。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter


_HEADING_TYPES = {"title", "heading", "section_title"}
_ATOMIC_TYPES = {"table", "equation", "figure", "figure_caption", "table_caption"}
_TEXT_TYPES = {"text", "paragraph", "list", "list_item", "reference"}


@dataclass
class LayoutChunk:
    text: str
    page_start: int
    page_end: int
    section: str
    content_type: str
    block_ids: list[str] = field(default_factory=list)
    bbox: list[float] = field(default_factory=list)
    image_id: str = ""
    image_path: str = ""
    image_mime_type: str = ""


def _integer(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _ordered(blocks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """显式阅读顺序优先，缺失时保持解析器给出的原始顺序。"""
    indexed = list(enumerate(blocks))
    return [
        block
        for _, block in sorted(
            indexed,
            key=lambda pair: (
                _integer(pair[1].get("page")),
                float(pair[1].get("reading_order", pair[0])),
                pair[0],
            ),
        )
    ]


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=min(overlap, max_chars // 4),
        separators=["\n\n", "\n", "。", "！", "？", ". ", "; ", " ", ""],
    )
    return [part.strip() for part in splitter.split_text(text) if part.strip()]


def _normalize_type(block: dict[str, Any]) -> str:
    kind = str(block.get("type") or block.get("block_type") or "text").lower()
    if "table" in kind:
        return "table_caption" if "caption" in kind else "table"
    if "figure" in kind or "image" in kind:
        return "figure_caption" if "caption" in kind else "figure"
    if "equation" in kind or "formula" in kind:
        return "equation"
    if "title" in kind or "heading" in kind:
        return "heading"
    if "list" in kind:
        return "list_item"
    return kind if kind in _TEXT_TYPES else "paragraph"


def layout_chunks(
    blocks: list[dict[str, Any]], target_chars: int, max_chars: int, overlap: int
) -> list[LayoutChunk]:
    """按章节和版面块切分，返回可直接转换为 Document 的中间结果。"""
    chunks: list[LayoutChunk] = []
    section = "正文"
    pending: list[dict[str, Any]] = []

    def emit(items: list[dict[str, Any]], content_type: str = "paragraph") -> None:
        if not items:
            return
        text = "\n\n".join(str(item.get("text", "")).strip() for item in items).strip()
        if not text:
            return
        pages = [_integer(item.get("page")) for item in items]
        page_ends = [_integer(item.get("page_end"), page) for item, page in zip(items, pages)]
        ids = [str(item.get("block_id")) for item in items if item.get("block_id")]
        bbox = items[0].get("bbox") if len(items) == 1 else []
        for part in _split_long(text, max_chars, overlap):
            # 图片资产只属于单一 figure 原子块，普通段落不得继承其路径。
            source = items[0] if len(items) == 1 else {}
            chunks.append(
                LayoutChunk(
                    part, min(pages), max(page_ends), section, content_type, ids,
                    bbox or [], str(source.get("image_id") or ""),
                    str(source.get("image_path") or ""),
                    str(source.get("image_mime_type") or ""),
                )
            )

    def flush() -> None:
        nonlocal pending
        emit(pending)
        pending = []

    for block in _ordered(blocks):
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        kind = _normalize_type(block)
        if kind in _HEADING_TYPES:
            flush()
            section = re.sub(r"^#{1,6}\s+", "", text).strip() or section
            continue
        if kind in _ATOMIC_TYPES:
            flush()
            emit([block], kind)
            continue
        candidate_length = sum(len(str(item.get("text", ""))) for item in pending) + len(text)
        if pending and candidate_length > target_chars:
            flush()
        pending.append(block)
    flush()
    return chunks


def scalar_metadata(value: Any) -> str | int | float | bool:
    """Chroma metadata 不接受 list/dict，复杂版面字段统一序列化。"""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
