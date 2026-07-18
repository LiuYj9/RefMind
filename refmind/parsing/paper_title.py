"""从解析结果与 PDF 首页恢复论文原始题名，上传文件名仅作为最后兜底。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

_SPACE_RE = re.compile(r"\s+")
_MARKDOWN_RE = re.compile(r"^#{1,6}\s*")
_SECTION_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.、]?\s*)?"
    r"(?:abstract|摘要|introduction|引言|绪论|contents?|目录|"
    r"references?|参考文献|acknowledg(?:e)?ments?|致谢|"
    r"methods?|方法|results?|结果|conclusions?|结论)\s*$",
    re.I,
)
_NOISE_RE = re.compile(
    r"(?:doi\s*:|https?://|www\.|@|issn|isbn|arxiv|"
    r"transactions\s+on|proceedings\s+of|journal\s+of|"
    r"volume\s+\d+|vol\.\s*\d+|copyright|all rights reserved|"
    r"microsoft\s+word\s*-)",
    re.I,
)


def _clean(value: object) -> str:
    text = _MARKDOWN_RE.sub("", str(value or "").strip())
    return _SPACE_RE.sub(" ", text).strip(" \t\r\n-–—|_")


def _is_plausible(text: str) -> bool:
    if not (6 <= len(text) <= 320):
        return False
    if _SECTION_RE.fullmatch(text) or _NOISE_RE.search(text):
        return False
    alnum = sum(char.isalnum() for char in text)
    if alnum < 5 or alnum / max(1, len(text)) < 0.35:
        return False
    return True


def _parsed_candidates(parsed: dict[str, Any]) -> Iterable[tuple[float, str]]:
    explicit = _clean(parsed.get("paper_title"))
    if _is_plausible(explicit):
        yield 200.0, explicit

    blocks = [
        block
        for block in (parsed.get("blocks") or [])
        if isinstance(block, dict)
    ]
    blocks.sort(
        key=lambda block: (
            int(block.get("page") or 1),
            float(block.get("reading_order") or 0),
        )
    )
    headings: list[str] = []
    for order, block in enumerate(blocks):
        if int(block.get("page") or 1) != 1:
            continue
        kind = str(block.get("type") or "").lower()
        if "title" not in kind and "heading" not in kind:
            continue
        text = _clean(block.get("text"))
        if _is_plausible(text):
            headings.append(text)
            yield 120.0 - min(order, 20), text
    if 1 < len(headings) <= 3:
        combined = _clean(" ".join(headings))
        if _is_plausible(combined):
            yield 126.0, combined

    pages = parsed.get("pages") or []
    first_page = next(
        (
            page
            for page in pages
            if isinstance(page, dict) and int(page.get("page") or 1) == 1
        ),
        None,
    )
    if not first_page:
        return
    lines = [_clean(line) for line in str(first_page.get("text") or "").splitlines()]
    candidates = [line for line in lines[:20] if _is_plausible(line)]
    for position, text in enumerate(candidates[:8]):
        # 首页越靠前越可信，但低于结构化 heading 和字体信息。
        yield 55.0 - position * 2.0 + min(len(text), 120) / 30.0, text


def _pdf_candidates(pdf_path: Path) -> Iterable[tuple[float, str]]:
    if not pdf_path.is_file():
        return
    try:
        import fitz

        with fitz.open(pdf_path) as pdf:
            metadata_title = _clean((pdf.metadata or {}).get("title"))
            if _is_plausible(metadata_title):
                yield 90.0 + min(len(metadata_title), 120) / 30.0, metadata_title
            if not pdf:
                return
            page = pdf[0]
            page_height = max(1.0, float(page.rect.height))
            lines: list[tuple[float, float, str]] = []
            for block in page.get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    spans = line.get("spans") or []
                    text = _clean("".join(str(span.get("text") or "") for span in spans))
                    if not spans or not _is_plausible(text):
                        continue
                    size = max(float(span.get("size") or 0.0) for span in spans)
                    bbox = line.get("bbox") or block.get("bbox") or [0, 0, 0, 0]
                    y0 = float(bbox[1]) if len(bbox) >= 2 else 0.0
                    if y0 <= page_height * 0.65:
                        lines.append((size, y0, text))
            if not lines:
                return
            max_size = max(size for size, _y, _text in lines)
            large_lines: list[tuple[float, float, str]] = []
            for size, y0, text in sorted(lines, key=lambda item: item[1]):
                relative = size / max(1.0, max_size)
                if relative < 0.78:
                    continue
                large_lines.append((size, y0, text))
                score = 100.0 + relative * 12.0 - (y0 / page_height) * 8.0
                yield score + min(len(text), 120) / 30.0, text
            if 1 < len(large_lines) <= 4:
                combined = _clean(" ".join(text for _size, _y, text in large_lines))
                if _is_plausible(combined):
                    relative = max(size for size, _y, _text in large_lines) / max_size
                    yield 114.0 + relative * 8.0, combined
    except Exception:  # noqa: BLE001 - 题名增强失败不得阻断 PDF 正文入库
        return


def extract_paper_title(
    parsed: dict[str, Any],
    pdf_path: str | Path,
    fallback_filename: str,
) -> str:
    """返回最可信的论文题名；无法恢复时使用不带扩展名的上传文件名。"""
    fallback = _clean(Path(fallback_filename).stem) or "未命名论文"
    candidates: list[tuple[float, str]] = []
    try:
        candidates.extend(_parsed_candidates(parsed))
    except Exception:  # noqa: BLE001 - 非标准插件解析结果回退到 PDF/文件名
        pass
    try:
        candidates.extend(_pdf_candidates(Path(pdf_path)))
    except Exception:  # pragma: no cover - _pdf_candidates 已有内部防御
        pass
    if not candidates:
        return fallback

    # 与上传文件名完全相同的 metadata 很可能只是导出工具写入的文件名，适度降权。
    normalized_fallback = fallback.casefold()
    ranked = [
        (
            score - (35.0 if text.casefold() == normalized_fallback else 0.0),
            text,
        )
        for score, text in candidates
    ]
    ranked.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return ranked[0][1]
