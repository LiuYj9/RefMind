"""在不公开原始 PDF 路径的前提下，渲染引用页并尽力定位证据段落。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PdfLocationPreview:
    """供 Streamlit 展示的引用页预览。"""

    image: bytes
    page_number: int
    highlighted: bool


def _metadata_bbox(value: object) -> list[float] | None:
    raw: Any = value
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        numbers = [float(item) for item in raw]
    except (TypeError, ValueError):
        return None
    if numbers[2] <= numbers[0] or numbers[3] <= numbers[1]:
        return None
    return numbers


def _search_evidence_rects(page: Any, evidence_text: str) -> list[Any]:
    """用较稳定的文本窗口查找证据；扫描 PDF 或排版差异时允许无匹配降级。"""
    candidates: list[str] = []
    for raw_line in str(evidence_text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if len(line) < 12:
            continue
        candidates.extend((line[:180], line[:110], line[:70]))
        if len(candidates) >= 12:
            break
    for candidate in dict.fromkeys(candidates):
        try:
            rectangles = list(page.search_for(candidate) or [])
        except Exception:  # noqa: BLE001 - 不同 PyMuPDF/PDF 字体映射可能拒绝搜索
            rectangles = []
        if rectangles:
            return rectangles[:6]
    return []


def render_pdf_location(
    pdf_path: str | Path,
    page_number: int,
    *,
    evidence_text: str = "",
    bbox: object = None,
) -> PdfLocationPreview:
    """渲染 1-based 目标页；有 bbox/文本匹配时裁剪并高亮证据区域。"""
    import fitz

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在：{path}")
    requested_page = max(1, int(page_number or 1))
    with fitz.open(path) as pdf:
        if pdf.page_count <= 0:
            raise ValueError("PDF 不包含可渲染页面。")
        if requested_page > pdf.page_count:
            raise ValueError(
                f"引用页码超出 PDF 范围：第{requested_page}页 / 共{pdf.page_count}页。"
            )
        page = pdf[requested_page - 1]
        page_rect = page.rect
        rectangles: list[Any] = []

        parsed_bbox = _metadata_bbox(bbox)
        if parsed_bbox is not None:
            candidate = fitz.Rect(*parsed_bbox) & page_rect
            if not candidate.is_empty and not candidate.is_infinite:
                rectangles = [candidate]
        if not rectangles and evidence_text:
            rectangles = _search_evidence_rects(page, evidence_text)

        for rectangle in rectangles:
            try:
                page.add_highlight_annot(rectangle)
            except Exception:  # noqa: BLE001 - 图片块等区域可能不接受文本高亮注释
                pass

        clip = page_rect
        if rectangles:
            focus = fitz.Rect(rectangles[0])
            for rectangle in rectangles[1:]:
                focus.include_rect(rectangle)
            margin_x = max(24.0, focus.width * 0.12)
            margin_y = max(48.0, focus.height * 1.2)
            clip = fitz.Rect(
                focus.x0 - margin_x,
                focus.y0 - margin_y,
                focus.x1 + margin_x,
                focus.y1 + margin_y,
            ) & page_rect

        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(1.8, 1.8),
            clip=clip,
            alpha=False,
            annots=True,
        )
        return PdfLocationPreview(
            image=pixmap.tobytes("png"),
            page_number=requested_page,
            highlighted=bool(rectangles),
        )
