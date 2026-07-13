"""PDF 图片提取与本地 docstore 写入。

向量库不保存图片二进制：这里把原图写到按 ``doc_id`` 隔离的 docstore，
调用方只把其路径和视觉摘要写入索引。优先按 MinerU 的 figure bbox 裁剪，
因此矢量流程图也能作为一张可供视觉模型理解的图片保存下来。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def extract_pdf_figures(
    pdf_path: Path,
    *,
    docstore_dir: Path,
    doc_id: int,
    figure_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """提取论文图片并返回可回填到 layout block 的资产记录。

    无法从 MinerU 获得 figure block 时，仍会提取 PDF 的嵌入图片并创建
    synthetic block，确保 PyMuPDF 回退解析路径同样具备图片问答能力。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return []

    target_dir = docstore_dir / f"doc_{doc_id}" / "images"
    target_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}

    def persist(data: bytes, index: int, suffix: str = "png") -> Path | None:
        if not data:
            return None
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            return seen[digest]
        path = target_dir / f"figure_{index:03d}_{digest[:12]}.{suffix}"
        path.write_bytes(data)
        seen[digest] = path
        return path

    try:
        with fitz.open(pdf_path) as pdf:
            # 有版面块时，bbox 裁剪包含图片、矢量图和相邻图中文字，语义最完整。
            for index, block in enumerate(figure_blocks, start=1):
                page_no = _page_number(block.get("page"))
                if page_no > len(pdf):
                    continue
                data = _render_bbox(pdf[page_no - 1], block.get("bbox"))
                if data is None:
                    data = _first_embedded_image(pdf[page_no - 1])
                path = persist(data or b"", index)
                if path is None:
                    continue
                records.append(_record(block, path, page_no, index))

            # 没有 figure 布局时，至少保存 PDF 的 raster 图片；这条路径主要服务回退解析。
            if not figure_blocks:
                index = 0
                for page_no, page in enumerate(pdf, start=1):
                    for xref, *_ in page.get_images(full=True):
                        extracted = pdf.extract_image(xref)
                        data = extracted.get("image", b"")
                        path = persist(data, index + 1, _image_suffix(extracted))
                        if path is None:
                            continue
                        index += 1
                        records.append(
                            {
                                "block_id": f"figure_{page_no:04d}_{index:03d}",
                                "page": page_no,
                                "page_end": page_no,
                                "bbox": [],
                                "image_id": f"image_{index:03d}",
                                "image_path": str(path),
                                "image_mime_type": _mime_type(path),
                            }
                        )
    except Exception:
        # 图片增强不能让已有的 PDF 文本入库失败；调用方仍可索引图注。
        return records
    return records


def _record(block: dict[str, Any], path: Path, page_no: int, index: int) -> dict[str, Any]:
    return {
        "block_id": str(block.get("block_id") or f"figure_{page_no:04d}_{index:03d}"),
        "page": page_no,
        "page_end": _page_number(block.get("page_end"), page_no),
        "bbox": block.get("bbox") or [],
        "image_id": f"image_{index:03d}",
        "image_path": str(path),
        "image_mime_type": _mime_type(path),
    }


def _page_number(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _render_bbox(page: Any, bbox: Any) -> bytes | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        import fitz

        rect = page.rect
        clip = fitz.Rect(*(float(value) for value in bbox)) & rect
        if clip.is_empty or clip.width < 2 or clip.height < 2:
            return None
        # 2x 足以保留论文图中的坐标轴和标签，同时控制传给视觉模型的大小。
        return page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip).tobytes("png")
    except Exception:
        return None


def _first_embedded_image(page: Any) -> bytes | None:
    try:
        images = page.get_images(full=True)
        if not images:
            return None
        return page.parent.extract_image(images[0][0]).get("image")
    except Exception:
        return None


def _image_suffix(image: dict[str, Any]) -> str:
    ext = str(image.get("ext") or "png").lower()
    return ext if ext.isalnum() and len(ext) <= 5 else "png"


def _mime_type(path: Path) -> str:
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(path.suffix.lower().lstrip("."), "image/png")
