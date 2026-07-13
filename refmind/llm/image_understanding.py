"""图片摘要和多模态消息组装。

图片只在两种时机发送到全模态模型：入库时生成摘要，以及检索命中其摘要后
回答问题。所有文件路径均须位于本地 docstore 内，避免 metadata 被篡改后读取
任意文件。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import HumanMessage

from ..config import settings
from .factory import get_multimodal_llm


def summarize_image(path: str | Path, *, caption: str = "") -> str:
    """使用 qwen3.5-omni 输出用于文本检索的、克制的结构化图片摘要。"""
    image = _data_url(path)
    if not image:
        return ""
    prompt = (
        "请为论文图片生成可用于语义检索的结构化中文摘要。只描述图中可见内容，"
        "不要补充论文外知识或猜测数值。输出包含：图片类型、主体/流程或趋势、"
        "关键标签与关系。若图中文字无法辨认请明确说明。"
    )
    if caption:
        prompt += f"\n关联图注（仅作辅助，不要照抄或扩写）：{caption}"
    response = get_multimodal_llm(temperature=0.0).invoke(
        [HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image}}])]
    )
    content = response.content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content).strip()
    return str(content or "").strip()


def build_visual_content(
    documents: Iterable[Any], *, max_images: int | None = None
) -> list[dict[str, Any]]:
    """从命中的图片摘要块加载有限数量的原图，返回 OpenAI 兼容 content items。"""
    limit = settings.image_max_per_answer if max_images is None else max_images
    content: list[dict[str, Any]] = []
    seen: set[str] = set()
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        path = metadata.get("image_path")
        if not path or str(path) in seen:
            continue
        data_url = _data_url(path)
        if not data_url:
            continue
        seen.add(str(path))
        content.append({"type": "image_url", "image_url": {"url": data_url}})
        if len(content) >= limit:
            break
    return content


def _data_url(path: str | Path) -> str:
    try:
        candidate = Path(path).resolve()
        root = settings.docstore_dir.resolve()
        candidate.relative_to(root)  # 防止向量 metadata 诱导任意文件读取。
        if not candidate.is_file() or candidate.stat().st_size > settings.image_max_bytes:
            return ""
        raw = candidate.read_bytes()
    except (OSError, ValueError):
        return ""
    if not raw:
        return ""
    mime = _mime_type(candidate)
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _mime_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.lower(), "image/png")
