"""文本翻译。可选传入文献片段作为术语参考，让译文风格更贴近领域用语。"""

from __future__ import annotations

from .factory import get_llm

_TRANSLATE_SYSTEM = (
    "你是一个专业的学术文献翻译助手。请准确、流畅地翻译用户提供的文本，"
    "保持专业术语的准确性，不要添加解释或评论，只输出译文。"
)


def _build_prompt(text: str, target_language: str, context: str = "") -> str:
    parts = [_TRANSLATE_SYSTEM]
    if context.strip():
        parts.append(
            "以下是来自相关文献的参考片段，请在翻译时尽量与其中的专业术语、"
            "表述风格保持一致：\n" + context.strip()
        )
    parts.append(f"请将以下文本翻译为{target_language}：\n\n{text}")
    return "\n\n".join(parts)


def translate(text: str, target_language: str = "中文", context: str = "") -> str:
    if not text.strip():
        return ""
    llm = get_llm(temperature=0.0)
    return llm.invoke(_build_prompt(text, target_language, context)).content


def stream_translate(text: str, target_language: str = "中文", context: str = ""):
    if not text.strip():
        return
    llm = get_llm(temperature=0.0)
    for chunk in llm.stream(_build_prompt(text, target_language, context)):
        if chunk.content:
            yield chunk.content
