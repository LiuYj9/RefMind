"""文献翻译工具（独立于 RAG 问答流程）。

支持可选的「文献上下文」：传入从历史输入文献中检索到的片段，提示模型在翻译时
对齐专业术语，从而获得更准确、风格一致的译文。
"""

from __future__ import annotations

from .factory import get_llm

# 翻译系统提示词
_TRANSLATE_SYSTEM = (
    "你是一个专业的学术文献翻译助手。请准确、流畅地翻译用户提供的文本，"
    "保持专业术语的准确性，不要添加解释或评论，只输出译文。"
)


def _build_prompt(text: str, target_language: str, context: str = "") -> str:
    """构造翻译提示词；提供 context 时要求模型对齐其中的术语。"""
    parts = [_TRANSLATE_SYSTEM]
    if context.strip():
        parts.append(
            "以下是来自相关文献的参考片段，请在翻译时尽量与其中的专业术语、"
            "表述风格保持一致：\n" + context.strip()
        )
    parts.append(f"请将以下文本翻译为{target_language}：\n\n{text}")
    return "\n\n".join(parts)


def translate(text: str, target_language: str = "中文", context: str = "") -> str:
    """将 ``text`` 翻译为 ``target_language``，可选传入文献 ``context`` 对齐术语。"""
    if not text.strip():
        return ""
    llm = get_llm(temperature=0.0)
    return llm.invoke(_build_prompt(text, target_language, context)).content


def stream_translate(text: str, target_language: str = "中文", context: str = ""):
    """以流式方式逐块返回译文，适配前端流式展示。"""
    if not text.strip():
        return
    llm = get_llm(temperature=0.0)
    for chunk in llm.stream(_build_prompt(text, target_language, context)):
        if chunk.content:
            yield chunk.content
