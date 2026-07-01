"""文档自动摘要。"""

from __future__ import annotations

from langchain_core.documents import Document

from .factory import get_llm

_MAX_CHARS = 6000

_SUMMARY_INSTRUCTION = (
    "请为以下学术文献内容生成一个约200字的中文摘要，"
    "突出研究目的、方法和主要结论，语言简洁专业：\n\n"
)


def generate_summary(documents: list[Document], max_chunks: int = 5) -> str:
    if not documents:
        return ""
    content = "\n".join(d.page_content for d in documents[:max_chunks])
    return generate_summary_from_text(content)


def generate_summary_from_text(text: str) -> str:
    if not text.strip():
        return ""
    prompt = f"{_SUMMARY_INSTRUCTION}{text[:_MAX_CHARS]}"
    return get_llm(temperature=0.2).invoke(prompt).content
