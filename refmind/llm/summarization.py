"""文档自动摘要生成。

从解析后文档的前若干个分块中提取内容，生成约 200 字的中文摘要，
作为该文献的主要内容预览存入数据库。
"""

from __future__ import annotations

from langchain_core.documents import Document

from .factory import get_llm

# 送入模型的最大字符数，避免超出上下文
_MAX_CHARS = 6000

_SUMMARY_INSTRUCTION = (
    "请为以下学术文献内容生成一个约200字的中文摘要，"
    "突出研究目的、方法和主要结论，语言简洁专业：\n\n"
)


def generate_summary(documents: list[Document], max_chunks: int = 5) -> str:
    """从前 ``max_chunks`` 个分块生成简洁的中文摘要。"""
    if not documents:
        return ""
    content = "\n".join(d.page_content for d in documents[:max_chunks])
    return generate_summary_from_text(content)


def generate_summary_from_text(text: str) -> str:
    """直接对原始文本生成摘要。"""
    if not text.strip():
        return ""
    prompt = f"{_SUMMARY_INSTRUCTION}{text[:_MAX_CHARS]}"
    return get_llm(temperature=0.2).invoke(prompt).content
