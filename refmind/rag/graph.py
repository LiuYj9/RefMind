"""RAG 对话流程：retrieve -> generate -> END。

retrieve 一步走完"混合召回 → 重排精排 → 上下文压缩"：先混合检索拿到一批候选，
再用重排模型精排，最后压缩去冗余得到最终上下文。generate 拼提示词调模型作答，
检索不到内容时直接返回固定话术，避免模型自由发挥。历史消息在图外组装后经 state 传入。
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from ..llm import get_llm
from ..rag.compression import compress_context
from ..rag.memory import RelevantMemory
from ..rag.reranker import rerank
from ..rag.retrieval import get_retriever

NO_CONTEXT_REPLY = "已检索历史文档，暂未找到相关内容，无法回答"


def get_system_prompt() -> str:
    return """你是一个专业的文献知识库助手。你需要严格遵循以下规则：
1. 仅基于用户上传的论文PDF内容回答问题。提供的"参考文档"是从用户上传论文中检索到的片段。
2. 如果参考文档中没有相关信息，或者信息不足以支撑答案，请直接回复："已检索历史文档，暂未找到相关内容，无法回答"，不要编造或使用外部知识。
3. 回答时应尽可能引用来源（如文档名、页码），并将回答组织得清晰、准确。
4. 回答使用中文，但若涉及专业术语可保留英文。
5. 保持礼貌、专业，不要与用户进行无关的闲聊。"""


def format_documents(documents: list[Document]) -> str:
    """把检索分块渲染成带来源标签的文本，便于引用溯源。"""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        source = meta.get("filename", "未知文档")
        page = meta.get("page", "?")
        label = f"[片段{i} | 来源: {source} | 第{page}页"
        section = meta.get("section")
        if section:
            label += f" | 章节: {section}"
        label += "]"
        blocks.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(blocks)


class GraphState(TypedDict, total=False):
    question: str
    group_id: int
    history: list[BaseMessage]
    documents: list[Document]
    answer: str


def _retrieve_node(state: GraphState) -> GraphState:
    question = state["question"]
    retriever = get_retriever(state["group_id"])
    candidates: list[Document] = []
    if retriever is not None:
        candidates = retriever.invoke(question)
    if not candidates:
        return {"documents": []}
    # 混合召回 -> 重排精排 -> 上下文压缩
    reranked = rerank(question, candidates)
    documents = compress_context(question, reranked)
    return {"documents": documents}


def _generate_node(state: GraphState) -> GraphState:
    documents = state.get("documents") or []
    if not documents:
        return {"answer": NO_CONTEXT_REPLY}

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_system_prompt()),
            ("placeholder", "{history}"),
            ("human", "问题：{question}\n\n参考文档：\n{documents}"),
        ]
    )
    chain = prompt | get_llm()
    response = chain.invoke(
        {
            "history": state.get("history", []),
            "question": state["question"],
            "documents": format_documents(documents),
        }
    )
    return {"answer": response.content}


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", _retrieve_node)
    workflow.add_node("generate", _generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def answer_question(
    question: str, group_id: int, memory: RelevantMemory | None = None
) -> dict[str, Any]:
    """跑一次完整问答，返回 {"answer", "documents"}；传入 memory 则记录本轮。"""
    history: list[BaseMessage] = []
    if memory is not None:
        history = memory.get_relevant_history(question)

    graph = _get_graph()
    result = graph.invoke(
        {"question": question, "group_id": group_id, "history": history}
    )

    answer = result.get("answer", NO_CONTEXT_REPLY)
    if memory is not None:
        memory.add_user_message(question)
        memory.add_ai_message(answer)

    return {"answer": answer, "documents": result.get("documents", [])}
