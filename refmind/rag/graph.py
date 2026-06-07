"""RefMind 的 LangGraph 对话流水线（对应 LangChain 1.0 的 LangGraph 1.0）。

状态图：``retrieve`` -> ``generate`` -> END。

* ``retrieve``：对当前组执行混合检索，得到参考文档。
* ``generate``：构造受约束的提示词并调用对话模型；若未检索到文档，
  则直接返回固定的“未找到”话术。

记忆（相关历史）在图外组装后通过状态传入，保持节点为纯函数（仅依赖 state），
符合 LangGraph 1.0 的推荐用法。
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from ..llm import get_llm
from ..rag.memory import RelevantMemory
from ..rag.retrieval import get_retriever

# 检索不到相关内容时的固定回复
NO_CONTEXT_REPLY = "已检索历史文档，暂未找到相关内容，无法回答"


def get_system_prompt() -> str:
    """返回系统提示词。"""
    return """你是一个专业的文献知识库助手。你需要严格遵循以下规则：
1. 仅基于用户上传的论文PDF内容回答问题。提供的"参考文档"是从用户上传论文中检索到的片段。
2. 如果参考文档中没有相关信息，或者信息不足以支撑答案，请直接回复："已检索历史文档，暂未找到相关内容，无法回答"，不要编造或使用外部知识。
3. 回答时应尽可能引用来源（如文档名、页码），并将回答组织得清晰、准确。
4. 回答使用中文，但若涉及专业术语可保留英文。
5. 保持礼貌、专业，不要与用户进行无关的闲聊。"""


def format_documents(documents: list[Document]) -> str:
    """将检索到的分块渲染为带来源标签的文本，用于约束生成与溯源。"""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        source = meta.get("filename", "未知文档")
        page = meta.get("page", "?")
        blocks.append(
            f"[片段{i} | 来源: {source} | 第{page}页]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


class GraphState(TypedDict, total=False):
    """状态图的状态结构。"""

    question: str
    group_id: int
    history: list[BaseMessage]
    documents: list[Document]
    answer: str


def _retrieve_node(state: GraphState) -> GraphState:
    """检索节点：基于组 ID 获取混合检索结果。"""
    retriever = get_retriever(state["group_id"])
    documents: list[Document] = []
    if retriever is not None:
        documents = retriever.invoke(state["question"])
    return {"documents": documents}


def _generate_node(state: GraphState) -> GraphState:
    """生成节点：构造提示词并调用对话模型。"""
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
    """构建并编译状态图。"""
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", _retrieve_node)
    workflow.add_node("generate", _generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


# 惰性编译的全局图实例
_GRAPH = None


def _get_graph():
    """返回缓存的已编译图。"""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def answer_question(
    question: str, group_id: int, memory: RelevantMemory | None = None
) -> dict[str, Any]:
    """执行一次完整的 RAG 流程。

    返回 ``{"answer": str, "documents": list[Document]}``；
    当提供记忆对象时，会将本轮问答写入记忆。
    """
    # 在图外组装相关历史，保持图节点为纯函数
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
