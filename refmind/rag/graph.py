"""RAG 对话流程：基础 LangGraph + 可降级的 multi-agent 增强。

基础图保留 ``retrieve -> generate``，便于评测和故障降级。启用 multi-agent 时，
规划角色生成少量互补子查询，检索角色并发召回，合并后只做一次重排与压缩，
最后由生成/审校角色输出答案。所有增强步骤都可回退到原单查询链路。
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from ..agents import AnswerDraft, MultiAgentConfig, MultiAgentOrchestrator
from ..config import settings
from ..llm import get_llm
from ..plugins import CoreHook, get_plugin_manager
from ..rag.compression import compress_context
from ..rag.memory import RelevantMemory
from ..rag.reranker import rerank
from ..rag.retrieval import get_retriever

NO_CONTEXT_REPLY = "已检索历史文档，暂未找到相关内容，无法回答"
RETRIEVAL_ERROR_REPLY = "文献检索服务暂时不可用，请稍后重试；本轮未生成答案。"
SERVICE_ERROR_REPLY = "问答服务调用失败，请检查运行环境或稍后重试；本轮未生成答案。"


def get_system_prompt() -> str:
    return """你是一个专业的文献知识库助手。你需要严格遵循以下规则：
1. 仅基于用户上传的论文PDF内容回答问题。提供的"参考文档"是从用户上传论文中检索到的片段。
2. 如果参考文档中没有相关信息，或者信息不足以支撑答案，请直接回复："已检索历史文档，暂未找到相关内容，无法回答"，不要编造或使用外部知识。
3. 回答中的引用统一使用参考文档标签（如 [片段1]），不要在正文重复完整长文件名；
   详细文档名与页码会由界面的“参考来源”区域展示。
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


def _run_text_hook(
    hook: CoreHook,
    value: str,
    *,
    metadata: dict[str, Any],
) -> str:
    """执行文本 hook，并阻止错误插件把核心值替换成不兼容类型。"""
    candidate = get_plugin_manager().run_hook(
        hook, value, metadata=metadata
    ).value
    return candidate if isinstance(candidate, str) and candidate.strip() else value


def _run_document_hook(
    hook: CoreHook,
    documents: list[Document],
    *,
    metadata: dict[str, Any],
) -> list[Document]:
    """执行文档 hook；第三方返回值不合法时保留上一阶段证据。"""
    candidate = get_plugin_manager().run_hook(
        hook, documents, metadata=metadata
    ).value
    if isinstance(candidate, (list, tuple)) and all(
        isinstance(item, Document) for item in candidate
    ):
        return list(candidate)
    return documents


def _retrieve_node(state: GraphState) -> GraphState:
    return {
        "documents": _retrieve_documents(state["question"], state["group_id"])
    }


def _retrieve_documents(question: str, group_id: int) -> list[Document]:
    """执行原始单查询链路，作为基础图与 multi-agent 的共同基线。"""
    retriever = get_retriever(group_id)
    candidates = _invoke_retriever(retriever, question, group_id)
    if not candidates:
        return []
    # 混合召回 -> 重排精排 -> 上下文压缩
    return _postprocess_documents(question, candidates)


def _invoke_retriever(
    retriever: Any,
    question: str,
    group_id: int,
    *,
    original_question: str | None = None,
) -> list[Document]:
    """调用检索器并开放安全 hook，multi-agent 的每个子查询复用此边界。"""
    query = _run_text_hook(
        CoreHook.BEFORE_RETRIEVE,
        question,
        metadata={
            "group_id": group_id,
            "original_question": original_question or question,
        },
    )
    documents = list(retriever.invoke(query) or []) if retriever is not None else []
    return _run_document_hook(
        CoreHook.AFTER_RETRIEVE,
        documents,
        metadata={"group_id": group_id, "query": query},
    )


def _postprocess_documents(
    question: str, candidates: list[Document]
) -> list[Document]:
    """多路召回合并后统一精排，避免每个子查询分别截断而丢失全局证据。"""
    return compress_context(question, rerank(question, candidates))


def _generate_node(state: GraphState) -> GraphState:
    documents = _prepare_documents_for_generation(
        state["question"],
        state.get("documents") or [],
        history_size=len(state.get("history", [])),
    )
    return {
        "answer": _generate_answer(
            state["question"],
            documents,
            state.get("history", []),
        ),
        # hook 可能替换证据，必须把同一份文档返回给 UI，避免答案与引用不一致。
        "documents": documents,
    }


def _prepare_documents_for_generation(
    question: str,
    documents: list[Document],
    *,
    history_size: int,
) -> list[Document]:
    return _run_document_hook(
        CoreHook.BEFORE_GENERATE,
        documents,
        metadata={"question": question, "history_size": history_size},
    )


def _generate_answer(
    question: str,
    documents: list[Document],
    history: list[BaseMessage] | None = None,
    *,
    run_after_hook: bool = True,
) -> str:
    """仅依据已选证据生成答案；无证据时固定拒答。"""
    if not documents:
        return NO_CONTEXT_REPLY

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
            "history": history or [],
            "question": question,
            "documents": format_documents(documents),
        }
    )
    answer = response.content
    if not run_after_hook:
        return answer
    return _run_text_hook(
        CoreHook.AFTER_GENERATE,
        answer,
        metadata={"question": question, "document_count": len(documents)},
    )


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


def _answer_with_agents(
    question: str,
    group_id: int,
    history: list[BaseMessage],
) -> dict[str, Any]:
    """运行受控 multi-agent；返回值额外携带可观测的降级信息。"""
    # 先取得一次缓存检索器，再把其 invoke 注入 worker，避免线程内并发改写全局缓存。
    retriever = get_retriever(group_id)
    if retriever is None:
        return {
            "answer": NO_CONTEXT_REPLY,
            "documents": [],
            "queries": [question],
            "used_multi_agent": False,
            "degraded": False,
            "retrieval_failed": False,
            "warnings": (),
        }

    config = MultiAgentConfig(
        max_subqueries=settings.multi_agent_max_subqueries,
        max_workers=settings.multi_agent_max_workers,
        retrieval_timeout_seconds=settings.multi_agent_retrieval_timeout,
        enable_evidence_review=settings.evidence_review_enabled,
        enable_answer_review=settings.answer_review_enabled,
    )
    # 角色只依赖 Runnable.invoke，因此继续复用现有供应商无关模型工厂与熔断能力。
    agent_llm = get_llm(temperature=0.0)
    orchestrator = MultiAgentOrchestrator(
        config,
        planner_llm=agent_llm,
        evidence_reviewer_llm=agent_llm,
        answer_reviewer_llm=agent_llm,
    )

    def _retrieve(query: str) -> list[Document]:
        return _invoke_retriever(
            retriever,
            query,
            group_id,
            original_question=question,
        )

    def _answer(query: str, documents: list[Document]) -> str:
        return _generate_answer(
            query, documents, history, run_after_hook=False
        )

    def _prepare(query: str, documents: list[Document]) -> list[Document]:
        return _prepare_documents_for_generation(
            query, documents, history_size=len(history)
        )

    def _baseline(query: str) -> AnswerDraft:
        documents = _retrieve_documents(query, group_id)
        documents = _prepare_documents_for_generation(
            query, documents, history_size=len(history)
        )
        return AnswerDraft(
            _generate_answer(
                query, documents, history, run_after_hook=False
            ),
            documents,
        )

    result = orchestrator.run(
        question,
        retrieve=_retrieve,
        postprocess=_postprocess_documents,
        prepare=_prepare,
        answer=_answer,
        baseline=_baseline,
    )
    answer = (
        RETRIEVAL_ERROR_REPLY
        if result.retrieval_failed and not result.documents
        else result.answer
    )
    if result.documents and not result.retrieval_failed:
        # multi-agent 的 after_generate 必须观察审校后的最终答案，而不是中间草稿。
        answer = _run_text_hook(
            CoreHook.AFTER_GENERATE,
            answer,
            metadata={
                "question": question,
                "document_count": len(result.documents),
            },
        )
    return {
        "answer": answer,
        "documents": result.documents,
        "queries": result.queries,
        "used_multi_agent": result.used_multi_agent,
        "degraded": result.degraded,
        "retrieval_failed": result.retrieval_failed,
        "warnings": result.warnings,
    }


def answer_question(
    question: str, group_id: int, memory: RelevantMemory | None = None
) -> dict[str, Any]:
    """跑一次完整问答，返回 {"answer", "documents"}；传入 memory 则记录本轮。"""
    history: list[BaseMessage] = []
    if memory is not None:
        history = memory.get_relevant_history(question)

    try:
        if settings.multi_agent_enabled:
            result = _answer_with_agents(question, group_id, history)
        else:
            graph = _get_graph()
            result = graph.invoke(
                {"question": question, "group_id": group_id, "history": history}
            )
    except Exception as exc:  # noqa: BLE001
        # 最终边界统一收敛环境/API 异常，UI 仍能展示原始类型用于排障。
        result = {
            "answer": SERVICE_ERROR_REPLY,
            "documents": [],
            "queries": [question],
            "used_multi_agent": False,
            "degraded": True,
            "retrieval_failed": False,
            "service_failed": True,
            "warnings": (f"{type(exc).__name__}: {exc}",),
        }

    answer = result.get("answer", NO_CONTEXT_REPLY)
    if memory is not None:
        memory.add_user_message(question)
        memory.add_ai_message(answer)

    return {
        "answer": answer,
        "documents": result.get("documents", []),
        "queries": result.get("queries", [question]),
        "used_multi_agent": result.get("used_multi_agent", False),
        "degraded": result.get("degraded", False),
        "retrieval_failed": result.get("retrieval_failed", False),
        "service_failed": result.get("service_failed", False),
        "warnings": result.get("warnings", ()),
    }
