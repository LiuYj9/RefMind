"""供应商无关、可故障降级的多智能体 RAG 编排。

流程由四个职责不同的角色组成：

1. 规划智能体把复杂问题拆成不超过三个、可独立检索的子查询；
2. 检索智能体在有界线程池中并行召回，并按稳定分块标识合并去重；
3. 可选的证据审查智能体剔除与问题无关的分块；
4. 可选的答案审校智能体只修正证据无法支撑的陈述。

任一增强环节失败都会保留上一阶段的有效结果。编排主流程发生未预期异常时，
则调用注入的 baseline，或在未注入 baseline 时退回“原问题单次检索 + 原答案生成”。
"""

from __future__ import annotations

import json
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from langchain_core.documents import Document


class Invokable(Protocol):
    """规划/审查模型所需的最小接口，兼容 LangChain Runnable。"""

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any: ...


RetrieveCallable = Callable[[str], Sequence[Document]]
AnswerCallable = Callable[[str, list[Document]], str]
PostprocessCallable = Callable[[str, list[Document]], Sequence[Document]]


@dataclass(frozen=True, slots=True)
class MultiAgentConfig:
    """构造期注入的编排配置，避免读取或修改全局 settings。"""

    max_subqueries: int = 3
    max_workers: int = 3
    retrieval_timeout_seconds: float = 30.0
    enable_evidence_review: bool = False
    enable_answer_review: bool = False
    max_review_documents: int = 12
    max_document_chars: int = 2_000

    def __post_init__(self) -> None:
        # 查询数硬限制为 3，避免规划失控造成延迟和费用线性放大。
        if not 1 <= self.max_subqueries <= 3:
            raise ValueError("max_subqueries 必须在 1 到 3 之间")
        # 同时设置绝对上限，防止外部配置错误创建过多线程。
        if not 1 <= self.max_workers <= 8:
            raise ValueError("max_workers 必须在 1 到 8 之间")
        if not 0 < self.retrieval_timeout_seconds <= 300:
            raise ValueError("retrieval_timeout_seconds 必须在 0 到 300 秒之间")
        if self.max_review_documents < 1:
            raise ValueError("max_review_documents 必须大于 0")
        if self.max_document_chars < 100:
            raise ValueError("max_document_chars 不能小于 100")


@dataclass(slots=True)
class AnswerDraft:
    """baseline 返回的原始答案及其证据。"""

    answer: str
    documents: list[Document] = field(default_factory=list)


@dataclass(slots=True)
class AgentRunResult:
    """编排结果；前两个字段可直接适配现有 ``answer_question``。"""

    answer: str
    documents: list[Document]
    queries: list[str]
    used_multi_agent: bool
    degraded: bool = False
    retrieval_failed: bool = False
    warnings: tuple[str, ...] = ()

    def to_answer_dict(self) -> dict[str, Any]:
        """保留现有问答接口的 ``answer`` / ``documents`` 返回形状。"""
        return {"answer": self.answer, "documents": self.documents}


@dataclass(slots=True)
class _PlanningResult:
    queries: list[str]
    degraded: bool = False
    warning: str | None = None


@dataclass(slots=True)
class _RetrievalResult:
    documents: list[Document]
    degraded: bool = False
    failed: bool = False
    warnings: list[str] = field(default_factory=list)


def _response_text(response: Any) -> str:
    """统一提取字符串和 LangChain 消息对象的文本内容。"""
    if isinstance(response, str):
        return response.strip()
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if content is None:
        raise TypeError("模型响应缺少字符串 content")
    return str(content).strip()


def _invoke_text(llm: Invokable, prompt: str) -> str:
    return _response_text(llm.invoke(prompt))


def _parse_json_payload(text: str) -> Any:
    """从纯 JSON 或 Markdown 围栏中解析首个 JSON 对象。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        last_fence = cleaned.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            cleaned = cleaned[first_newline + 1 : last_fence].strip()

    positions = [pos for pos in (cleaned.find("{"), cleaned.find("[")) if pos >= 0]
    if not positions:
        raise ValueError("响应中未找到 JSON")
    payload, _ = json.JSONDecoder().raw_decode(cleaned[min(positions) :])
    return payload


def _normalise_queries(question: str, raw_queries: Any, limit: int) -> list[str]:
    if isinstance(raw_queries, dict):
        raw_queries = raw_queries.get("subqueries") or raw_queries.get("queries")
    if not isinstance(raw_queries, list):
        raise ValueError("规划响应必须包含 subqueries 数组")

    queries: list[str] = []
    seen: set[str] = set()
    for value in raw_queries:
        if not isinstance(value, str):
            continue
        query = " ".join(value.split()).strip()
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            queries.append(query)
        if len(queries) >= limit:
            break
    return queries or [question]


class PlanningAgent:
    """判断检索角度并生成最多三个互补子查询。"""

    def __init__(self, llm: Invokable | None, config: MultiAgentConfig) -> None:
        self._llm = llm
        self._config = config

    def plan(self, question: str) -> _PlanningResult:
        if self._llm is None or self._config.max_subqueries == 1:
            return _PlanningResult([question])

        prompt = f"""你是文献检索规划智能体。判断问题是否包含多个概念、比较对象或因果步骤。
简单问题保持为一个查询；复杂问题拆成互补且可独立检索的子查询。
最多返回 {self._config.max_subqueries} 个，不要回答问题，不要引入题外概念。
只输出 JSON：{{"subqueries": ["查询1", "查询2"]}}
原问题（JSON 字符串）：{json.dumps(question, ensure_ascii=False)}"""
        try:
            payload = _parse_json_payload(_invoke_text(self._llm, prompt))
            return _PlanningResult(
                _normalise_queries(question, payload, self._config.max_subqueries)
            )
        except Exception as exc:  # noqa: BLE001 - 规划失败必须安全降级
            return _PlanningResult(
                [question],
                degraded=True,
                warning=f"规划智能体不可用，已退回原问题：{exc}",
            )


def _document_key(document: Document) -> tuple[Any, ...]:
    """优先使用稳定 chunk_id；旧数据缺字段时逐级回退。"""
    meta = document.metadata or {}
    chunk_id = meta.get("chunk_id")
    if chunk_id not in (None, ""):
        return ("chunk_id", str(chunk_id))

    doc_id = meta.get("doc_id")
    chunk_index = meta.get("chunk_index")
    if doc_id not in (None, "", -1) and chunk_index not in (None, ""):
        return (
            "position",
            str(doc_id),
            str(meta.get("version", "")),
            str(chunk_index),
        )

    # 最后回退到来源、页码和规范化文本，兼容测试数据与早期入库数据。
    content = " ".join(document.page_content.split())
    return (
        "content",
        str(meta.get("source") or meta.get("filename") or ""),
        str(meta.get("page", "")),
        content,
    )


def _deduplicate_documents(groups: Sequence[Sequence[Document]]) -> list[Document]:
    documents: list[Document] = []
    seen: set[tuple[Any, ...]] = set()
    for group in groups:
        for document in group:
            key = _document_key(document)
            if key in seen:
                continue
            seen.add(key)
            documents.append(document)
    return documents


class ParallelRetrievalAgent:
    """并发执行子查询，并以确定性顺序合并证据。"""

    def __init__(self, config: MultiAgentConfig) -> None:
        self._config = config

    def retrieve(
        self,
        question: str,
        queries: list[str],
        retrieve_query: RetrieveCallable,
    ) -> _RetrievalResult:
        if len(queries) == 1:
            query = queries[0]
            try:
                documents = list(retrieve_query(query) or [])
            except Exception as exc:  # noqa: BLE001
                if query.casefold() == question.casefold():
                    return _RetrievalResult(
                        documents=[],
                        degraded=True,
                        failed=True,
                        warnings=[f"单查询检索失败：{exc}"],
                    )
                documents = []

            if documents or query.casefold() == question.casefold():
                return _RetrievalResult(documents)

            # 规划器可能只返回一个改写查询；改写无结果时仍要尝试用户原问题。
            try:
                baseline_docs = list(retrieve_query(question) or [])
                return _RetrievalResult(
                    documents=baseline_docs,
                    degraded=True,
                    warnings=["改写查询未召回证据，已退回原问题单查询"],
                )
            except Exception as exc:  # noqa: BLE001
                return _RetrievalResult(
                    documents=[],
                    degraded=True,
                    failed=True,
                    warnings=[f"原问题单查询失败：{exc}"],
                )

        worker_count = min(self._config.max_workers, len(queries))
        results: list[list[Document]] = [[] for _ in queries]
        warnings: list[str] = []

        # 线程池只在本次请求内存活；模块不保存 executor、future 或检索缓存。
        # 每个 worker 仅把本地结果写回主线程，避免并发修改共享列表/缓存。
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="refmind-retrieval",
        )
        timed_out = False
        try:
            futures = {
                executor.submit(retrieve_query, query): (index, query)
                for index, query in enumerate(queries)
            }
            for future in as_completed(
                futures, timeout=self._config.retrieval_timeout_seconds
            ):
                index, query = futures[future]
                try:
                    results[index] = list(future.result() or [])
                except Exception as exc:  # noqa: BLE001 - 单路失败不影响其他证据
                    warnings.append(f"子查询检索失败（{query}）：{exc}")
        except FuturesTimeoutError:
            timed_out = True
            warnings.append(
                f"并行检索超过 {self._config.retrieval_timeout_seconds:g} 秒，"
                "已停止等待未完成子查询"
            )
            for future in futures:
                future.cancel()
        finally:
            # Python 无法强杀已运行线程；不等待可让请求先降级返回，底层 HTTP 仍应设超时。
            executor.shutdown(wait=False, cancel_futures=True)

        merged = _deduplicate_documents(results)
        if merged:
            return _RetrievalResult(
                documents=merged,
                degraded=bool(warnings) or timed_out,
                warnings=warnings,
            )

        if timed_out:
            return _RetrievalResult(
                documents=[],
                degraded=True,
                failed=True,
                warnings=warnings,
            )

        # 所有增强检索均失败或为空时，最后用原问题走一次单查询路径。
        try:
            baseline_docs = list(retrieve_query(question) or [])
            warnings.append("子查询未召回证据，已退回原问题单查询")
            return _RetrievalResult(
                documents=baseline_docs,
                degraded=True,
                warnings=warnings,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"原问题单查询也失败：{exc}")
            return _RetrievalResult(
                documents=[],
                degraded=True,
                failed=True,
                warnings=warnings,
            )


def _format_review_documents(
    documents: list[Document], config: MultiAgentConfig
) -> str:
    blocks: list[str] = []
    for index, document in enumerate(
        documents[: config.max_review_documents], start=1
    ):
        metadata = document.metadata or {}
        source = metadata.get("filename") or metadata.get("source") or "未知来源"
        page = metadata.get("page", "?")
        content = document.page_content[: config.max_document_chars]
        blocks.append(f"[{index}] 来源={source} 页码={page}\n{content}")
    return "\n\n".join(blocks)


class EvidenceReviewAgent:
    """只负责判断证据相关性，不生成最终答案。"""

    def __init__(self, llm: Invokable | None, config: MultiAgentConfig) -> None:
        self._llm = llm
        self._config = config

    def review(
        self, question: str, documents: list[Document]
    ) -> tuple[list[Document], str | None]:
        if not self._config.enable_evidence_review or not documents:
            return documents, None
        if self._llm is None:
            return documents, "未注入证据审查模型，已保留全部证据"

        reviewable = documents[: self._config.max_review_documents]
        prompt = f"""你是证据审查智能体，只判断片段能否帮助回答问题。
保留直接相关、定义关键概念或提供必要比较依据的片段；排除仅关键词重合的片段。
只输出 JSON，编号从 1 开始：{{"keep": [1, 3]}}
问题：{json.dumps(question, ensure_ascii=False)}

候选证据：
{_format_review_documents(documents, self._config)}"""
        try:
            payload = _parse_json_payload(_invoke_text(self._llm, prompt))
            keep = payload.get("keep") if isinstance(payload, dict) else None
            if not isinstance(keep, list):
                raise ValueError("证据审查响应缺少 keep 数组")
            indexes = {
                value - 1
                for value in keep
                if isinstance(value, int) and 1 <= value <= len(reviewable)
            }
            return [doc for index, doc in enumerate(reviewable) if index in indexes], None
        except Exception as exc:  # noqa: BLE001 - 审查失败不能丢失已有证据
            return documents, f"证据审查失败，已保留原证据：{exc}"


class AnswerReviewAgent:
    """依据已选证据审校草稿，失败时原样返回草稿。"""

    def __init__(self, llm: Invokable | None, config: MultiAgentConfig) -> None:
        self._llm = llm
        self._config = config

    def review(
        self,
        question: str,
        documents: list[Document],
        draft: str,
    ) -> tuple[str, str | None]:
        if not self._config.enable_answer_review or not documents:
            return draft, None
        if self._llm is None:
            return draft, "未注入答案审校模型，已保留原答案"

        prompt = f"""你是答案审校智能体。仅依据给定证据检查草稿是否有无依据断言、
遗漏关键限定或错误引用。若有问题请做最小必要修正；不要补充外部知识。
只输出 JSON：{{"answer": "审校后的完整答案"}}
问题：{json.dumps(question, ensure_ascii=False)}
原答案：{json.dumps(draft, ensure_ascii=False)}

证据：
{_format_review_documents(documents, self._config)}"""
        try:
            payload = _parse_json_payload(_invoke_text(self._llm, prompt))
            answer = payload.get("answer") if isinstance(payload, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("答案审校响应缺少非空 answer")
            return answer.strip(), None
        except Exception as exc:  # noqa: BLE001 - 审校是可选增强，必须保留草稿
            return draft, f"答案审校失败，已保留原答案：{exc}"


class MultiAgentOrchestrator:
    """组合各专家角色，并把所有增强失败收敛为可观察的降级结果。"""

    def __init__(
        self,
        config: MultiAgentConfig | None = None,
        *,
        planner_llm: Invokable | None = None,
        evidence_reviewer_llm: Invokable | None = None,
        answer_reviewer_llm: Invokable | None = None,
    ) -> None:
        self.config = config or MultiAgentConfig()
        self._planner = PlanningAgent(planner_llm, self.config)
        self._retriever = ParallelRetrievalAgent(self.config)
        self._evidence_reviewer = EvidenceReviewAgent(
            evidence_reviewer_llm, self.config
        )
        self._answer_reviewer = AnswerReviewAgent(
            answer_reviewer_llm, self.config
        )

    def run(
        self,
        question: str,
        *,
        retrieve: RetrieveCallable,
        answer: AnswerCallable,
        postprocess: PostprocessCallable | None = None,
        prepare: PostprocessCallable | None = None,
        baseline: Callable[[str], AnswerDraft] | None = None,
    ) -> AgentRunResult:
        """运行增强问答；``baseline`` 用于兜底复用项目原始问答路径。

        ``postprocess`` 注入已有的“重排 + 压缩”流水线。``prepare`` 无论前者
        是否降级都会在生成前执行，适合放置必须与最终引用保持一致的证据 hook。
        """
        warnings: list[str] = []
        try:
            planning = self._planner.plan(question)
            if planning.warning:
                warnings.append(planning.warning)

            retrieval = self._retriever.retrieve(
                question, planning.queries, retrieve
            )
            warnings.extend(retrieval.warnings)

            documents = retrieval.documents
            postprocess_warning: str | None = None
            if postprocess is not None and documents:
                try:
                    documents = list(postprocess(question, documents) or [])
                except Exception as exc:  # noqa: BLE001 - 保留已召回证据
                    postprocess_warning = f"重排/压缩失败，已保留原始召回：{exc}"
                    warnings.append(postprocess_warning)

            prepare_warning: str | None = None
            if prepare is not None and documents:
                try:
                    documents = list(prepare(question, documents) or [])
                except Exception as exc:  # noqa: BLE001
                    prepare_warning = f"生成前证据准备失败，已保留原证据：{exc}"
                    warnings.append(prepare_warning)

            documents, evidence_warning = self._evidence_reviewer.review(
                question, documents
            )
            if evidence_warning:
                warnings.append(evidence_warning)

            draft = answer(question, documents)
            final_answer, answer_warning = self._answer_reviewer.review(
                question, documents, draft
            )
            if answer_warning:
                warnings.append(answer_warning)

            return AgentRunResult(
                answer=final_answer,
                documents=documents,
                queries=planning.queries,
                used_multi_agent=len(planning.queries) > 1,
                degraded=planning.degraded
                or retrieval.degraded
                or bool(postprocess_warning)
                or bool(prepare_warning)
                or bool(evidence_warning)
                or bool(answer_warning),
                retrieval_failed=retrieval.failed,
                warnings=tuple(warnings),
            )
        except Exception as exc:  # noqa: BLE001 - 主流程最终安全网
            warnings.append(f"多智能体流程异常，已退回基线问答：{exc}")
            return self._run_baseline(
                question, retrieve, answer, baseline, warnings
            )

    @staticmethod
    def _run_baseline(
        question: str,
        retrieve: RetrieveCallable,
        answer: AnswerCallable,
        baseline: Callable[[str], AnswerDraft] | None,
        warnings: list[str],
    ) -> AgentRunResult:
        if baseline is not None:
            draft = baseline(question)
        else:
            # 未提供原流程回调时，仍保证退回单查询而不是再次执行规划。
            documents = list(retrieve(question) or [])
            draft = AnswerDraft(answer(question, documents), documents)
        return AgentRunResult(
            answer=draft.answer,
            documents=list(draft.documents),
            queries=[question],
            used_multi_agent=False,
            degraded=True,
            warnings=tuple(warnings),
        )
