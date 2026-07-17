"""供应商无关、可故障降级的多智能体 RAG 编排。

流程由四个职责不同的角色组成：

1. 规划智能体把复杂问题拆成不超过三个、可独立检索的子查询；
2. 检索智能体在有界线程池中并行召回，并按稳定分块标识合并去重；
3. 可选的证据审查智能体判断证据能否直接回答原问题，并剔除无关分块；
4. 可选的答案审校智能体检查问题对齐与证据支撑，只做必要修正。

任一增强环节失败都会保留上一阶段的有效结果。编排主流程发生未预期异常时，
则调用注入的 baseline，或在未注入 baseline 时退回“原问题单次检索 + 原答案生成”。
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import OrderedDict
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Protocol, Sequence

from langchain_core.documents import Document

from ..citations import build_citation_label


INSUFFICIENT_EVIDENCE_REPLY = "已检索历史文档，暂未找到相关内容，无法回答"
_EVIDENCE_COVERAGE = {"direct", "scoped", "insufficient"}
_EVIDENCE_ALIGNMENT_RUBRIC = """请使用以下通用判定框架，不依赖具体学科或关键词：
1. 提取问题意图：识别问题类型（定义、存在性、方法、原因/机理、数值、比较/排序、总结/列举或其他）、
   核心对象、必须得到支持的谓词/关系、强制限定条件，以及“任一/部分/全部”等覆盖要求。
2. 核心关系优先：证据必须支持问题所问的同一谓词或作用目标。只因对象、术语或领域相同，不能
   把证据支持的另一种效果、原因、指标或任务替换成用户所问关系；未经证据建立的因果链也不能补齐。
3. 范围兼容：对象的具体子类、特定工况或一种实例，可以支持非穷举的宽泛问题，但只能标记为
   scoped，答案必须保留更窄的对象、条件、指标和适用范围，不能外推到全部场景。
4. 按问题类型判断充分性：
   - 存在性/举例：一个 direct 或 scoped 实例即可支持“有/存在”，但不能由“未检索到”推出“不存在”；
   - 定义、方法、原因/机理、数值：必须明确支持所问关系及强制限定，定性证据不能回答精确数值；
   - 比较/排序：必须覆盖所比较对象、同一指标和必要条件，单方材料通常不足；
   - 总结/列举：非穷举问题可以给出有证据的部分并声明范围；含“全部、所有、最优、唯一”等要求时，
     证据必须满足相应覆盖，不能把局部结果包装成完整结论。
5. 结论分级：direct 表示直接且范围相符；scoped 表示关系成立但证据范围比问题窄；insufficient
   表示核心关系、强制条件或覆盖要求不足。存在相互冲突的直接证据时保留双方，不得静默选边。"""
_PLANNING_CACHE_MAX_SIZE = 256
_PLANNING_CACHE: OrderedDict[tuple[int, str, int], tuple[str, ...]] = OrderedDict()
_PLANNING_CACHE_LOCK = RLock()


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
    rejection_stage: str | None = None
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


_TRAILING_QUERY_PUNCTUATION_RE = re.compile(r"[\s?？!！。．.;；:：]+$")


def canonicalize_retrieval_query(value: str) -> str:
    """规范化检索问题；尾部标点差异不应产生不同的召回锚点。"""
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = " ".join(normalized.split()).strip()
    canonical = _TRAILING_QUERY_PUNCTUATION_RE.sub("", normalized).strip()
    return canonical or normalized


def _query_identity(value: str) -> str:
    return canonicalize_retrieval_query(value).casefold()


def _with_retrieval_anchor(
    queries: list[str], anchor: str, limit: int
) -> list[str]:
    """把原问题固定为第一路召回；规划查询只能扩展，不能替代原问题。"""
    anchored: list[str] = []
    seen: set[str] = set()
    for candidate in [canonicalize_retrieval_query(anchor), *queries]:
        query = " ".join(str(candidate).split()).strip()
        key = _query_identity(query)
        if not query or not key or key in seen:
            continue
        seen.add(key)
        anchored.append(query)
        if len(anchored) >= limit:
            break
    return anchored or [canonicalize_retrieval_query(anchor)]


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
        key = _query_identity(query)
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
        question = canonicalize_retrieval_query(question)
        if self._llm is None or self._config.max_subqueries == 1:
            return _PlanningResult([question])

        cache_key = (
            id(self._llm),
            _query_identity(question),
            self._config.max_subqueries,
        )
        with _PLANNING_CACHE_LOCK:
            cached = _PLANNING_CACHE.get(cache_key)
            if cached is not None:
                _PLANNING_CACHE.move_to_end(cache_key)
                return _PlanningResult(list(cached))

        prompt = f"""你是文献检索规划智能体。判断问题是否包含多个概念、比较对象或因果步骤。
简单问题保持为一个查询；复杂问题拆成互补且可独立检索的子查询。
最多返回 {self._config.max_subqueries} 个，不要回答问题，不要引入题外概念。
只输出 JSON：{{"subqueries": ["查询1", "查询2"]}}
原问题（JSON 字符串）：{json.dumps(question, ensure_ascii=False)}"""
        try:
            payload = _parse_json_payload(_invoke_text(self._llm, prompt))
            queries = _normalise_queries(
                question, payload, self._config.max_subqueries
            )
            with _PLANNING_CACHE_LOCK:
                _PLANNING_CACHE[cache_key] = tuple(queries)
                _PLANNING_CACHE.move_to_end(cache_key)
                while len(_PLANNING_CACHE) > _PLANNING_CACHE_MAX_SIZE:
                    _PLANNING_CACHE.popitem(last=False)
            return _PlanningResult(queries)
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
        content = document.page_content[: config.max_document_chars]
        blocks.append(f"[{index}] {build_citation_label(document, index)}\n{content}")
    return "\n\n".join(blocks)


class EvidenceReviewAgent:
    """判断候选证据能否直接回答问题，不生成最终答案。"""

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
        prompt = f"""你是证据可回答性审查智能体，只判断候选片段能否支撑对原问题的回答。
{_EVIDENCE_ALIGNMENT_RUBRIC}

answerable 仅在 coverage 为 direct 或 scoped 且至少有一个有效证据编号时为 true。
keep 只保留直接证据、理解核心概念所必需的证据，以及完成比较不可缺少的证据。
只输出 JSON，编号从 1 开始：
{{"question_type": "方法", "requested_relation": "简述所问关系", "coverage": "scoped",
  "answerable": true, "keep": [1, 3], "scope_limit": "必须保留的范围", "reason": "简短理由"}}
或 {{"question_type": "数值", "requested_relation": "简述所问关系", "coverage": "insufficient",
  "answerable": false, "keep": [], "scope_limit": "", "reason": "证据不足之处"}}
问题：{json.dumps(question, ensure_ascii=False)}

候选证据：
{_format_review_documents(documents, self._config)}"""

        def _evaluate(prompt_text: str) -> list[Document]:
            payload = _parse_json_payload(_invoke_text(self._llm, prompt_text))
            answerable = (
                payload.get("answerable") if isinstance(payload, dict) else None
            )
            if not isinstance(answerable, bool):
                raise ValueError("证据审查响应缺少布尔值 answerable")
            coverage = payload.get("coverage")
            if coverage not in _EVIDENCE_COVERAGE:
                raise ValueError("证据审查响应 coverage 无效")
            if not answerable:
                if coverage != "insufficient":
                    raise ValueError("不可回答时 coverage 必须为 insufficient")
                return []
            if coverage == "insufficient":
                raise ValueError("可回答时 coverage 不能为 insufficient")
            keep = payload.get("keep") if isinstance(payload, dict) else None
            if not isinstance(keep, list):
                raise ValueError("证据审查响应缺少 keep 数组")
            indexes = {
                value - 1
                for value in keep
                if isinstance(value, int)
                and not isinstance(value, bool)
                and 1 <= value <= len(reviewable)
            }
            if not indexes:
                raise ValueError("证据审查判定可回答，但 keep 中没有有效编号")
            return [
                doc for index, doc in enumerate(reviewable) if index in indexes
            ]

        try:
            selected = _evaluate(prompt)
            if selected:
                return selected, None
            # 单次 LLM 否定可能有采样抖动；只有两次独立判断都不足才真正拒答。
            confirmed = _evaluate(
                prompt
                + "\n\n请独立复核上一判断。不要因倾向拒答而提高门槛，仍严格按上述框架输出 JSON。"
            )
            if confirmed:
                return confirmed, "证据审查首次拒答，二次复核后确认可回答"
            return [], None
        except Exception as exc:  # noqa: BLE001 - 审查失败不能丢失已有证据
            return documents, f"证据审查失败，已保留原证据：{exc}"


class AnswerReviewAgent:
    """依据已选证据检查问题对齐与草稿支撑，失败时原样返回草稿。"""

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

        prompt = f"""你是答案审校智能体。仅依据给定证据检查草稿是否回答了原问题，
是否有无依据断言、遗漏强制限定、范围外推或错误引用。
{_EVIDENCE_ALIGNMENT_RUBRIC}

如果证据不足以直接回答原问题，answerable 必须为 false，answer 必须是固定拒答文本：
“{INSUFFICIENT_EVIDENCE_REPLY}”。如果可以回答，则做最小必要修正；不要补充外部知识。
coverage 为 scoped 时，审校后的答案必须显式保留证据支持的较窄范围；不得把局部证据写成
普遍、完整、最优或唯一结论。
必须保留或原样使用证据中的完整引用标签，不得改写为 [片段N] 或 [N]。
只输出 JSON：
{{"coverage": "direct", "answerable": true, "answer": "审校后的完整答案", "reason": "简短理由"}}
或 {{"coverage": "insufficient", "answerable": false,
  "answer": "{INSUFFICIENT_EVIDENCE_REPLY}", "reason": "证据不足"}}
问题：{json.dumps(question, ensure_ascii=False)}
原答案：{json.dumps(draft, ensure_ascii=False)}

证据：
{_format_review_documents(documents, self._config)}"""

        def _evaluate(prompt_text: str) -> str | None:
            payload = _parse_json_payload(_invoke_text(self._llm, prompt_text))
            answerable = (
                payload.get("answerable") if isinstance(payload, dict) else None
            )
            if not isinstance(answerable, bool):
                raise ValueError("答案审校响应缺少布尔值 answerable")
            coverage = payload.get("coverage")
            if coverage not in _EVIDENCE_COVERAGE:
                raise ValueError("答案审校响应 coverage 无效")
            if not answerable:
                if coverage != "insufficient":
                    raise ValueError("不可回答时 coverage 必须为 insufficient")
                return None
            if coverage == "insufficient":
                raise ValueError("可回答时 coverage 不能为 insufficient")
            answer = payload.get("answer") if isinstance(payload, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("答案审校响应缺少非空 answer")
            return answer.strip()

        try:
            answer = _evaluate(prompt)
            if answer is not None:
                return answer, None
            confirmed = _evaluate(
                prompt
                + "\n\n请独立复核上一拒答判断。不要因草稿措辞可修正就否定证据，仍严格输出 JSON。"
            )
            if confirmed is not None:
                return confirmed, "答案审校首次拒答，二次复核后确认可回答"
            return INSUFFICIENT_EVIDENCE_REPLY, None
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
        review_question: str | None = None,
        retrieval_anchor: str | None = None,
        retrieval_expansions: Sequence[str] = (),
    ) -> AgentRunResult:
        """运行增强问答；``baseline`` 用于兜底复用项目原始问答路径。

        ``postprocess`` 注入已有的“重排 + 压缩”流水线。``prepare`` 无论前者
        是否降级都会在生成前执行，适合放置必须与最终引用保持一致的证据 hook。
        ``question`` 可包含用于召回的记忆增强信息；``review_question`` 始终保留
        用户原问题，避免研究背景把证据审查带偏。``retrieval_anchor`` 会被规范化
        并固定为第一路查询，规划器只能追加检索角度而不能替代原问题。记忆等
        ``retrieval_expansions`` 仅作为附加召回角度，不参与规划或全局重排问题。
        """
        warnings: list[str] = []
        try:
            grounding_question = review_question or question
            planning = self._planner.plan(question)
            fallback_query = canonicalize_retrieval_query(
                retrieval_anchor or grounding_question
            )
            if retrieval_anchor is not None:
                planning.queries = _with_retrieval_anchor(
                    [*planning.queries, *retrieval_expansions],
                    fallback_query,
                    self.config.max_subqueries,
                )
            if planning.warning:
                warnings.append(planning.warning)

            retrieval = self._retriever.retrieve(
                fallback_query, planning.queries, retrieve
            )
            warnings.extend(retrieval.warnings)

            documents = retrieval.documents
            postprocess_warning: str | None = None
            if postprocess is not None and documents:
                try:
                    documents = list(postprocess(fallback_query, documents) or [])
                except Exception as exc:  # noqa: BLE001 - 保留已召回证据
                    postprocess_warning = f"重排/压缩失败，已保留原始召回：{exc}"
                    warnings.append(postprocess_warning)

            prepare_warning: str | None = None
            if prepare is not None and documents:
                try:
                    documents = list(prepare(grounding_question, documents) or [])
                except Exception as exc:  # noqa: BLE001
                    prepare_warning = f"生成前证据准备失败，已保留原证据：{exc}"
                    warnings.append(prepare_warning)

            evidence_input_count = len(documents)
            documents, evidence_warning = self._evidence_reviewer.review(
                grounding_question, documents
            )
            evidence_rejected = evidence_input_count > 0 and not documents
            if evidence_warning:
                warnings.append(evidence_warning)

            draft = answer(grounding_question, documents)
            final_answer, answer_warning = self._answer_reviewer.review(
                grounding_question, documents, draft
            )
            if answer_warning:
                warnings.append(answer_warning)
            answer_rejected = (
                bool(documents)
                and final_answer == INSUFFICIENT_EVIDENCE_REPLY
            )
            if final_answer == INSUFFICIENT_EVIDENCE_REPLY:
                # 拒答不应继续携带相邻主题文献，否则界面会追加误导性参考来源。
                documents = []

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
                rejection_stage=(
                    "evidence_review"
                    if evidence_rejected
                    else "answer_review"
                    if answer_rejected
                    else None
                ),
                warnings=tuple(warnings),
            )
        except Exception as exc:  # noqa: BLE001 - 主流程最终安全网
            warnings.append(f"多智能体流程异常，已退回基线问答：{exc}")
            return self._run_baseline(
                canonicalize_retrieval_query(
                    retrieval_anchor or review_question or question
                ),
                retrieve,
                answer,
                baseline,
                warnings,
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
