"""生成侧指标（LLM-as-judge）：Faithfulness、Answer Relevance、
Context Precision、Context Recall。裁判用项目自带对话模型，仅返回 JSON。
裁判不可用时对应指标返回 None，汇总时跳过。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


def _get_judge(temperature: float = 0.0):
    """获取裁判模型（复用项目的对话模型工厂）。"""
    from refmind.llm import get_llm

    return get_llm(temperature=temperature)


def _extract_json(text: str) -> Optional[dict]:
    """从模型输出中稳健地提取第一段 JSON 对象。"""
    if not text:
        return None
    # 去掉可能的 ```json``` 代码块包裹
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _judge_json(prompt: str, temperature: float = 0.0) -> Optional[dict]:
    """调用裁判模型并解析 JSON；失败返回 None。"""
    try:
        judge = _get_judge(temperature)
        resp = judge.invoke(prompt)
        return _extract_json(resp.content)
    except Exception as exc:  # noqa: BLE001
        print(f"[generation] 裁判调用失败：{exc}")
        return None


def _clamp01(value: Any) -> Optional[float]:
    """把裁判返回的数值裁剪到 [0, 1]。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, v))


# 各指标提示词
_FAITHFULNESS_PROMPT = """你是严格的 RAG 评测裁判。请判断"答案"中的每条事实性陈述是否能被"上下文"支撑。

评测步骤：
1. 将答案拆分为若干条独立的事实性陈述（忽略礼貌用语、无信息的套话）。
2. 对每条陈述判断：上下文是否明确支持（supported）。
3. faithfulness = 被支持的陈述数 / 陈述总数。若无任何事实性陈述，faithfulness 记为 1.0。

只返回如下 JSON（不要输出任何解释）：
{{"total_claims": <整数>, "supported_claims": <整数>, "faithfulness": <0到1小数>}}

问题：
{question}

上下文：
{context}

答案：
{answer}"""

_ANSWER_RELEVANCE_PROMPT = """你是严格的 RAG 评测裁判。请评估"答案"在多大程度上直接、切题地回应了"问题"。

评分标准（0~1）：
- 1.0：完全切题、直接回答了问题，无冗余或跑题。
- 0.5：部分切题，或答案含较多与问题无关的信息。
- 0.0：答非所问，或仅回复"无法回答/未找到"之类且问题本应可答。

只返回如下 JSON（不要输出任何解释）：
{{"answer_relevance": <0到1小数>}}

问题：
{question}

答案：
{answer}"""

_CONTEXT_PRECISION_PROMPT = """你是严格的 RAG 评测裁判。下面给出针对某问题检索到的若干"上下文片段"。
请逐个判断：该片段对"回答该问题"是否有用（useful=1）或无用/无关（useful=0）。

context_precision = 有用片段数 / 片段总数。

只返回如下 JSON（不要输出任何解释）：
{{"usefulness": [<0或1>, ...], "context_precision": <0到1小数>}}

问题：
{question}

标准答案（供判断相关性参考）：
{ground_truth}

上下文片段（按检索顺序）：
{numbered_context}"""

_CONTEXT_RECALL_PROMPT = """你是严格的 RAG 评测裁判。请判断"标准答案"中的每条信息点是否能在"上下文"中找到依据。

评测步骤：
1. 将标准答案拆分为若干条独立信息点。
2. 对每条信息点判断：上下文是否能支撑（attributed）。
3. context_recall = 可被支撑的信息点数 / 信息点总数。

只返回如下 JSON（不要输出任何解释）：
{{"total_points": <整数>, "attributed_points": <整数>, "context_recall": <0到1小数>}}

上下文：
{context}

标准答案：
{ground_truth}"""


@dataclass
class GenerationScores:
    """单个问题的生成侧指标（None 表示未评测/裁判不可用）。"""

    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        def r(x: Optional[float]) -> Optional[float]:
            return round(x, 4) if x is not None else None

        return {
            "faithfulness": r(self.faithfulness),
            "answer_relevance": r(self.answer_relevance),
            "context_precision": r(self.context_precision),
            "context_recall": r(self.context_recall),
        }


def evaluate_generation(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    temperature: float = 0.0,
) -> GenerationScores:
    """用 LLM 裁判计算四项生成侧指标。"""
    scores = GenerationScores()
    context_join = "\n\n".join(contexts) if contexts else "（无检索上下文）"
    numbered = (
        "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
        if contexts
        else "（无检索上下文）"
    )

    # Faithfulness
    data = _judge_json(
        _FAITHFULNESS_PROMPT.format(
            question=question, context=context_join, answer=answer
        ),
        temperature,
    )
    if data is not None:
        scores.faithfulness = _clamp01(data.get("faithfulness"))

    # Answer Relevance
    data = _judge_json(
        _ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer),
        temperature,
    )
    if data is not None:
        scores.answer_relevance = _clamp01(data.get("answer_relevance"))

    # Context Precision
    data = _judge_json(
        _CONTEXT_PRECISION_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            numbered_context=numbered,
        ),
        temperature,
    )
    if data is not None:
        scores.context_precision = _clamp01(data.get("context_precision"))

    # Context Recall
    data = _judge_json(
        _CONTEXT_RECALL_PROMPT.format(
            context=context_join, ground_truth=ground_truth
        ),
        temperature,
    )
    if data is not None:
        scores.context_recall = _clamp01(data.get("context_recall"))

    return scores
