"""Stable research request/result contracts shared by HTTP and evaluation."""
from __future__ import annotations
from typing import Annotated, Any, Literal, TypedDict
from pydantic import BaseModel, Field, model_validator


class AbstractChoice(BaseModel):
    search_id: str
    paper_id: str


class ResearchScope(BaseModel):
    mode: Literal['library', 'external', 'mixed'] = 'library'
    group_id: Annotated[int, Field(strict=True, gt=0)] | None = None
    selected_doc_ids: list[Annotated[int, Field(strict=True, gt=0)]] | None = Field(default=None, max_length=500)
    asset_ids: list[str] = Field(default_factory=list, max_length=20)
    abstracts: list[AbstractChoice] = Field(default_factory=list, max_length=20)

    @model_validator(mode='after')
    def explicit_source_selection(self):
        if self.mode == 'library' and (self.asset_ids or self.abstracts):
            raise ValueError('请先主动选择外部或混合研读模式')
        if self.mode in {'external', 'mixed'} and not (self.asset_ids or self.abstracts):
            raise ValueError('请先选择外部摘要或全文')
        return self


class EvaluationTrace(TypedDict, total=False):
    pipeline: str
    original_question: str
    queries: list[dict[str, Any]]
    scope: dict[str, Any]
    retrieval_channels: dict[str, Any]
    retrieval_counts: dict[str, int]
    stages: dict[str, Any]
    timings: dict[str, float]
    packing: dict[str, Any]
    answer_review: dict[str, Any]
    pre_generation: dict[str, Any]
    versions: dict[str, Any]
    costs: dict[str, Any]


class AnswerResult(TypedDict, total=False):
    status: Literal['answered', 'scoped', 'insufficient', 'clarification', 'system_error']
    answer: str
    documents: list[Any]
    warnings: tuple[str, ...] | list[str]
    task_description: dict[str, Any]
    evidence_review: dict[str, Any]
    trace: EvaluationTrace
    research_scope: dict[str, Any]
    service_failed: bool
