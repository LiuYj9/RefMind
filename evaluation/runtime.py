"""Whitelisted per-run RAG overrides; credentials, paths and persistence are excluded."""
from __future__ import annotations

OVERRIDES={
    "CHUNK_SIZE":("chunk_size",int),"CHUNK_OVERLAP":("chunk_overlap",int),
    "LAYOUT_CHUNK_MAX_CHARS":("layout_chunk_max_chars",int),
    "RECALL_TOP_K":("recall_top_k",int),"RETRIEVAL_TOP_K":("retrieval_top_k",int),
    "RERANK_ENABLED":("rerank_enabled",bool),"RERANK_TOP_N":("rerank_top_n",int),
    "CONTEXT_COMPRESSION_ENABLED":("context_compression_enabled",bool),
    "CONTEXT_BODY_PRIORITY":("context_body_priority_enabled",bool),
    "CONTEXT_MAX_CHARS":("context_max_chars",int),"CONTEXT_MAX_TOKENS":("context_max_tokens",int),
    "MODEL_CONTEXT_TOKENS":("model_context_tokens",int),"ANSWER_TOKEN_RESERVE":("answer_token_reserve",int),
    "MULTI_AGENT_ENABLED":("multi_agent_enabled",bool),"MULTI_AGENT_MAX_SUBQUERIES":("multi_agent_max_subqueries",int),
    "MULTI_AGENT_EVIDENCE_REVIEW":("evidence_review_enabled",bool),
    "MULTI_AGENT_ANSWER_REVIEW":("answer_review_enabled",bool),
    "MULTI_AGENT_CONFIRM_REFUSAL":("refusal_confirmation_enabled",bool),
}


def validate_overrides(values):
    if not isinstance(values,dict) or not set(values)<=set(OVERRIDES):
        raise ValueError("runtime overrides only accept whitelisted RAG parameters, never credentials or storage paths")
    out={}
    for key,value in values.items():
        kind=OVERRIDES[key][1]
        if kind is bool:
            raw=str(value).casefold()
            if raw not in {"true","false"}:
                raise ValueError("boolean override must be true or false")
            out[key]=raw
        else:
            if isinstance(value,bool):
                raise ValueError("numeric override cannot be boolean")
            try:
                integer=int(str(value))
            except (ValueError,TypeError):
                raise ValueError("numeric override must be a nonnegative integer") from None
            if integer<0 or integer>2_000_000:
                raise ValueError("numeric runtime override outside bounded range")
            out[key]=str(integer)
    return out


def parse_overrides(arguments):
    values={}
    for argument in arguments:
        if "=" not in argument:
            raise ValueError("--set expects KEY=VALUE")
        key,value=argument.split("=",1)
        if key in values:
            raise ValueError("duplicate runtime override")
        values[key]=value
    return validate_overrides(values)


def effective_overrides(settings,values):
    result={}
    for key,value in validate_overrides(values).items():
        attribute,kind=OVERRIDES[key]
        if not hasattr(settings,attribute):
            raise ValueError("installed settings does not support runtime override: "+key)
        actual=getattr(settings,attribute)
        expected=value=="true" if kind is bool else int(value)
        if actual!=expected:
            raise ValueError("runtime override was not consumed by installed settings: "+key)
        result[key]={"requested":value,"effective":actual,"attribute":attribute}
    return result
