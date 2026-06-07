"""模型工厂。

通过阿里云 DashScope 的 OpenAI 兼容接口创建对话模型与嵌入模型。
模型客户端惰性创建并缓存，因此导入本模块不会强制要求配置有效的 API 密钥。
"""

from __future__ import annotations

from functools import lru_cache

from ..config import settings


@lru_cache(maxsize=1)
def get_embedding_model():
    """返回缓存的嵌入模型客户端（OpenAI 兼容）。"""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.api_base,
        api_key=settings.dashscope_api_key,
        # 通义嵌入模型无需 OpenAI 的 token 长度校验
        check_embedding_ctx_length=False,
        # DashScope 兼容接口单次请求最多 10 条，超出会报 400，需限制批大小
        chunk_size=settings.embedding_batch_size,
    )


@lru_cache(maxsize=4)
def get_llm(temperature: float | None = None):
    """返回缓存的对话模型客户端（OpenAI 兼容）。

    ``temperature`` 为 None 时使用配置中的默认温度。
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.api_base,
        api_key=settings.dashscope_api_key,
        temperature=settings.llm_temperature if temperature is None else temperature,
    )


def reset_model_cache() -> None:
    """清空对话 / 嵌入模型客户端缓存（配置变更后调用）。"""
    get_embedding_model.cache_clear()
    get_llm.cache_clear()
