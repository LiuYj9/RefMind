"""对话模型与嵌入模型的创建入口。

对话模型带一个简单的熔断降级：主模型连续失败多次后暂时切到备选模型，
冷却一段时间再放行一次探测，成功就切回主模型。
"""

from __future__ import annotations

import enum
import time
import threading
from functools import lru_cache

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..config import settings


class _State(enum.Enum):
    CLOSED = "closed"        # 正常，直接用主模型
    OPEN = "open"            # 熔断，直接走备选
    HALF_OPEN = "half_open"  # 半开，放行一次探测


class _CircuitBreaker:
    """主模型熔断器，线程安全。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = _State.CLOSED
        self._failure_count = 0
        self._opened_at = 0.0

    @property
    def failure_threshold(self) -> int:
        return settings.llm_circuit_failure_threshold

    @property
    def cooldown_seconds(self) -> int:
        return settings.llm_health_check_interval

    def _transition_to(self, target: _State) -> None:
        self.state = target
        if target == _State.CLOSED:
            self._failure_count = 0
        elif target == _State.OPEN:
            self._opened_at = time.time()

    def _increment_failure(self) -> bool:
        self._failure_count += 1
        return self._failure_count >= self.failure_threshold

    def should_allow_primary(self) -> bool:
        with self._lock:
            if self.state == _State.CLOSED:
                return True
            if self.state == _State.HALF_OPEN:
                return True
            if self.cooldown_seconds <= 0:
                self._transition_to(_State.CLOSED)
                return True
            if (time.time() - self._opened_at) > self.cooldown_seconds:
                self._transition_to(_State.HALF_OPEN)
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            if self.state == _State.HALF_OPEN:
                self._transition_to(_State.CLOSED)

    def record_failure(self) -> None:
        with self._lock:
            if self.state == _State.CLOSED:
                if self._increment_failure():
                    self._transition_to(_State.OPEN)
            elif self.state == _State.HALF_OPEN:
                self._transition_to(_State.OPEN)

    def force_reset(self) -> None:
        with self._lock:
            self._transition_to(_State.CLOSED)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self._failure_count,
                "threshold": self.failure_threshold,
                "cooldown_s": self.cooldown_seconds,
            }


_circuit_breaker = _CircuitBreaker()


@lru_cache(maxsize=1)
def get_embedding_model():
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.api_base,
        api_key=settings.dashscope_api_key,
        check_embedding_ctx_length=False,
        # DashScope 单次最多 10 条，超了会 400
        chunk_size=settings.embedding_batch_size,
    )


def _build_chat(model: str, base_url: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=settings.dashscope_api_key,
        temperature=temperature,
    )


@lru_cache(maxsize=4)
def _get_primary_model(temperature: float) -> ChatOpenAI:
    return _build_chat(settings.llm_model, settings.api_base, temperature)


@lru_cache(maxsize=4)
def _get_fallback_model(temperature: float) -> ChatOpenAI | None:
    if not settings.fallback_llm_model:
        return None
    base_url = settings.fallback_api_base or settings.api_base
    return _build_chat(settings.fallback_llm_model, base_url, temperature)


class _FallbackLLM(Runnable):
    """对话模型代理，继承 Runnable 以支持 ``prompt | llm`` 组合。

    主模型失败到阈值后熔断走备选，冷却后半开探测。
    """

    def __init__(self, temperature: float) -> None:
        self._temperature = temperature

    def invoke(self, input, config=None, **kwargs):  # noqa: A002
        last_error: Exception | None = None
        if _circuit_breaker.should_allow_primary():
            try:
                model = _get_primary_model(self._temperature)
                result = model.invoke(input, config=config, **kwargs)
                _circuit_breaker.record_success()
                return result
            except Exception as exc:
                _circuit_breaker.record_failure()
                last_error = exc
                if _circuit_breaker.state != _State.OPEN:
                    raise

        fallback = _get_fallback_model(self._temperature)
        if fallback is not None:
            return fallback.invoke(input, config=config, **kwargs)
        if last_error is not None:
            raise last_error
        raise RuntimeError("主模型熔断中且未配置备选模型（请设置 LLM_FALLBACK_MODEL）。")

    def stream(self, input, config=None, **kwargs):  # noqa: A002
        if _circuit_breaker.should_allow_primary():
            try:
                model = _get_primary_model(self._temperature)
                stream_iter = model.stream(input, config=config, **kwargs)
                first = next(stream_iter, None)
                _circuit_breaker.record_success()
                if first is not None:
                    yield first
                yield from stream_iter
                return
            except StopIteration:
                _circuit_breaker.record_success()
                return
            except Exception:
                _circuit_breaker.record_failure()
                if _circuit_breaker.state != _State.OPEN:
                    raise

        fallback = _get_fallback_model(self._temperature)
        if fallback is not None:
            yield from fallback.stream(input, config=config, **kwargs)
            return

    def __repr__(self) -> str:
        cb = _circuit_breaker.to_dict()
        return (
            f"<FallbackLLM primary={settings.llm_model} state={cb['state']} "
            f"failures={cb['failure_count']}/{cb['threshold']}"
            f" fallback={settings.fallback_llm_model or '-'}>"
        )


def get_llm(temperature: float | None = None):
    temp = settings.llm_temperature if temperature is None else temperature
    return _FallbackLLM(temp)


def get_llm_status() -> dict:
    cb = _circuit_breaker.to_dict()
    return {
        "primary_model": settings.llm_model,
        "circuit_state": cb["state"],
        "failure_count": cb["failure_count"],
        "failure_threshold": cb["threshold"],
        "cooldown_s": cb["cooldown_s"],
        "fallback_model": settings.fallback_llm_model or None,
        "active_model": (
            settings.llm_model
            if cb["state"] != "open"
            else (settings.fallback_llm_model or settings.llm_model)
        ),
    }


def reset_model_cache() -> None:
    get_embedding_model.cache_clear()
    _get_primary_model.cache_clear()
    _get_fallback_model.cache_clear()
    _circuit_breaker.force_reset()
