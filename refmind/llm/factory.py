"""模型工厂。

通过阿里云 DashScope 的 OpenAI 兼容接口创建对话模型与嵌入模型。
模型客户端惰性创建并缓存，因此导入本模块不会强制要求配置有效的 API 密钥。

支持熔断降级：主模型连续失败 N 次后熔断（Open），经过冷却期进入半开（Half-Open）
状态放行一路探测；探测成功则关闭熔断器恢复主模型。
"""

from __future__ import annotations

import enum
import time
import threading
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..config import settings


# ---------------------------------------------------------------------------
# 熔断器三态模型
# ---------------------------------------------------------------------------
class _State(enum.Enum):
    CLOSED = "closed"        # 正常，直接调用主模型
    OPEN = "open"            # 熔断，跳过主模型直接走备选
    HALF_OPEN = "half_open"  # 半开，尝试放行一路请求探测主模型


class _CircuitBreaker:
    """LLM 主模型熔断器（线程安全，模块级单例）。

    三态转换规则::

        CLOSED ──连续失败≥阈值──▶ OPEN ──冷却超时──▶ HALF_OPEN
          ▲                                              │
          └──────── 成功 ──────────◀─────────────────────┘
                                  (失败则重新 OPEN)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = _State.CLOSED
        self._failure_count = 0
        self._opened_at = 0.0  # OPEN 状态的开始时间

    @property
    def failure_threshold(self) -> int:
        return settings.llm_circuit_failure_threshold

    @property
    def cooldown_seconds(self) -> int:
        return settings.llm_health_check_interval

    # ------------------------------------------------------------------ #
    #  内部状态转换
    # ------------------------------------------------------------------ #

    def _transition_to(self, target: _State) -> None:
        prev = self.state
        self.state = target
        if target == _State.CLOSED:
            self._failure_count = 0
        elif target == _State.OPEN:
            self._opened_at = time.time()
        # HALF_OPEN 不重置计数，保留以便探测失败后重新 OPEN

    def _increment_failure(self) -> bool:
        """记录一次失败；返回 True 表示达到阈值应熔断。"""
        self._failure_count += 1
        return self._failure_count >= self.failure_threshold

    # ------------------------------------------------------------------ #
    #  公共方法
    # ------------------------------------------------------------------ #

    def should_allow_primary(self) -> bool:
        """当前请求是否应尝试主模型。"""
        with self._lock:
            if self.state == _State.CLOSED:
                return True
            if self.state == _State.HALF_OPEN:
                return True  # 放行探测
            # OPEN：检查冷却时间
            if self.cooldown_seconds <= 0:
                self._transition_to(_State.CLOSED)
                return True
            if (time.time() - self._opened_at) > self.cooldown_seconds:
                self._transition_to(_State.HALF_OPEN)
                return True  # 进入半开，放行一路
            return False  # 冷却中，拒绝主模型

    def record_success(self) -> None:
        """主模型调用成功。"""
        with self._lock:
            if self.state == _State.HALF_OPEN:
                # 探测成功，关闭熔断器
                self._transition_to(_State.CLOSED)

    def record_failure(self) -> None:
        """主模型调用失败。"""
        with self._lock:
            if self.state == _State.CLOSED:
                if self._increment_failure():
                    self._transition_to(_State.OPEN)
                # 未达阈值：保持 CLOSED，继续重试主模型
            elif self.state == _State.HALF_OPEN:
                # 探测失败，重新熔断
                self._transition_to(_State.OPEN)

    def force_reset(self) -> None:
        """强制重置为 CLOSED（配置变更时使用）。"""
        with self._lock:
            self._transition_to(_State.CLOSED)

    def to_dict(self) -> dict:
        """导出状态，供前端展示。"""
        with self._lock:
            return {
                "state": self.state.value,
                "failure_count": self._failure_count,
                "threshold": self.failure_threshold,
                "cooldown_s": self.cooldown_seconds,
            }


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------
_circuit_breaker = _CircuitBreaker()


# ---------------------------------------------------------------------------
# 嵌入模型（独立，不受熔断影响）
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embedding_model():
    """返回缓存的嵌入模型客户端（OpenAI 兼容）。"""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.api_base,
        api_key=settings.dashscope_api_key,
        # 通义嵌入模型无需 OpenAI 的 token 长度校验
        check_embedding_ctx_length=False,
        # DashScope 兼容接口单次请求最多 10 条，超出会报 400，需限制批大小
        chunk_size=settings.embedding_batch_size,
    )


# ---------------------------------------------------------------------------
# 底层模型缓存（按温度值区分实例）
# ---------------------------------------------------------------------------
def _build_chat(model: str, base_url: str, temperature: float) -> ChatOpenAI:
    """创建 ChatOpenAI 实例。"""
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=settings.dashscope_api_key,
        temperature=temperature,
    )


@lru_cache(maxsize=4)
def _get_primary_model(temperature: float) -> ChatOpenAI:
    """返回主模型实例（按温度缓存）。"""
    return _build_chat(settings.llm_model, settings.api_base, temperature)


@lru_cache(maxsize=4)
def _get_fallback_model(temperature: float) -> ChatOpenAI | None:
    """返回备选模型实例；未配置时返回 None。"""
    if not settings.fallback_llm_model:
        return None
    base_url = settings.fallback_api_base or settings.api_base
    return _build_chat(settings.fallback_llm_model, base_url, temperature)


# ---------------------------------------------------------------------------
# 带熔断降级的 LLM 代理
# ---------------------------------------------------------------------------
class _FallbackLLM:
    """LLM 代理，对外兼容 ChatOpenAI 的 invoke/stream 接口。

    内部使用 ``_CircuitBreaker`` 控制主模型访问：
    - CLOSED：连续失败达阈值才熔断，偶发失败容错
    - OPEN：冷却期内直接走备选模型，不阻塞等待
    - HALF_OPEN：放行一路探测，成功后关闭熔断器
    """

    def __init__(self, temperature: float) -> None:
        self._temperature = temperature

    # ------------------------------------------------------------------ #
    #  invoke
    # ------------------------------------------------------------------ #
    def invoke(self, *args, **kwargs):
        """同步调用。"""
        if _circuit_breaker.should_allow_primary():
            try:
                model = _get_primary_model(self._temperature)
                result = model.invoke(*args, **kwargs)
                _circuit_breaker.record_success()
                return result
            except Exception:
                _circuit_breaker.record_failure()
                # 未熔断（CLOSED 但未达阈值）时，继续向上抛出让调用方感知
                if _circuit_breaker.state != _State.OPEN:
                    raise

        # OPEN 状态或熔断后：走备选
        fallback = _get_fallback_model(self._temperature)
        if fallback is not None:
            return fallback.invoke(*args, **kwargs)
        raise  # 无备选模型

    # ------------------------------------------------------------------ #
    #  stream
    # ------------------------------------------------------------------ #
    def stream(self, *args, **kwargs):
        """流式调用。

        通过消费首个 chunk 检测故障，确保流式场景也能正确记录熔断状态。
        """
        if _circuit_breaker.should_allow_primary():
            try:
                model = _get_primary_model(self._temperature)
                stream_iter = model.stream(*args, **kwargs)
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
            yield from fallback.stream(*args, **kwargs)
            return

    def __repr__(self) -> str:
        cb = _circuit_breaker.to_dict()
        return (
            f"<FallbackLLM primary={settings.llm_model} state={cb['state']} "
            f"failures={cb['failure_count']}/{cb['threshold']}"
            f" fallback={settings.fallback_llm_model or '-'}>"
        )


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------
def get_llm(temperature: float | None = None):
    """返回带熔断降级的对话模型代理。

    ``temperature`` 为 None 时使用配置中的默认温度。
    """
    temp = settings.llm_temperature if temperature is None else temperature
    return _FallbackLLM(temp)


def get_llm_status() -> dict:
    """返回当前熔断器与模型状态，供前端展示。"""
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
    """清空模型缓存并重置熔断器（配置变更后调用）。"""
    get_embedding_model.cache_clear()
    _get_primary_model.cache_clear()
    _get_fallback_model.cache_clear()
    _circuit_breaker.force_reset()