"""RefMind 插件协议与公共数据类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Generic, Mapping, Protocol, TypeVar


T = TypeVar("T")


class CoreHook(str, Enum):
    """内置流程约定的 hook 名称；插件也可以注册自定义名称。"""

    BEFORE_PARSE = "before_parse"
    AFTER_PARSE = "after_parse"
    BEFORE_INGEST = "before_ingest"
    AFTER_INGEST = "after_ingest"
    BEFORE_RETRIEVE = "before_retrieve"
    AFTER_RETRIEVE = "after_retrieve"
    BEFORE_GENERATE = "before_generate"
    AFTER_GENERATE = "after_generate"
    COLLECT_CONTEXT = "collect_context"


@dataclass(frozen=True)
class HookEvent(Generic[T]):
    """传给 hook 的不可变事件；插件通过返回值替换流水线中的 value。"""

    name: str
    value: T
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 冻结 metadata，避免一个插件原地修改后污染其他插件的输入。
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


HookCallback = Callable[[HookEvent[Any]], Any | None]


class PluginRegistrar(Protocol):
    """插件注册阶段可使用的最小接口。"""

    def add_hook(self, hook: str | CoreHook, callback: HookCallback) -> None:
        """为当前插件注册一个回调。"""


class RefMindPlugin(Protocol):
    """推荐的插件对象协议。"""

    name: str

    def register(self, registrar: PluginRegistrar) -> None:
        """声明插件提供的 hook。"""


@dataclass(frozen=True)
class PluginDescriptor:
    """已成功激活的插件信息。"""

    name: str
    version: str = ""
    source: str = "explicit"
    hooks: tuple[str, ...] = ()


@dataclass(frozen=True)
class PluginFailure:
    """被隔离的插件异常，便于 UI/日志展示而不终止主流程。"""

    plugin: str
    phase: str
    error_type: str
    message: str
    hook: str = ""


@dataclass(frozen=True)
class HookOutcome(Generic[T]):
    """hook 链执行结果。"""

    value: T
    failures: tuple[PluginFailure, ...] = ()


@dataclass(frozen=True)
class PluginDiscoveryReport:
    """一次发现操作的可观测结果。"""

    loaded: tuple[PluginDescriptor, ...] = ()
    failures: tuple[PluginFailure, ...] = ()

