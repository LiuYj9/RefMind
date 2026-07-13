"""轻量插件管理器：显式注册、模块/entry point 发现与异常隔离。"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from copy import deepcopy
from collections import defaultdict, deque
from importlib import metadata as importlib_metadata
from threading import RLock
from types import ModuleType
from typing import Any, Iterable, Mapping

from .base import (
    CoreHook,
    HookCallback,
    HookEvent,
    HookOutcome,
    PluginDescriptor,
    PluginDiscoveryReport,
    PluginFailure,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_ENTRY_POINT_GROUP = "refmind.plugins"
DEFAULT_MODULES_ENV = "REFMIND_PLUGIN_MODULES"


def _hook_name(hook: str | CoreHook) -> str:
    name = hook.value if isinstance(hook, CoreHook) else str(hook).strip()
    if not name:
        raise ValueError("hook 名称不能为空")
    return name


class _ScopedRegistrar:
    """先暂存 hook，注册函数全部成功后再一次性提交。"""

    def __init__(self) -> None:
        self.hooks: list[tuple[str, HookCallback]] = []

    def add_hook(self, hook: str | CoreHook, callback: HookCallback) -> None:
        if not callable(callback):
            raise TypeError("hook callback 必须可调用")
        self.hooks.append((_hook_name(hook), callback))


class PluginManager:
    """线程安全的插件/hook 注册表。

    hook 按注册顺序串行执行。回调返回 ``None`` 表示只观察事件；返回其他值
    表示把该值传给后续插件。单个插件异常只会被记录，不会打断主流程。
    """

    def __init__(self) -> None:
        self._lock = RLock()
        # 同步 hook 可能在 multi-agent 检索线程中并发触发，串行执行保护插件内部状态。
        self._execution_lock = RLock()
        self._plugins: dict[str, PluginDescriptor] = {}
        self._hooks: dict[str, list[tuple[str, HookCallback]]] = defaultdict(list)
        self._failures: deque[PluginFailure] = deque(maxlen=200)

    @property
    def plugins(self) -> tuple[PluginDescriptor, ...]:
        with self._lock:
            return tuple(self._plugins.values())

    @property
    def failures(self) -> tuple[PluginFailure, ...]:
        with self._lock:
            return tuple(self._failures)

    def register_plugin(
        self,
        plugin: Any,
        *,
        name: str | None = None,
        source: str = "explicit",
        replace: bool = False,
    ) -> PluginDescriptor:
        """显式注册插件对象、插件类或 ``register(registrar)`` 函数。"""

        candidate = plugin() if inspect.isclass(plugin) else plugin
        plugin_name = (name or getattr(candidate, "name", "")).strip()
        if not plugin_name:
            plugin_name = getattr(candidate, "__name__", "").strip()
        if not plugin_name:
            raise ValueError("插件必须提供 name，或在注册时显式传入 name")

        with self._lock:
            if plugin_name in self._plugins and not replace:
                raise ValueError(f"插件已注册: {plugin_name}")

        setup = getattr(candidate, "register", None)
        if not callable(setup):
            setup = candidate if callable(candidate) else None
        if setup is None:
            raise TypeError(f"插件 {plugin_name} 未提供 register(registrar)")

        registrar = _ScopedRegistrar()
        # 注册阶段先写临时 registrar，避免失败插件留下半套 hook。
        setup(registrar)
        hook_names = tuple(hook for hook, _ in registrar.hooks)
        descriptor = PluginDescriptor(
            name=plugin_name,
            version=str(getattr(candidate, "version", "")),
            source=source,
            hooks=hook_names,
        )

        with self._lock:
            if replace and plugin_name in self._plugins:
                self._remove_locked(plugin_name)
            elif plugin_name in self._plugins:
                # 处理两个线程同时注册同名插件的竞态。
                raise ValueError(f"插件已注册: {plugin_name}")
            self._plugins[plugin_name] = descriptor
            for hook, callback in registrar.hooks:
                self._hooks[hook].append((plugin_name, callback))
        return descriptor

    def unregister(self, name: str) -> bool:
        with self._lock:
            if name not in self._plugins:
                return False
            self._remove_locked(name)
            return True

    def _remove_locked(self, name: str) -> None:
        self._plugins.pop(name, None)
        for hook_name in tuple(self._hooks):
            callbacks = [item for item in self._hooks[hook_name] if item[0] != name]
            if callbacks:
                self._hooks[hook_name] = callbacks
            else:
                del self._hooks[hook_name]

    def run_hook(
        self,
        hook: str | CoreHook,
        value: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> HookOutcome[Any]:
        """在同步流程执行 hook；异步回调应改用 :meth:`run_hook_async`。"""

        name = _hook_name(hook)
        with self._lock:
            callbacks = tuple(self._hooks.get(name, ()))

        failures: list[PluginFailure] = []
        current = value
        for plugin_name, callback in callbacks:
            try:
                # 回调只拿到当前值的副本：先原地修改再抛错也不会污染核心值。
                plugin_value = deepcopy(current)
                with self._execution_lock:
                    result = callback(HookEvent(name, plugin_value, metadata or {}))
                if inspect.isawaitable(result):
                    # 主动关闭 coroutine，避免错误用法额外触发未等待警告。
                    close = getattr(result, "close", None)
                    if callable(close):
                        close()
                    raise TypeError("同步 hook 返回了 awaitable，请使用 run_hook_async")
                if result is not None:
                    current = result
            except Exception as exc:  # 插件边界必须隔离第三方异常
                failures.append(self._record_failure(plugin_name, "hook", exc, name))
        return HookOutcome(current, tuple(failures))

    async def run_hook_async(
        self,
        hook: str | CoreHook,
        value: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> HookOutcome[Any]:
        """执行同步或异步 hook，同时保留注册顺序。"""

        name = _hook_name(hook)
        with self._lock:
            callbacks = tuple(self._hooks.get(name, ()))

        failures: list[PluginFailure] = []
        current = value
        for plugin_name, callback in callbacks:
            try:
                plugin_value = deepcopy(current)
                result = callback(HookEvent(name, plugin_value, metadata or {}))
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    current = result
            except Exception as exc:  # 插件边界必须隔离第三方异常
                failures.append(self._record_failure(plugin_name, "hook", exc, name))
        return HookOutcome(current, tuple(failures))

    def discover(
        self,
        *,
        modules: Iterable[str] = (),
        include_environment: bool = True,
        include_entry_points: bool = True,
        env_var: str = DEFAULT_MODULES_ENV,
        entry_point_group: str = DEFAULT_ENTRY_POINT_GROUP,
    ) -> PluginDiscoveryReport:
        """发现插件；失败项被收集到报告，其他插件仍会继续加载。"""

        loaded: list[PluginDescriptor] = []
        failures: list[PluginFailure] = []
        module_names = [name.strip() for name in modules if name.strip()]
        if include_environment:
            module_names.extend(
                name.strip() for name in os.getenv(env_var, "").split(",") if name.strip()
            )

        # 保序去重，避免参数和环境变量同时声明同一模块。
        for module_name in dict.fromkeys(module_names):
            try:
                module = importlib.import_module(module_name)
                candidate = self._candidate_from_module(module)
                loaded.append(
                    self.register_plugin(
                        candidate,
                        name=self._candidate_name(module),
                        source=f"module:{module_name}",
                    )
                )
            except Exception as exc:
                failures.append(self._record_failure(module_name, "discover", exc))

        if include_entry_points:
            try:
                entry_points = importlib_metadata.entry_points()
                selected = (
                    entry_points.select(group=entry_point_group)
                    if hasattr(entry_points, "select")
                    else entry_points.get(entry_point_group, ())
                )
            except Exception as exc:
                failures.append(self._record_failure(entry_point_group, "entry_points", exc))
                selected = ()
            for entry_point in selected:
                entry_name = getattr(entry_point, "name", repr(entry_point))
                try:
                    loaded.append(
                        self.register_plugin(
                            entry_point.load(),
                            name=entry_name,
                            source=f"entry_point:{entry_point_group}",
                        )
                    )
                except Exception as exc:
                    failures.append(self._record_failure(entry_name, "discover", exc))

        return PluginDiscoveryReport(tuple(loaded), tuple(failures))

    @staticmethod
    def _candidate_from_module(module: ModuleType) -> Any:
        # 模块可导出 plugin 对象，也可直接提供 register(registrar)。
        return getattr(module, "plugin", module)

    @staticmethod
    def _candidate_name(module: ModuleType) -> str:
        candidate = getattr(module, "plugin", module)
        return str(
            getattr(candidate, "name", "")
            or getattr(module, "PLUGIN_NAME", "")
            or module.__name__
        )

    def _record_failure(
        self, plugin: str, phase: str, exc: Exception, hook: str = ""
    ) -> PluginFailure:
        failure = PluginFailure(
            plugin=plugin,
            phase=phase,
            error_type=type(exc).__name__,
            message=str(exc),
            hook=hook,
        )
        with self._lock:
            self._failures.append(failure)
        LOGGER.warning(
            "RefMind plugin failure: plugin=%s phase=%s hook=%s",
            plugin,
            phase,
            hook,
            exc_info=exc,
        )
        return failure


_DEFAULT_MANAGER: PluginManager | None = None
_DEFAULT_MANAGER_DISCOVERED = False
_DEFAULT_MANAGER_LOCK = RLock()


def get_plugin_manager(*, discover: bool = True) -> PluginManager:
    """返回进程级管理器；首次获取时可发现环境模块和 entry points。"""

    global _DEFAULT_MANAGER, _DEFAULT_MANAGER_DISCOVERED
    with _DEFAULT_MANAGER_LOCK:
        if _DEFAULT_MANAGER is None:
            _DEFAULT_MANAGER = PluginManager()
        if discover and not _DEFAULT_MANAGER_DISCOVERED:
            _DEFAULT_MANAGER.discover()
            _DEFAULT_MANAGER_DISCOVERED = True
        return _DEFAULT_MANAGER
