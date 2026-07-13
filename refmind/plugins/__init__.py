"""RefMind 可选插件系统。"""

from .base import (
    CoreHook,
    HookEvent,
    HookOutcome,
    PluginDescriptor,
    PluginDiscoveryReport,
    PluginFailure,
    PluginRegistrar,
    RefMindPlugin,
)
from .manager import (
    DEFAULT_ENTRY_POINT_GROUP,
    DEFAULT_MODULES_ENV,
    PluginManager,
    get_plugin_manager,
)

__all__ = [
    "CoreHook",
    "DEFAULT_ENTRY_POINT_GROUP",
    "DEFAULT_MODULES_ENV",
    "HookEvent",
    "HookOutcome",
    "PluginDescriptor",
    "PluginDiscoveryReport",
    "PluginFailure",
    "PluginManager",
    "PluginRegistrar",
    "RefMindPlugin",
    "get_plugin_manager",
]
