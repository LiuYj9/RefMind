from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest
from unittest.mock import patch

from refmind.plugins import CoreHook, PluginManager


class _IncrementPlugin:
    name = "increment"
    version = "1.0"

    def register(self, registrar) -> None:
        registrar.add_hook(CoreHook.AFTER_RETRIEVE, lambda event: event.value + 1)


class _BrokenHookPlugin:
    name = "broken"

    def register(self, registrar) -> None:
        def fail(_event):
            raise RuntimeError("plugin boom")

        registrar.add_hook(CoreHook.AFTER_RETRIEVE, fail)


class PluginManagerTests(unittest.TestCase):
    def test_hook_chain_isolates_failure_and_keeps_transforming(self) -> None:
        manager = PluginManager()
        manager.register_plugin(_IncrementPlugin())
        manager.register_plugin(_BrokenHookPlugin())
        manager.register_plugin(
            lambda registrar: registrar.add_hook(
                CoreHook.AFTER_RETRIEVE,
                lambda event: event.value * 2,
            ),
            name="double",
        )

        result = manager.run_hook(CoreHook.AFTER_RETRIEVE, 2)

        self.assertEqual(result.value, 6)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].plugin, "broken")
        self.assertEqual(result.failures[0].hook, "after_retrieve")

    def test_failed_plugin_cannot_mutate_core_value_in_place(self) -> None:
        manager = PluginManager()

        def mutate_then_fail(event) -> None:
            event.value.clear()
            raise RuntimeError("mutation must be isolated")

        manager.register_plugin(
            lambda registrar: registrar.add_hook(
                CoreHook.AFTER_RETRIEVE, mutate_then_fail
            ),
            name="mutating-broken",
        )
        original = ["evidence"]

        result = manager.run_hook(CoreHook.AFTER_RETRIEVE, original)

        self.assertEqual(original, ["evidence"])
        self.assertEqual(result.value, ["evidence"])

    def test_failed_registration_is_atomic(self) -> None:
        manager = PluginManager()

        def broken_setup(registrar) -> None:
            registrar.add_hook("custom", lambda event: "leaked")
            raise ValueError("setup failed")

        with self.assertRaises(ValueError):
            manager.register_plugin(broken_setup, name="bad-setup")

        self.assertEqual(manager.plugins, ())
        self.assertEqual(manager.run_hook("custom", "original").value, "original")

    def test_async_hook_supports_sync_and_async_callbacks(self) -> None:
        manager = PluginManager()

        async def async_callback(event):
            await asyncio.sleep(0)
            return event.value + "-async"

        manager.register_plugin(
            lambda registrar: registrar.add_hook("custom", async_callback),
            name="async-plugin",
        )
        manager.register_plugin(
            lambda registrar: registrar.add_hook(
                "custom", lambda event: event.value + "-sync"
            ),
            name="sync-plugin",
        )

        result = asyncio.run(manager.run_hook_async("custom", "start"))
        self.assertEqual(result.value, "start-async-sync")

    def test_environment_module_discovery(self) -> None:
        module_name = "refmind_test_plugin_module"
        module = types.ModuleType(module_name)
        module.PLUGIN_NAME = "environment-plugin"

        def register(registrar) -> None:
            registrar.add_hook("custom", lambda event: event.value + "-env")

        module.register = register
        sys.modules[module_name] = module
        try:
            manager = PluginManager()
            with patch.dict(os.environ, {"REFMIND_PLUGIN_MODULES": module_name}):
                report = manager.discover(include_entry_points=False)
            self.assertEqual([plugin.name for plugin in report.loaded], ["environment-plugin"])
            self.assertEqual(manager.run_hook("custom", "ok").value, "ok-env")
        finally:
            sys.modules.pop(module_name, None)

    def test_entry_point_discovery_isolated(self) -> None:
        class EntryPoint:
            name = "entry-plugin"

            def load(self):
                return _IncrementPlugin

        class EntryPoints(list):
            def select(self, **kwargs):
                self.selected_group = kwargs["group"]
                return self

        points = EntryPoints([EntryPoint()])
        manager = PluginManager()
        with patch("refmind.plugins.manager.importlib_metadata.entry_points", return_value=points):
            report = manager.discover(include_environment=False)

        self.assertEqual(points.selected_group, "refmind.plugins")
        self.assertEqual(report.loaded[0].name, "entry-plugin")
        self.assertEqual(manager.run_hook(CoreHook.AFTER_RETRIEVE, 1).value, 2)


if __name__ == "__main__":
    unittest.main()
