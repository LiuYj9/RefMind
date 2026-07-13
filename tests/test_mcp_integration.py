from __future__ import annotations

import asyncio
import json
import os
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

from refmind.integrations import (
    ExternalContextBundle,
    MCPClient,
    MCPContextProvider,
    MCPDependencyMissing,
    MCPError,
    MCPManager,
    MCPServerConfig,
    MCPTransport,
    load_mcp_server_configs,
)
from refmind.integrations.mcp import _SDKBindings


class _FakeParameters:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _FakeHTTPClient:
    instances = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _FakeTimeout:
    def __init__(self, timeout, **kwargs) -> None:
        self.timeout = timeout
        self.kwargs = kwargs


class _FakeCancelScope:
    def __init__(self, **_kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class _FakeSession:
    def __init__(self, read, write, **kwargs) -> None:
        self.read = read
        self.write = write
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def initialize(self):
        return {
            "protocolVersion": "2025-06-18",
            "serverInfo": {"name": "fake-server", "version": "1"},
            "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
        }

    async def send_ping(self):
        return None

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                {
                    "name": "search",
                    "description": "Search papers",
                    "inputSchema": {"type": "object"},
                    "outputSchema": {"type": "object"},
                }
            ]
        )

    async def list_resources(self):
        return SimpleNamespace(
            resources=[
                {
                    "uri": "paper://1",
                    "name": "Paper",
                    "mimeType": "text/plain",
                }
            ]
        )

    async def list_prompts(self):
        return SimpleNamespace(prompts=[{"name": "review", "description": "Review"}])

    async def call_tool(self, name, arguments):
        return {
            "content": [{"type": "text", "text": f"{name}:{arguments['query']}"}],
            "structuredContent": {"count": 1},
            "isError": False,
        }

    async def read_resource(self, uri):
        return SimpleNamespace(
            contents=[
                {
                    "uri": uri,
                    "text": f"content:{uri}",
                    "mimeType": "text/plain",
                }
            ]
        )


class _FakeBindingsFactory:
    def __init__(self, session_class=_FakeSession) -> None:
        self.stdio_parameters = None
        self.http_call = None
        self.session_class = session_class

    def bindings(self) -> _SDKBindings:
        factory = self

        @asynccontextmanager
        async def stdio_client(parameters):
            factory.stdio_parameters = parameters
            yield "stdio-read", "stdio-write"

        @asynccontextmanager
        async def streamable_http_client(url, **kwargs):
            factory.http_call = (url, kwargs)
            yield "http-read", "http-write", lambda: "session-id"

        return _SDKBindings(
            ClientSession=self.session_class,
            StdioServerParameters=_FakeParameters,
            stdio_client=stdio_client,
            streamable_http_client=streamable_http_client,
            AsyncClient=_FakeHTTPClient,
            Timeout=_FakeTimeout,
            CancelScope=_FakeCancelScope,
            types=SimpleNamespace(PaginatedRequestParams=_FakeParameters),
        )


class MCPIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeHTTPClient.instances.clear()

    def test_stdio_probe_list_and_call_with_fake_sdk(self) -> None:
        factory = _FakeBindingsFactory()
        config = MCPServerConfig(
            name="local",
            transport="stdio",
            command="python",
            args=("server.py",),
            env={"TOKEN": "secret"},
        )

        async def exercise():
            async with MCPClient(config, bindings_loader=factory.bindings) as client:
                probe = await client.probe()
                result = await client.call_tool("search", {"query": "rag"})
                return probe, result

        probe, result = asyncio.run(exercise())
        self.assertTrue(probe.connected)
        self.assertEqual(probe.protocol_version, "2025-06-18")
        self.assertEqual(probe.tools[0].name, "search")
        self.assertEqual(probe.resources[0].uri, "paper://1")
        self.assertEqual(result.text, "search:rag")
        self.assertEqual(result.structured_content, {"count": 1})
        self.assertEqual(factory.stdio_parameters.command, "python")

    def test_streamable_http_uses_preconfigured_http_client(self) -> None:
        factory = _FakeBindingsFactory()
        config = MCPServerConfig(
            name="remote",
            transport=MCPTransport.STREAMABLE_HTTP,
            url="https://example.test/mcp",
            headers={"Authorization": "Bearer hidden"},
            timeout_seconds=12,
        )

        async def exercise():
            async with MCPClient(config, bindings_loader=factory.bindings) as client:
                return await client.list_tools()

        tools = asyncio.run(exercise())
        self.assertEqual(tools[0].name, "search")
        self.assertEqual(factory.http_call[0], "https://example.test/mcp")
        http_client = factory.http_call[1]["http_client"]
        timeout = http_client.kwargs["timeout"]
        self.assertEqual(timeout.timeout, 12)
        self.assertEqual(timeout.kwargs["read"], 300)
        self.assertTrue(http_client.kwargs["follow_redirects"])
        self.assertEqual(
            http_client.kwargs["headers"]["Authorization"],
            "Bearer hidden",
        )

    def test_missing_sdk_degrades_to_probe_result(self) -> None:
        config = MCPServerConfig(name="local", transport="stdio", command="server")

        def missing_loader():
            raise MCPDependencyMissing("not installed")

        manager = MCPManager(
            (config,),
            client_factory=lambda item: MCPClient(item, bindings_loader=missing_loader),
        )
        probe = asyncio.run(manager.probe("local"))
        self.assertFalse(probe.connected)
        self.assertFalse(probe.sdk_available)
        self.assertIn("not installed", probe.errors["connection"])

    def test_no_configuration_is_a_safe_noop(self) -> None:
        manager = MCPManager()
        self.assertEqual(manager.servers, ())
        self.assertEqual(asyncio.run(manager.probe_all()), ())

    def test_context_provider_does_not_feed_answer_by_default(self) -> None:
        factory = _FakeBindingsFactory()
        config = MCPServerConfig(name="local", transport="stdio", command="server")
        provider = MCPContextProvider(
            config,
            tool_name="search",
            client_factory=lambda item: MCPClient(item, bindings_loader=factory.bindings),
        )

        bundle = asyncio.run(provider.provide("retrieval"))
        self.assertTrue(bundle.available)
        self.assertEqual(bundle.items[0].content.splitlines()[0], "search:retrieval")
        self.assertEqual(bundle.render_for_answer(), "")

        allowed_provider = MCPContextProvider(
            config,
            resource_uri="paper://1",
            allow_in_answers=True,
            client_factory=lambda item: MCPClient(item, bindings_loader=factory.bindings),
        )
        allowed = asyncio.run(allowed_provider.provide("ignored"))
        self.assertIn("content:paper://1", allowed.render_for_answer())

    def test_resource_normalization_keeps_uri_and_mime_type(self) -> None:
        factory = _FakeBindingsFactory()
        config = MCPServerConfig(name="local", transport="stdio", command="server")

        async def exercise():
            async with MCPClient(config, bindings_loader=factory.bindings) as client:
                return await client.read_resource("paper://1")

        contents = asyncio.run(exercise())
        self.assertEqual(contents[0].kind, "text")
        self.assertEqual(contents[0].uri, "paper://1")
        self.assertEqual(contents[0].mime_type, "text/plain")

    def test_unconfigured_provider_returns_empty_bundle(self) -> None:
        bundle = asyncio.run(MCPContextProvider(None).provide("query"))
        self.assertIsInstance(bundle, ExternalContextBundle)
        self.assertFalse(bundle.available)
        self.assertEqual(bundle.items, ())

    def test_environment_json_handles_partial_invalid_configuration(self) -> None:
        raw = json.dumps(
            [
                {"name": "ok", "transport": "stdio", "command": "server"},
                {"name": "bad", "transport": "stdio"},
            ]
        )
        with patch.dict(os.environ, {"REFMIND_MCP_SERVERS": raw}):
            result = load_mcp_server_configs()
        self.assertEqual([config.name for config in result.servers], ["ok"])
        self.assertEqual(len(result.errors), 1)

    def test_environment_json_isolates_duplicate_server_names(self) -> None:
        raw = json.dumps(
            [
                {"name": "same", "transport": "stdio", "command": "one"},
                {"name": "same", "transport": "stdio", "command": "two"},
            ]
        )
        with patch.dict(os.environ, {"REFMIND_MCP_SERVERS": raw}):
            result = load_mcp_server_configs()

        self.assertEqual([config.command for config in result.servers], ["one"])
        self.assertEqual(len(result.errors), 1)

    def test_repeated_pagination_cursor_is_rejected(self) -> None:
        class RepeatingCursorSession(_FakeSession):
            async def list_tools(self, params=None):
                return {
                    "tools": [],
                    "nextCursor": "same-cursor",
                }

        factory = _FakeBindingsFactory(RepeatingCursorSession)
        config = MCPServerConfig(name="local", transport="stdio", command="server")

        async def exercise():
            async with MCPClient(config, bindings_loader=factory.bindings) as client:
                await client.list_tools()

        with self.assertRaisesRegex(MCPError, "重复 cursor"):
            asyncio.run(exercise())

    def test_failed_connection_clears_state_even_when_cleanup_fails(self) -> None:
        class FailingInitializeSession(_FakeSession):
            async def initialize(self):
                raise ValueError("initialize failed")

        @asynccontextmanager
        async def bad_stdio(_parameters):
            try:
                yield "read", "write"
            finally:
                raise RuntimeError("cleanup failed")

        bindings = _SDKBindings(
            ClientSession=FailingInitializeSession,
            StdioServerParameters=_FakeParameters,
            stdio_client=bad_stdio,
            streamable_http_client=lambda *_args, **_kwargs: None,
            AsyncClient=_FakeHTTPClient,
            Timeout=_FakeTimeout,
            CancelScope=_FakeCancelScope,
            types=SimpleNamespace(PaginatedRequestParams=_FakeParameters),
        )
        client = MCPClient(
            MCPServerConfig(name="local", transport="stdio", command="server"),
            bindings_loader=lambda: bindings,
        )

        with self.assertRaisesRegex(ValueError, "initialize failed"):
            asyncio.run(client.__aenter__())

        self.assertIsNone(client._stack)
        self.assertIsNone(client._session)
        self.assertIsNone(client._bindings)


if __name__ == "__main__":
    unittest.main()
