"""官方 MCP Python SDK 的可选客户端封装。

本模块只在真正连接时导入 ``mcp``/``httpx``，因此未安装可选依赖、没有
配置 MCP 服务时，RefMind 的核心问答流程仍可正常工作。
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any, Callable, Mapping
from urllib.parse import urlparse


DEFAULT_CONFIG_ENV = "REFMIND_MCP_SERVERS"


class MCPTransport(str, Enum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


class MCPError(RuntimeError):
    """RefMind MCP 集成错误基类。"""


class MCPDependencyMissing(MCPError):
    """尚未安装官方 MCP Python SDK。"""


class MCPNotConnected(MCPError):
    """客户端尚未进入异步上下文。"""


@dataclass(frozen=True)
class MCPServerConfig:
    """单个 MCP 服务配置，支持 stdio 和 Streamable HTTP。"""

    name: str
    transport: MCPTransport | str
    command: str = ""
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    cwd: str | None = None
    url: str = ""
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("MCP 服务 name 不能为空")
        try:
            transport = (
                self.transport
                if isinstance(self.transport, MCPTransport)
                else MCPTransport(str(self.transport).strip().lower().replace("-", "_"))
            )
        except ValueError as exc:
            raise ValueError(f"不支持的 MCP transport: {self.transport}") from exc
        if transport is MCPTransport.STDIO and not self.command.strip():
            raise ValueError("stdio MCP 服务必须配置 command")
        if transport is MCPTransport.STREAMABLE_HTTP and not self.url.strip():
            raise ValueError("Streamable HTTP MCP 服务必须配置 url")
        if transport is MCPTransport.STREAMABLE_HTTP:
            parsed_url = urlparse(self.url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                raise ValueError("Streamable HTTP MCP url 必须是有效的 http(s) 地址")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "args", tuple(str(item) for item in self.args))
        object.__setattr__(self, "env", dict(self.env))
        object.__setattr__(self, "headers", dict(self.headers))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MCPServerConfig":
        """从 JSON/配置映射构造，忽略空的可选字段。"""

        args = value.get("args") or ()
        if isinstance(args, (str, bytes)):
            raise TypeError("MCP args 必须是字符串数组")
        return cls(
            name=str(value.get("name", "")),
            transport=str(value.get("transport", "")),
            command=str(value.get("command", "")),
            args=tuple(args),
            env=dict(value.get("env") or {}),
            cwd=str(value["cwd"]) if value.get("cwd") else None,
            url=str(value.get("url", "")),
            headers=dict(value.get("headers") or {}),
            timeout_seconds=float(value.get("timeout_seconds", 30.0)),
        )


@dataclass(frozen=True)
class MCPToolInfo:
    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MCPResourceInfo:
    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = ""


@dataclass(frozen=True)
class MCPPromptInfo:
    name: str
    description: str = ""


@dataclass(frozen=True)
class MCPContent:
    """将 SDK 的多种内容块归一化为稳定的 RefMind 边界类型。"""

    kind: str
    text: str = ""
    data: Any = None
    mime_type: str = ""
    uri: str = ""


@dataclass(frozen=True)
class MCPCallResult:
    content: tuple[MCPContent, ...] = ()
    structured_content: Mapping[str, Any] | None = None
    is_error: bool = False

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.content if block.text)


@dataclass(frozen=True)
class MCPProbe:
    """服务能力探测结果；连接失败也作为普通结果返回。"""

    server: str
    sdk_available: bool
    connected: bool
    protocol_version: str = ""
    server_info: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    tools: tuple[MCPToolInfo, ...] = ()
    resources: tuple[MCPResourceInfo, ...] = ()
    prompts: tuple[MCPPromptInfo, ...] = ()
    errors: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPConfigurationResult:
    servers: tuple[MCPServerConfig, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SDKBindings:
    ClientSession: Any
    StdioServerParameters: Any
    stdio_client: Callable[..., Any]
    streamable_http_client: Callable[..., Any]
    AsyncClient: Any
    Timeout: Any
    CancelScope: Any
    types: Any


def mcp_sdk_available() -> bool:
    """不导入依赖，仅判断官方 SDK 是否可发现。"""

    try:
        return importlib.util.find_spec("mcp") is not None
    except (ImportError, ValueError):
        return False


def _load_sdk_bindings() -> _SDKBindings:
    """真正使用 MCP 时才导入官方 SDK，保持依赖完全可选。"""

    try:
        mcp = importlib.import_module("mcp")
        mcp_types = importlib.import_module("mcp.types")
        stdio = importlib.import_module("mcp.client.stdio")
        streamable_http = importlib.import_module("mcp.client.streamable_http")
        httpx = importlib.import_module("httpx")
        anyio = importlib.import_module("anyio")
    except (ImportError, ModuleNotFoundError) as exc:
        raise MCPDependencyMissing(
            "MCP 功能未启用；请安装稳定版官方 Python SDK：mcp>=1.27,<2"
        ) from exc
    try:
        return _SDKBindings(
            ClientSession=mcp.ClientSession,
            StdioServerParameters=mcp.StdioServerParameters,
            stdio_client=stdio.stdio_client,
            # 官方 v1.x 使用带下划线的名称；旧 streamablehttp_client 已弃用。
            streamable_http_client=streamable_http.streamable_http_client,
            AsyncClient=httpx.AsyncClient,
            Timeout=httpx.Timeout,
            CancelScope=anyio.CancelScope,
            types=mcp_types,
        )
    except AttributeError as exc:
        raise MCPDependencyMissing(
            "当前 mcp 包接口不兼容；RefMind 需要稳定版 mcp>=1.27,<2"
        ) from exc


def _attribute(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
        if isinstance(value, Mapping) and name in value:
            return value[name]
    return default


def _model_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json", by_alias=True, exclude_none=True))
    return {
        key: item
        for key, item in vars(value).items()
        if not key.startswith("_")
    } if hasattr(value, "__dict__") else {"value": str(value)}


def _normalize_content(block: Any) -> MCPContent:
    kind = str(_attribute(block, "type", default=type(block).__name__))
    text = str(_attribute(block, "text", default="") or "")
    mime_type = str(_attribute(block, "mimeType", "mime_type", default="") or "")
    data = _attribute(block, "data", default=None)
    if data is None and not text:
        data = _model_mapping(block)
    return MCPContent(kind=kind, text=text, data=data, mime_type=mime_type)


def _normalize_resource_content(block: Any) -> MCPContent:
    """归一化 TextResourceContents/BlobResourceContents，并保留块级 URI。"""
    text = str(_attribute(block, "text", default="") or "")
    blob = _attribute(block, "blob", default=None)
    return MCPContent(
        kind="text" if text else "blob",
        text=text,
        data=blob,
        mime_type=str(_attribute(block, "mimeType", "mime_type", default="") or ""),
        uri=str(_attribute(block, "uri", default="") or ""),
    )


class MCPClient:
    """一个连接对应一个异步上下文的轻量 MCP 客户端。"""

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        bindings_loader: Callable[[], _SDKBindings] = _load_sdk_bindings,
    ) -> None:
        self.config = config
        self._bindings_loader = bindings_loader
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self._initialize_result: Any = None
        self._bindings: _SDKBindings | None = None

    async def __aenter__(self) -> "MCPClient":
        if self._stack is not None:
            raise MCPError("MCPClient 不支持重复进入同一上下文")
        bindings = self._bindings_loader()
        self._bindings = bindings
        stack = AsyncExitStack()
        self._stack = stack
        try:
            if self.config.transport is MCPTransport.STDIO:
                params = bindings.StdioServerParameters(
                    command=self.config.command,
                    args=list(self.config.args),
                    env=dict(self.config.env) or None,
                    cwd=self.config.cwd,
                )
                read_stream, write_stream = await stack.enter_async_context(
                    bindings.stdio_client(params)
                )
            else:
                # v1.28 建议通过预配置 httpx.AsyncClient 传 headers/timeout。
                http_client = bindings.AsyncClient(
                    headers=dict(self.config.headers),
                    # MCP 官方默认允许长时间 SSE 读取，并跟随常见的 307/308 重定向。
                    timeout=bindings.Timeout(
                        self.config.timeout_seconds,
                        read=max(300.0, self.config.timeout_seconds),
                    ),
                    follow_redirects=True,
                )
                await stack.enter_async_context(http_client)
                read_stream, write_stream, _ = await stack.enter_async_context(
                    bindings.streamable_http_client(
                        self.config.url,
                        http_client=http_client,
                    )
                )

            session = bindings.ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self.config.timeout_seconds),
            )
            self._session = await stack.enter_async_context(session)
            self._initialize_result = await self._session.initialize()
            return self
        except BaseException as exc:
            try:
                # 取消请求时也要让 session/transport 完成异步清理。
                with bindings.CancelScope(shield=True):
                    await stack.aclose()
            except BaseException as cleanup_exc:
                # 保留最初的连接异常，清理异常作为诊断附注而不是覆盖根因。
                if hasattr(exc, "add_note"):
                    exc.add_note(f"MCP 连接清理也失败：{cleanup_exc}")
            finally:
                self._stack = None
                self._session = None
                self._initialize_result = None
                self._bindings = None
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        stack = self._stack
        bindings = self._bindings
        self._stack = None
        self._session = None
        self._initialize_result = None
        self._bindings = None
        if stack is not None:
            if bindings is None:
                await stack.__aexit__(exc_type, exc, traceback)
            else:
                with bindings.CancelScope(shield=True):
                    await stack.__aexit__(exc_type, exc, traceback)

    def _require_session(self) -> Any:
        if self._session is None:
            raise MCPNotConnected("请先使用 'async with MCPClient(...)'")
        return self._session

    async def ping(self) -> None:
        await self._require_session().send_ping()

    async def list_tools(self) -> tuple[MCPToolInfo, ...]:
        raw_tools = await self._list_all("list_tools", "tools")
        return tuple(
            MCPToolInfo(
                name=str(_attribute(tool, "name", default="")),
                description=str(_attribute(tool, "description", default="") or ""),
                input_schema=dict(
                    _attribute(tool, "inputSchema", "input_schema", default={}) or {}
                ),
                output_schema=(
                    dict(schema)
                    if (schema := _attribute(tool, "outputSchema", "output_schema")) is not None
                    else None
                ),
            )
            for tool in raw_tools
        )

    async def list_resources(self) -> tuple[MCPResourceInfo, ...]:
        raw_resources = await self._list_all("list_resources", "resources")
        return tuple(
            MCPResourceInfo(
                uri=str(_attribute(resource, "uri", default="")),
                name=str(_attribute(resource, "name", default="") or ""),
                description=str(_attribute(resource, "description", default="") or ""),
                mime_type=str(_attribute(resource, "mimeType", "mime_type", default="") or ""),
            )
            for resource in raw_resources
        )

    async def list_prompts(self) -> tuple[MCPPromptInfo, ...]:
        raw_prompts = await self._list_all("list_prompts", "prompts")
        return tuple(
            MCPPromptInfo(
                name=str(_attribute(prompt, "name", default="")),
                description=str(_attribute(prompt, "description", default="") or ""),
            )
            for prompt in raw_prompts
        )

    async def _list_all(self, method_name: str, field_name: str) -> tuple[Any, ...]:
        """遍历官方 SDK 的 cursor 分页，并防御服务端重复 cursor。"""

        session = self._require_session()
        method = getattr(session, method_name)
        response = await method()
        items: list[Any] = list(_attribute(response, field_name, default=()))
        cursor = _attribute(response, "nextCursor", "next_cursor", default=None)
        seen_cursors: set[str] = set()
        page_count = 1
        while cursor:
            cursor_text = str(cursor)
            if cursor_text in seen_cursors:
                raise MCPError(f"{method_name} 返回了重复 cursor")
            if page_count >= 100 or len(items) >= 10_000:
                raise MCPError(f"{method_name} 分页超过安全上限")
            seen_cursors.add(cursor_text)
            bindings = self._bindings
            if bindings is None:
                raise MCPNotConnected("MCP SDK binding 已释放")
            params = bindings.types.PaginatedRequestParams(cursor=cursor_text)
            response = await method(params=params)
            items.extend(_attribute(response, field_name, default=()))
            page_count += 1
            cursor = _attribute(response, "nextCursor", "next_cursor", default=None)
        if len(items) > 10_000:
            raise MCPError(f"{method_name} 条目超过安全上限")
        return tuple(items)

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None = None
    ) -> MCPCallResult:
        if not name.strip():
            raise ValueError("MCP tool name 不能为空")
        result = await self._require_session().call_tool(
            name,
            arguments=dict(arguments or {}),
        )
        structured = _attribute(
            result, "structuredContent", "structured_content", default=None
        )
        return MCPCallResult(
            content=tuple(
                _normalize_content(block)
                for block in _attribute(result, "content", default=())
            ),
            structured_content=dict(structured) if structured is not None else None,
            is_error=bool(_attribute(result, "isError", "is_error", default=False)),
        )

    async def read_resource(self, uri: str) -> tuple[MCPContent, ...]:
        if not uri.strip():
            raise ValueError("MCP resource URI 不能为空")
        result = await self._require_session().read_resource(uri)
        return tuple(
            _normalize_resource_content(block)
            for block in _attribute(result, "contents", default=())
        )

    async def probe(self) -> MCPProbe:
        """探测当前连接；单项能力失败不会影响其他列表。"""

        session = self._require_session()
        errors: dict[str, str] = {}

        async def collect(label: str, operation: Callable[[], Any], empty: Any) -> Any:
            try:
                return await operation()
            except Exception as exc:  # 不同服务可能只实现部分 capability
                errors[label] = f"{type(exc).__name__}: {exc}"
                return empty

        await collect("ping", self.ping, None)
        tools = await collect("tools", self.list_tools, ())
        resources = await collect("resources", self.list_resources, ())
        prompts = await collect("prompts", self.list_prompts, ())
        initialized = self._initialize_result
        capabilities = _attribute(initialized, "capabilities", default=None)
        if capabilities is None and hasattr(session, "get_server_capabilities"):
            capabilities = session.get_server_capabilities()
        return MCPProbe(
            server=self.config.name,
            sdk_available=True,
            connected=True,
            protocol_version=str(
                _attribute(initialized, "protocolVersion", "protocol_version", default="")
            ),
            server_info=_model_mapping(
                _attribute(initialized, "serverInfo", "server_info", default=None)
            ),
            capabilities=_model_mapping(capabilities),
            tools=tools,
            resources=resources,
            prompts=prompts,
            errors=errors,
        )


def load_mcp_server_configs(
    env_var: str = DEFAULT_CONFIG_ENV,
) -> MCPConfigurationResult:
    """从 JSON 环境变量读取服务列表；无配置或坏配置均不会中断应用。"""

    raw = os.getenv(env_var, "").strip()
    if not raw:
        return MCPConfigurationResult()
    try:
        values = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        return MCPConfigurationResult(errors=(f"{env_var}: 非法 JSON ({exc})",))
    if not isinstance(values, list):
        return MCPConfigurationResult(errors=(f"{env_var}: 顶层必须是数组",))

    servers: list[MCPServerConfig] = []
    errors: list[str] = []
    seen_names: set[str] = set()
    for index, value in enumerate(values):
        try:
            if not isinstance(value, Mapping):
                raise TypeError("服务配置必须是对象")
            config = MCPServerConfig.from_mapping(value)
            if config.name in seen_names:
                raise ValueError(f"MCP 服务名称重复: {config.name}")
            seen_names.add(config.name)
            servers.append(config)
        except (TypeError, ValueError) as exc:
            errors.append(f"{env_var}[{index}]: {exc}")
    return MCPConfigurationResult(tuple(servers), tuple(errors))


class MCPManager:
    """多个可选 MCP 服务的统一 probe/list/call 入口。"""

    def __init__(
        self,
        configs: tuple[MCPServerConfig, ...] = (),
        *,
        client_factory: Callable[[MCPServerConfig], MCPClient] = MCPClient,
        configuration_errors: tuple[str, ...] = (),
    ) -> None:
        self._configs: dict[str, MCPServerConfig] = {}
        self._client_factory = client_factory
        self.configuration_errors = configuration_errors
        for config in configs:
            if config.name in self._configs:
                raise ValueError(f"MCP 服务名称重复: {config.name}")
            self._configs[config.name] = config

    @classmethod
    def from_environment(
        cls,
        env_var: str = DEFAULT_CONFIG_ENV,
        *,
        client_factory: Callable[[MCPServerConfig], MCPClient] = MCPClient,
    ) -> "MCPManager":
        loaded = load_mcp_server_configs(env_var)
        return cls(
            loaded.servers,
            client_factory=client_factory,
            configuration_errors=loaded.errors,
        )

    @property
    def servers(self) -> tuple[str, ...]:
        return tuple(self._configs)

    def get_config(self, server: str) -> MCPServerConfig | None:
        return self._configs.get(server)

    async def probe(self, server: str) -> MCPProbe:
        config = self._configs.get(server)
        if config is None:
            return MCPProbe(
                server=server,
                sdk_available=mcp_sdk_available(),
                connected=False,
                errors={"configuration": f"未配置 MCP 服务: {server}"},
            )
        try:
            async with self._client_factory(config) as client:
                return await client.probe()
        except Exception as exc:
            return MCPProbe(
                server=server,
                sdk_available=not isinstance(exc, MCPDependencyMissing),
                connected=False,
                errors={"connection": f"{type(exc).__name__}: {exc}"},
            )

    async def probe_all(self) -> tuple[MCPProbe, ...]:
        """没有配置时返回空元组，作为可选能力安全降级。"""

        if not self._configs:
            return ()
        return tuple(await asyncio.gather(*(self.probe(name) for name in self._configs)))

    async def list_tools(self, server: str) -> tuple[MCPToolInfo, ...]:
        config = self._require_config(server)
        async with self._client_factory(config) as client:
            return await client.list_tools()

    async def list_resources(self, server: str) -> tuple[MCPResourceInfo, ...]:
        config = self._require_config(server)
        async with self._client_factory(config) as client:
            return await client.list_resources()

    async def list_prompts(self, server: str) -> tuple[MCPPromptInfo, ...]:
        config = self._require_config(server)
        async with self._client_factory(config) as client:
            return await client.list_prompts()

    async def call_tool(
        self,
        server: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> MCPCallResult:
        config = self._require_config(server)
        async with self._client_factory(config) as client:
            return await client.call_tool(tool, arguments)

    async def read_resource(
        self, server: str, uri: str
    ) -> tuple[MCPContent, ...]:
        config = self._require_config(server)
        async with self._client_factory(config) as client:
            return await client.read_resource(uri)

    def _require_config(self, server: str) -> MCPServerConfig:
        config = self._configs.get(server)
        if config is None:
            raise MCPError(f"未配置 MCP 服务: {server}")
        return config
