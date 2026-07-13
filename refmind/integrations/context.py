"""把 MCP 工具或资源适配为 RefMind 可审计的外部上下文。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .mcp import MCPCallResult, MCPClient, MCPContent, MCPServerConfig


@dataclass(frozen=True)
class ExternalContextItem:
    content: str
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # MCP 属于外部数据源，默认不允许进入最终答案的 prompt。
    allow_answer_use: bool = False

    def __post_init__(self) -> None:
        # 同插件事件一样冻结 metadata，保证审计字段不会被下游意外篡改。
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ExternalContextBundle:
    items: tuple[ExternalContextItem, ...] = ()
    available: bool = True
    error: str = ""

    def render_for_answer(self) -> str:
        """只渲染显式获准的内容；默认配置下必然返回空字符串。"""

        return "\n\n".join(
            f"[外部来源: {item.source}]\n{item.content}"
            for item in self.items
            if item.allow_answer_use and item.content
        )


class MCPContextProvider:
    """按需调用一个 MCP tool/resource 的上下文提供者。

    ``allow_in_answers`` 默认为 ``False``。因此调用结果可以用于 UI 展示、
    调试或后续人工确认，但不会被 :meth:`ExternalContextBundle.render_for_answer`
    拼入模型上下文。
    """

    def __init__(
        self,
        config: MCPServerConfig | None,
        *,
        tool_name: str = "",
        resource_uri: str = "",
        query_argument: str = "query",
        static_arguments: Mapping[str, Any] | None = None,
        allow_in_answers: bool = False,
        max_chars: int = 8000,
        client_factory: Callable[[MCPServerConfig], MCPClient] = MCPClient,
    ) -> None:
        if tool_name and resource_uri:
            raise ValueError("tool_name 与 resource_uri 只能配置一个")
        if max_chars <= 0:
            raise ValueError("max_chars 必须大于 0")
        self.config = config
        self.tool_name = tool_name.strip()
        self.resource_uri = resource_uri.strip()
        self.query_argument = query_argument.strip()
        self.static_arguments = dict(static_arguments or {})
        self.allow_in_answers = allow_in_answers
        self.max_chars = max_chars
        self._client_factory = client_factory

    async def provide(self, query: str) -> ExternalContextBundle:
        """获取外部上下文；未配置/连接失败时返回空 bundle。"""

        if self.config is None:
            return ExternalContextBundle(available=False, error="未配置 MCP 服务")
        if not self.tool_name and not self.resource_uri:
            return ExternalContextBundle(available=False, error="未配置 MCP tool/resource")

        try:
            async with self._client_factory(self.config) as client:
                if self.tool_name:
                    arguments = dict(self.static_arguments)
                    if self.query_argument:
                        arguments[self.query_argument] = query
                    result = await client.call_tool(self.tool_name, arguments)
                    if result.is_error:
                        return ExternalContextBundle(
                            available=False,
                            error=result.text or f"MCP tool 调用失败: {self.tool_name}",
                        )
                    text = self._tool_text(result)
                    source = f"mcp://{self.config.name}/tools/{self.tool_name}"
                    metadata = {"kind": "tool", "is_error": result.is_error}
                else:
                    contents = await client.read_resource(self.resource_uri)
                    text = self._content_text(contents)
                    source = self.resource_uri
                    metadata = {"kind": "resource", "server": self.config.name}
        except Exception as exc:
            return ExternalContextBundle(
                available=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        if not text:
            return ExternalContextBundle()
        item = ExternalContextItem(
            content=text[: self.max_chars],
            source=source,
            metadata=metadata,
            allow_answer_use=self.allow_in_answers,
        )
        return ExternalContextBundle((item,))

    @staticmethod
    def _content_text(contents: tuple[MCPContent, ...]) -> str:
        blocks: list[str] = []
        for content in contents:
            if content.text:
                blocks.append(content.text)
            elif content.data is not None:
                blocks.append(json.dumps(content.data, ensure_ascii=False, default=str))
        return "\n".join(blocks)

    @classmethod
    def _tool_text(cls, result: MCPCallResult) -> str:
        text = cls._content_text(result.content)
        if result.structured_content is not None:
            structured = json.dumps(
                result.structured_content,
                ensure_ascii=False,
                default=str,
            )
            return f"{text}\n{structured}".strip()
        return text
