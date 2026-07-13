"""探测已配置的 MCP 服务，或显式调用一个工具。

示例：
    python scripts/probe_mcp.py
    python scripts/probe_mcp.py --server papers --tool search \
        --arguments '{"query": "retrieval augmented generation"}'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

# 允许从项目根目录直接运行脚本。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

# Windows 默认控制台可能仍是 GBK，统一 UTF-8 便于输出协议元数据和中文错误。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from refmind.integrations import MCPManager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="探测 RefMind 的可选 MCP 服务")
    parser.add_argument("--server", help="服务名；不指定时探测全部服务")
    parser.add_argument("--tool", help="要调用的 tool 名称（必须同时指定 --server）")
    parser.add_argument(
        "--arguments",
        default="{}",
        help="tool 参数 JSON，默认 {}",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    manager = MCPManager.from_environment()
    for error in manager.configuration_errors:
        print(f"配置错误：{error}", file=sys.stderr)

    if args.tool:
        if not args.server:
            print("调用 tool 时必须指定 --server。", file=sys.stderr)
            return 2
        try:
            arguments = json.loads(args.arguments)
        except json.JSONDecodeError as exc:
            print(f"--arguments 不是合法 JSON：{exc}", file=sys.stderr)
            return 2
        if not isinstance(arguments, dict):
            print("--arguments 顶层必须是 JSON 对象。", file=sys.stderr)
            return 2
        try:
            result = await manager.call_tool(args.server, args.tool, arguments)
        except Exception as exc:  # 探测脚本需要输出可操作错误，而不是长堆栈
            print(f"MCP tool 调用失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 1 if result.is_error else 0

    names = (args.server,) if args.server else manager.servers
    if not names:
        print(
            "未配置 MCP 服务。请先在 .env 设置 REFMIND_MCP_SERVERS；核心 RAG 不受影响。"
        )
        return 0

    probes = [await manager.probe(name) for name in names]
    print(
        json.dumps(
            [asdict(probe) for probe in probes],
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0 if all(probe.connected for probe in probes) else 1


def main() -> int:
    return asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
