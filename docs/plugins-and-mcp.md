# 插件与 MCP 使用指南

## 1. 启用一个本地插件

插件是普通 Python 模块，通过 `register(registrar)` 声明 hook。仓库自带一个只给检索
metadata 加标签的无副作用示例：

```dotenv
REFMIND_PLUGIN_MODULES=examples.plugins.metadata_tag
```

也可用逗号启用多个模块。修改 `.env` 后重启应用，因为插件发现只在进程内首次获取管理器
时执行。外部 Python 包可以声明 `refmind.plugins` entry point，无需写入此变量。

最小插件：

```python
from langchain_core.documents import Document
from refmind.plugins import CoreHook

name = "my-plugin"

def after_retrieve(event):
    return [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "reviewed": True},
        )
        for doc in event.value
    ]

def register(registrar):
    registrar.add_hook(CoreHook.AFTER_RETRIEVE, after_retrieve)
```

### Core hooks

| Hook | `event.value` | 常用 metadata |
|---|---|---|
| `before_parse` | `Path` | `group_id`, `doc_id`, `filename` |
| `after_parse` | 归一化解析 `dict` | 同上 |
| `before_ingest` | `list[Document]` | 同上 |
| `after_ingest` | `DocumentRow` | 同上 |
| `before_retrieve` | 查询字符串 | `group_id`, `original_question` |
| `after_retrieve` | `list[Document]` | `group_id`, `query` |
| `before_generate` | 最终证据 `list[Document]` | `question`, `history_size` |
| `after_generate` | 答案字符串 | `question`, `document_count` |
| `collect_context` | 由调用方约定 | 用于自定义外部上下文流程 |

同步主流程应注册同步 callback。插件回调抛出的异常会写入 `PluginManager.failures` 并被隔离，
同步 callback 即使在并行检索中也会被管理器串行调用；异步 hook 若被并发调用，需要插件
自行保护内部可变状态。插件仍与 RefMind 进程同权限，请勿加载不可信模块。

## 2. 配置 MCP 服务

先安装可选依赖：

```bash
pip install -r requirements-mcp.txt
```

stdio 服务示例：

```dotenv
REFMIND_MCP_SERVERS=[{"name":"papers","transport":"stdio","command":"python","args":["path/to/server.py"]}]
```

Streamable HTTP 示例：

```dotenv
REFMIND_MCP_SERVERS=[{"name":"research","transport":"streamable_http","url":"https://example.com/mcp","headers":{"Authorization":"Bearer REPLACE_ME"}}]
```

`.env` 含凭证且已被 Git 忽略，不要把真实 token 写进 `.env.example`、源码或日志。

## 3. 先探测，再调用

```bash
# 列出所有服务的协议版本、tools、resources、prompts 与单项错误
python scripts/probe_mcp.py

# 只探测一个服务
python scripts/probe_mcp.py --server papers

# 确认 schema 与来源可信后再调用工具
python scripts/probe_mcp.py --server papers --tool search \
  --arguments '{"query":"retrieval augmented generation"}'
```

没有配置、SDK 未安装或服务连接失败时，探测会给出结构化错误，应用的 PDF 入库和 RAG 问答
仍可正常运行。

Streamlit 设置页也提供“探测 MCP 能力”按钮；它只读取 capability 与列表，不会调用工具或把
外部内容加入答案。

## 4. 在代码中使用

能力探测与工具调用：

```python
import asyncio
from refmind.integrations import MCPManager

async def main():
    manager = MCPManager.from_environment()
    probes = await manager.probe_all()
    result = await manager.call_tool("papers", "search", {"query": "RAG"})
    print(probes, result.text)

asyncio.run(main())
```

外部上下文适配器：

```python
from refmind.integrations import MCPContextProvider, MCPManager

manager = MCPManager.from_environment()
config = manager.get_config("papers")
provider = MCPContextProvider(config, tool_name="search")
bundle = await provider.provide("RAG")

assert bundle.render_for_answer() == ""  # 默认信任边界
```

只有在业务层确认服务身份、数据权限、引用格式和内容质量之后，才考虑构造
`MCPContextProvider(..., allow_in_answers=True)`。即便放行，也应把 MCP 来源与上传文献来源
分开展示，并为这条路径新增评测样本。
