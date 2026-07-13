# RefMind 文献知识库助手

RefMind 是一个面向科研文献阅读的 RAG 问答系统，基于 LangChain 1.0 与 LangGraph 搭建，
将离线入库与在线问答分开：
- 离线：PDF 解析清洗 → 按语义切分 Chunk → 标注来源/章节/页码/版本/权限等 metadata
  → 建立向量索引与 BM25 关键词索引。
- 在线：Query → 规划少量检索角度 → 并行混合召回 → 全局 reranker 精排 → 上下文压缩
  （去重、句级过滤、字数预算）→ LLM 生成并基于证据审校可溯源答案。

目标是在读论文时能快速问答，同时尽量避免大模型脱离原文乱答。

## 功能

- 混合召回：BM25 关键词 + Chroma 向量，等权融合，中文用 jieba 分词
- 重排精排：召回候选交给 rerank 模型（DashScope gte-rerank，缺失时回退嵌入相似度）
- 上下文压缩：去除重复分块、剔除离题句子、按字数预算截断，压缩进 Prompt 的内容
- 分块 metadata：来源、文档 id、页码、章节、版本、权限（按库隔离）、分块序号、字数
- PDF 解析：优先 MinerU（公式、表格、排版），未安装或失败时回退 PyMuPDF
- 多文献库隔离：每个库独立的 Chroma 集合与持久化目录，互不干扰
- 长对话记忆：按语义相似度筛选相关历史，而不是无脑保留最近若干轮
- 翻译与摘要：流式输出，翻译可结合当前文献库做术语对齐，入库时自动生成摘要
- 熔断降级：主对话模型连续失败后临时切到备选模型，冷却后再探测切回
- 受控 multi-agent：规划、并行检索、可选证据审查与答案审校职责分离，失败自动回退基线
- 插件扩展：解析、入库、检索和生成阶段提供类型化 hook，第三方异常不会打断主流程
- 可选 MCP：支持 stdio / Streamable HTTP 的能力探测、工具调用与资源读取，默认不把外部内容送入答案
- 可恢复入库：原子文件写入 + 异常补偿 + 启动恢复，未提交向量不会长期混入检索

## 技术栈

- Python 3.11+
- LangChain 1.0 + LangGraph（RAG 流程与状态编排）
- Chroma（向量检索）、rank-bm25 + jieba（关键词检索与中文分词）
- Streamlit（前端）
- DashScope（OpenAI 兼容接口的对话/嵌入模型）
- MCP Python SDK（可选，用于连接显式配置的外部工具与资源）
- SQLite（组、文档、会话、消息等元数据）

## 目录结构

```
app.py                     Streamlit 前端
refmind/
    config/                配置（.env 驱动，前端设置页可改）
    storage/               SQLite 持久化
    parsing/               PDF 解析（MinerU + PyMuPDF 回退）
    llm/                   模型工厂 + 熔断降级、翻译、摘要
    agents/                规划、并行检索、证据审查与答案审校
    plugins/               hook 协议、插件注册与发现
    integrations/          可选 MCP 客户端与外部上下文信任边界
    rag/                   分块入库、混合召回、重排、上下文压缩、记忆、LangGraph 流程
    services/              上传入库等业务编排
evaluation/                RAG 评测（检索/生成指标）
scripts/                   冒烟测试、模型探测等脚本
tests/                     不依赖真实 API 的单元与集成回归测试
```

## 快速开始

```bash
git clone https://github.com/LiuYj9/RefMind.git
cd RefMind

python -m venv .venv
.venv\Scripts\activate            # Windows

.venv\Scripts\python.exe -m pip install -r requirements.txt

# 只有需要连接 MCP 服务时才安装
.venv\Scripts\python.exe -m pip install -r requirements-mcp.txt

copy .env.example .env            # 填入 DASHSCOPE_API_KEY

.venv\Scripts\python.exe -m streamlit run app.py  # 默认 http://localhost:8888
```

> Windows 上请勿直接运行裸 `streamlit`：它可能来自另一套 Python。始终用项目
> `.venv\Scripts\python.exe -m streamlit`，确保 DashScope、LangChain 与前端版本一致。

若出现 `No module named 'dashscope'` 或“调用报错”，先执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app.py
```

需要高精度解析时再额外安装 MinerU：`pip install mineru`。

详细架构见 [docs/architecture.md](./docs/architecture.md)，插件与 MCP 配置见
[docs/plugins-and-mcp.md](./docs/plugins-and-mcp.md)。

## 主要设计

- 分块 metadata：每个 Chunk 带来源、文档 id、页码、章节、版本、权限（按库隔离）、
  分块序号与字数，既支持答案溯源，也为后续按条件过滤/治理留出空间。
- 混合召回：向量负责语义泛化，BM25 负责专业术语精确匹配，两者等权融合出候选池。
- 重排精排：候选交给 reranker 按 (query, chunk) 相关性重新打分排序，优先 DashScope
  rerank 模型，未安装或失败时回退到嵌入余弦相似度，只保留最相关的前若干条。
- 上下文压缩：精排后再去掉近似重复分块、按句子粒度剔除离题内容、按字数预算截断，
  在保留关键证据的同时降低冗余与 token 消耗；嵌入不可用时退化为仅按字数截断。
- 记忆过滤：用余弦相似度筛掉与当前问题无关的历史消息，控制 token 成本。
- 无检索即拒答：LangGraph 的 retrieve→generate 流程里，检索为空时直接返回“未找到相关内容”，减少幻觉。
- 解析容错：MinerU 失败自动回退 PyMuPDF，保证基本可用。
- 可恢复写入/删除：源 PDF 使用 `doc_id` 唯一路径，解析 JSON 原子替换；异常时按
  `doc_id` 补偿，进程强杀留下的非 `ready` 记录会在下次启动继续清理。
- 受控 multi-agent：规划最多三个子查询，并发召回后统一重排/压缩；规划、单路检索、审校
  任一失败都保留上一阶段有效结果或回退单查询，不让增强能力成为单点故障。
- 插件隔离：hook 按注册顺序变换数据，插件注册和运行异常会被记录并隔离；核心阶段还会校验
  返回类型，避免错误插件污染主链路。
- MCP 信任边界：只连接 `.env` 显式声明的服务；外部结果默认只供探测/人工检查，必须由代码
  明确设置 `allow_in_answers=True` 才能进入答案上下文。
- 熔断降级：连续失败达到阈值后走备选模型，冷却后只允许一个 half-open 探测；流式输出中途
  失败不会与备选模型输出拼接。
- 动态配置：设置页改完参数写回 `.env`，并清掉模型/检索器缓存即时生效。

> multi-agent 只扩展检索角度，合并后仍复用同一套“重排 → 压缩 → 生成”生产链路；关闭后直接
> 回到原 LangGraph 基线，便于做同口径评测。

## 常用配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DASHSCOPE_API_KEY` | DashScope API Key | 必填 |
| `LLM_MODEL` | 对话模型 | `qwen3.7-plus` |
| `LLM_FALLBACK_MODEL` | 备选模型（留空则不降级） | 空 |
| `LLM_CIRCUIT_FAILURE_THRESHOLD` | 连续失败多少次熔断 | `3` |
| `LLM_HEALTH_CHECK_INTERVAL` | 熔断冷却秒数 | `60` |
| `EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-v4` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 分块大小 / 重叠 | `1000` / `200` |
| `RETRIEVAL_TOP_K` | 检索返回片段数 | `5` |
| `RECALL_TOP_K` | 重排前的召回候选数 | `20` |
| `RERANK_ENABLED` | 是否启用重排 | `true` |
| `RERANK_MODEL` | 重排模型（DashScope） | `gte-rerank-v2` |
| `RERANK_TOP_N` | 重排后保留片段数 | `5` |
| `CONTEXT_COMPRESSION_ENABLED` | 是否启用上下文压缩 | `true` |
| `CONTEXT_MAX_CHARS` | 送入 Prompt 的上下文字数上限 | `4000` |
| `MEMORY_MAX_TURNS` | 记忆保留轮数 | `30` |
| `MEMORY_RELEVANCE_THRESHOLD` | 记忆相关性阈值 | `0.3` |
| `MULTI_AGENT_ENABLED` | 启用受控 multi-agent | `true` |
| `MULTI_AGENT_MAX_SUBQUERIES` | 最大检索子查询数（1~3） | `3` |
| `MULTI_AGENT_MAX_WORKERS` | 并行检索线程数 | `3` |
| `MULTI_AGENT_RETRIEVAL_TIMEOUT` | 并行检索等待上限（秒） | `30` |
| `MULTI_AGENT_EVIDENCE_REVIEW` | 额外 LLM 证据审查 | `false` |
| `MULTI_AGENT_ANSWER_REVIEW` | 基于证据审校草稿 | `true` |
| `REFMIND_PLUGIN_MODULES` | 逗号分隔的插件模块 | 空 |
| `REFMIND_MCP_SERVERS` | MCP 服务 JSON 数组 | 空 |

其余配置见 `.env.example`。

## 说明

- 对话与嵌入都依赖有效的 `DASHSCOPE_API_KEY`。
- BM25 是内存索引，文档增删后会自动重建。
- 数据默认放在 `./data/`（已在 `.gitignore` 忽略）。

## 测试与探测

```bash
# 全部离线回归测试（不调用真实模型或 MCP 服务）
python -m unittest discover -s tests -v

# 探测 .env 中声明的 MCP 服务能力
python scripts/probe_mcp.py

# 调用已确认安全的 MCP 工具
python scripts/probe_mcp.py --server papers --tool search --arguments '{"query":"RAG"}'
```

端到端模型与向量库烟测仍使用 `python scripts/smoke_test.py`；它需要有效 API Key 和
`data/sample_paper.pdf`，不属于离线测试。

## 界面预览

| 对话问答 | 设置页 |
|----------|--------|
| ![主界面](./pictures/主界面.png) | ![设置界面](./pictures/设置界面.png) |
