# RefMind 文献知识库助手

RefMind 是一个面向科研文献阅读的 RAG 问答系统，基于 LangChain 1.0 与 LangGraph 搭建，
分为离线知识















- 离线：PDF 解析清洗 → 按语义切分 Chunk → 标注来源/章节/页码/版本/权限等 metadata
  → 建立向量索引与 BM25 关键词索引。
- 在线：Query → 向量 + 关键词混合召回一批候选 → reranker 精排 → 上下文压缩（去重、
  句级过滤、字数预算）→ LLM 基于压缩后的上下文生成可溯源答案。

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

## 技术栈

- Python 3.11+
- LangChain 1.0 + LangGraph（RAG 流程与状态编排）
- Chroma（向量检索）、rank-bm25 + jieba（关键词检索与中文分词）
- Streamlit（前端）
- DashScope（OpenAI 兼容接口的对话/嵌入模型）
- SQLite（组、文档、会话、消息等元数据）

## 目录结构

```
app.py                     Streamlit 前端
refmind/
    config/                配置（.env 驱动，前端设置页可改）
    storage/               SQLite 持久化
    parsing/               PDF 解析（MinerU + PyMuPDF 回退）
    llm/                   模型工厂 + 熔断降级、翻译、摘要
    rag/                   分块入库、混合召回、重排、上下文压缩、记忆、LangGraph 流程
    services/              上传入库等业务编排
evaluation/                RAG 评测（检索/生成指标）
scripts/                   冒烟测试、模型探测等脚本
```

## 快速开始

```bash
git clone https://github.com/LiuYj9/RefMind.git
cd RefMind

python -m venv .venv
.venv\Scripts\activate            # Windows

pip install -r requirements.txt

copy .env.example .env            # 填入 DASHSCOPE_API_KEY

streamlit run app.py              # 默认 http://localhost:8888
```

需要高精度解析时再额外安装 MinerU：`pip install mineru`。

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
- 熔断降级：连续失败达到阈值后走备选模型，冷却后半开探测，成功再切回，对上层透明。
- 动态配置：设置页改完参数写回 `.env`，并清掉模型/检索器缓存即时生效。

> retrieve 节点一步完成“混合召回 → 重排 → 上下文压缩”，因此评测/线上走的是同一条链路。

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

其余配置见 `.env.example`。

## 说明

- 对话与嵌入都依赖有效的 `DASHSCOPE_API_KEY`。
- BM25 是内存索引，文档增删后会自动重建。
- 数据默认放在 `./data/`（已在 `.gitignore` 忽略）。

## 界面预览

| 对话问答 | 设置页 |
|----------|--------|
| ![主界面](./pictures/主界面.png) | ![设置界面](./pictures/设置界面.png) |
