# RefMind · 文献知识库助手

基于大语言模型的智能文献知识库助手。支持高精度 PDF 解析、混合检索 RAG 问答、
多用户组独立知识库、长对话记忆、文档翻译与自动摘要。

## 功能特性

- **PDF 解析**：优先调用 [MinerU](https://github.com/opendatalab/MinerU)，
  未安装时自动回退到 PyMuPDF 文本抽取。
- **混合检索 RAG**：BM25 关键词检索 + Chroma 向量检索，权重各 0.5
  （`EnsembleRetriever`），支持中文 jieba 分词。
- **多用户组隔离**：每个组独立的 Chroma 集合与持久化目录。
- **长对话记忆**：保留最近 30 轮，按语义相似度过滤无关历史。
- **文档翻译 & 自动摘要**：基于 `qwen3.7-plus`，支持流式输出。
- **会话管理**：同一组可创建多个会话，共享知识库。

## 技术栈

Python 3.11+ · **LangChain 1.0** · **LangGraph 1.0** · Chroma · BM25 · Streamlit · DashScope（OpenAI 兼容）

> 注：本项目基于 LangChain 1.0 架构。`EnsembleRetriever` 等遗留组件在 1.0 中已迁移至
> `langchain-classic`，因此依赖中包含 `langchain-classic`。

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
# 编辑 .env 填入 DASHSCOPE_API_KEY

# 4. 启动
streamlit run app.py
```

### 安装 MinerU（可选，用于高精度解析）

未安装 MinerU 时项目使用 PyMuPDF 回退解析。安装 MinerU 后将 `.env` 中
`MINERU_BINARY_PATH` 设为对应命令（`mineru` 或旧版 `magic-pdf`）：

```bash
pip install mineru        # 参见官方仓库获取完整安装说明
```

## 项目结构

按职责将同类模块组织到独立子包中，结构清晰：

```
RefMind/
├── app.py                          # Streamlit 入口
├── requirements.txt
├── .env.example
└── refmind/
    ├── config/                     # 配置
    │   └── settings.py             # 环境变量 / 路径 / 全局设置
    ├── storage/                    # 数据持久化（SQLite）
    │   ├── models.py               # 数据行模型 + 建表 SQL
    │   ├── connection.py           # 连接管理 + 初始化
    │   └── repository.py           # 组/文档/会话/消息 CRUD
    ├── parsing/                    # PDF 解析
    │   └── pdf_parser.py           # MinerU + PyMuPDF 回退
    ├── llm/                        # 大模型相关
    │   ├── factory.py              # 对话/嵌入模型工厂
    │   ├── translation.py          # 翻译
    │   └── summarization.py        # 自动摘要
    ├── rag/                        # 检索增强生成
    │   ├── document_processor.py   # 分块 / 向量化 / Chroma 入库
    │   ├── retrieval.py            # 混合检索（BM25 + 向量）
    │   ├── memory.py               # 相关性过滤的长对话记忆
    │   └── graph.py                # LangGraph 状态图（检索→生成）
    └── services/                   # 高层业务编排
        └── ingestion.py            # 上传/入库流程
```

## 配置项（.env）

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key |
| `API_BASE` | OpenAI 兼容接口地址 |
| `LLM_MODEL` / `EMBEDDING_MODEL` | 对话 / 嵌入模型名 |
| `MINERU_BINARY_PATH` | MinerU 命令（`mineru` / `magic-pdf`） |
| `USE_FALLBACK_PARSER` | 强制使用 PyMuPDF 回退解析 |
| `RETRIEVAL_TOP_K` | 每个检索器返回片段数 |
| `MEMORY_MAX_TURNS` | 记忆保留轮数 |
| `MEMORY_RELEVANCE_THRESHOLD` | 历史消息相关性阈值 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 分块大小 / 重叠 |

## 说明

- 嵌入与对话模型通过 DashScope OpenAI 兼容接口调用，需要有效 API Key。
- BM25 索引为内存索引，文档增删后会自动重建。
- 数据默认存储在 `./data/` 下（已在 `.gitignore` 中忽略）。
