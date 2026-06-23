# RefMind · 智能文献知识库助手

<p align="center">
  <img src="https://img.shields.io/badge/RefMind-文献知识库助手-blue?style=for-the-badge&logo=readthedocs" />
  <br/>
  <sub>基于 LangGraph + LangChain 1.0 的端到端 RAG 文献问答系统</sub>
</p>

---

RefMind 专为科研文献阅读场景设计，通过 **高精度 PDF 解析 → 混合检索 RAG → LLM 智能问答** 的完整链路，解决文献信息过载与大模型"幻觉"问题。

---

## ✨ 核心功能

| 功能 | 说明 |
|------|------|
| 🧠 **混合检索 RAG** | BM25 关键词 + Chroma 向量检索（权重 0.5:0.5），jieba 中文分词 |
| 📄 **高精度 PDF 解析** | MinerU 结构化解析（公式/表格/排版），自动回退 PyMuPDF |
| 🔒 **多用户组隔离** | 独立 Chroma 集合 + 持久化目录，组间数据完全隔离 |
| 💬 **智能长对话记忆** | 最近 30 轮 + 语义相似度过滤，仅保留相关上下文 |
| 🌐 **文档翻译与摘要** | 流式输出，翻译可结合文献库术语对齐，摘要约 200 字 |
| 📂 **会话管理** | 同组多会话共享知识库、独立对话历史 |
| 🛡️ **LLM 熔断降级** | 三态熔断器（CLOSED/OPEN/HALF-OPEN），主模型故障自动切换备选 |

---

## �️ 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.11+ | 核心语言 |
| LangChain 1.0 + LangGraph 1.0 | RAG 流程编排与状态管理 |
| Chroma | 向量存储与检索 |
| BM25 + jieba | 关键词检索 + 中文分词 |
| Streamlit | Web 前端 |
| DashScope | 大模型服务（OpenAI 兼容接口） |
| SQLite | 元数据持久化 |

---

## 🏗️ 架构概览

```
app.py（Streamlit 前端）
    │
    ▼
services/（业务编排层）
    │
    ├── rag/（核心层）
    │   ├── retrieval   ← BM25 + 向量混合检索（EnsembleRetriever）
    │   ├── memory      ← 语义相关性过滤长对话记忆
    │   ├── processor   ← 分块 · 向量化 · Chroma 入库
    │   └── graph       ← LangGraph 状态图（retrieve → generate）
    │
    ├── llm/（模型层）
    │   ├── factory     ← 对话/嵌入模型工厂 + 熔断降级代理
    │   ├── translation ← 上下文增强翻译
    │   └── summarization ← 自动摘要
    │
    ├── parsing/（解析层）
    │   └── pdf_parser  ← MinerU（高精度）+ PyMuPDF（回退）
    │
    ├── storage/（持久层）
    │   └── SQLite CRUD（组 · 文档 · 会话 · 消息）
    │
    └── config/（配置层）
        └── .env 驱动，前端设置页可动态修改
```

---

## 🚀 快速开始

```bash
# 1. 克隆
git clone https://github.com/LiuYj9/RefMind.git && cd RefMind

# 2. 虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows

# 3. 依赖
pip install -r requirements.txt

# 4. 配置
copy .env.example .env        # 编辑 .env 填入 DASHSCOPE_API_KEY

# 5. 启动
streamlit run app.py           # http://localhost:8888

# 6. 可选：高精度解析
pip install mineru
```

---

## 🌟 关键技术点

**混合检索** — 向量检索解决语义泛化，BM25 解决专业术语精确匹配，0.5:0.5 权重融合。

**语义过滤记忆** — 不是简单保留最近 N 轮，而是用余弦相似度筛选与当前问题相关的历史消息，降低 token 成本。

**无检索拒答** — LangGraph `retrieve→generate` 流程中，零文档时直接返回"未找到相关内容"，杜绝幻觉。

**PDF 解析容错** — MinerU 优先（公式/表格高精度）→ 失败自动回退 PyMuPDF，兼顾质量与可用性。

**熔断降级** — 三态熔断器：连续失败 3 次熔断走备选 → 冷却 60s → 半开探测 → 成功切回主模型，全程对上层透明。

**多文献库隔离** — SQLite 元数据 + Chroma 向量库均按组隔离，级联删除。

**动态配置** — 前端设置页修改参数后即时写回 `.env`，自动清空模型/检索器缓存生效。

---

## ⚙️ 配置项

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key | *必填* |
| `LLM_MODEL` | 对话模型 | `qwen3.7-plus` |
| `LLM_FALLBACK_MODEL` | 备选模型（留空禁用降级） | `qwen-turbo` |
| `LLM_CIRCUIT_FAILURE_THRESHOLD` | 连续失败 N 次熔断 | `3` |
| `LLM_HEALTH_CHECK_INTERVAL` | 熔断冷却时间（秒） | `60` |
| `EMBEDDING_MODEL` | 嵌入模型 | `text-embedding-v4` |
| `CHUNK_SIZE` | 分块大小 | `1000` |
| `CHUNK_OVERLAP` | 分块重叠 | `200` |
| `RETRIEVAL_TOP_K` | 检索返回片段数 | `5` |
| `MEMORY_MAX_TURNS` | 记忆保留轮数 | `30` |
| `MEMORY_RELEVANCE_THRESHOLD` | 记忆相关性阈值 | `0.3` |

> 更多配置见 `.env.example`

---

## 📝 注意事项

- 嵌入与对话模型需有效 `DASHSCOPE_API_KEY`
- BM25 索引为内存索引，文档变更后自动重建
- 数据默认存储在 `./data/`（已在 `.gitignore` 中忽略）

---

## 🖥️ 界面预览

| 对话问答 | 设置页 |
|----------|--------|
| ![主界面](./pictures/主界面.png) | ![设置界面](./pictures/设置界面.png) |