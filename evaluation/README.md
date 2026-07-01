# RefMind RAG 评测框架

围绕**人工标注的 golden set**，从检索与生成两个维度量化 RefMind 的 RAG 效果。

- **检索侧**：Recall@K、MRR、nDCG@K —— 衡量"是否召回了正确证据、排得是否靠前"。
- **生成侧**（LLM-as-judge）：Faithfulness、Answer Relevance、Context Precision、Context Recall
  —— 衡量"答案是否忠于上下文、是否切题、上下文是否精准且完整"。

---

## 1. 评测架构

```
                        golden_set.json
             （问题 / 标准答案 / 证据页[filename,page]）
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  run_eval.py（CLI 入口）                       │
        │   · 自动生成内置合成语料（首次）               │
        │   · 解析参数、落盘 JSON + Markdown 报告        │
        └───────────────────────┬─────────────────────┘
                                │
                                ▼
        ┌─────────────────────────────────────────────┐
        │  evaluator.py（编排）                          │
        │   1) 建临时文献库 → ingest_pdf 入库 corpus     │
        │   2) 构建两类检索器：                          │
        │        · eval 检索器 (k=max_K) → 检索指标      │
        │        · 生产检索器 (k=top_k)  → 问答          │
        │   3) 逐题：检索 → answer_question → 打分        │
        │   4) 汇总均值 / 延迟分位数 / 拒答正确率         │
        └───────────┬───────────────────┬─────────────┘
                    │                   │
                    ▼                   ▼
        metrics/retrieval.py    metrics/generation.py
        Recall@K/MRR/nDCG@K     Faithfulness/AnsRel/
        （纯规则，可离线）        CtxPrec/CtxRecall（LLM 裁判）
                    │                   │
                    ▼                   ▼
              evaluation/reports/report_<时间戳>.{json,md}
```

复用被测系统本身的组件（`refmind.services.ingest_pdf`、`refmind.rag.answer_question`、
`refmind.rag.build_retriever`），确保**评测路径与线上路径一致**，量出来的就是真实系统效果。

---

## 2. 目录结构

```
evaluation/
├── run_eval.py              # CLI 入口：跑评测、生成报告
├── evaluator.py            # 编排：入库→检索→问答→打分→汇总
├── golden_set.json         # 人工标注：问题/标准答案/证据页
├── make_sample_corpus.py   # 生成"内容+页码已知"的合成论文，便于跑通流程
├── metrics/
│   ├── retrieval.py        # Recall@K、MRR、nDCG@K（规则计算）
│   └── generation.py       # 四项生成指标（LLM-as-judge）
├── corpus/                 # 语料 PDF（合成语料自动生成于此）
└── reports/                # 评测报告输出（JSON + Markdown）
```

---

## 3. golden set 标注规范

```jsonc
{
  "corpus": ["evaluation/corpus/RefMindNet.pdf"],  // 参与评测的 PDF（相对项目根）
  "page_tolerance": 0,                              // 证据页匹配容差（0=精确）
  "items": [
    {
      "id": "q1",
      "question": "……",                            // 问题
      "ground_truth": "……",                        // 标准答案（人工撰写）
      "evidence": [                                 // 证据页：答案出处
        {"filename": "RefMindNet.pdf", "page": 1}
      ]
    },
    {
      "id": "q6",
      "question": "与知识库无关的问题",
      "ground_truth": "无法回答……",
      "evidence": [],
      "expect_refusal": true                        // 期望系统拒答（考核幻觉抑制）
    }
  ]
}
```

**标注要点**
- 证据页 `page` 与解析后分块 metadata 的 `page` 对齐（PyMuPDF/MinerU 均按页 1 起）。
- 每题证据页尽量唯一、明确，指标才有区分度。
- 建议加入若干 `expect_refusal` 的越界问题，专门考核"无检索则拒答"策略。
- 规模建议：起步 30~50 题，覆盖事实型、跨页综合型、越界型三类。

---

## 4. 指标定义

### 4.1 检索侧（`metrics/retrieval.py`，纯规则、可离线）

某检索结果的 `(filename, page)` 命中证据集合（允许 `page_tolerance`）即视为"相关"。

| 指标 | 定义 | 侧重 |
|---|---|---|
| **Recall@K** | 前 K 个结果覆盖的证据页 / 全部证据页（按页去重） | 是否召回全 |
| **MRR** | 首个相关结果排名的倒数 `1/rank` | 相关项是否靠前 |
| **nDCG@K** | 二值相关性下的归一化折损累积增益 | 整体排序质量 |

### 4.2 生成侧（`metrics/generation.py`，LLM-as-judge）

用项目自带对话模型作裁判，每指标一条聚焦提示词，仅返回 JSON。

| 指标 | 定义 | 回答什么问题 |
|---|---|---|
| **Faithfulness** | 答案中被上下文支撑的陈述 / 全部陈述 | 有没有幻觉 |
| **Answer Relevance** | 答案对问题的切题程度（0~1） | 有没有跑题 |
| **Context Precision** | 检索上下文中"有用片段" / 全部片段 | 上下文信噪比 |
| **Context Recall** | 标准答案信息点中可被上下文支撑的比例 | 上下文是否完整 |

> 说明：这里的生成指标是可运行的工程化近似（单次评分式裁判），
> 便于快速迭代；如需更严格，可换成 RAGAS 官方库做交叉验证。

### 4.3 附加

- **拒答正确率**：`expect_refusal` 的题中，系统确实拒答的比例（衡量幻觉抑制）。
- **延迟**：端到端 `answer_question` 的 mean / P50 / P95 / P99。

---

## 5. 运行方式

> 需先激活装好依赖的环境（本项目为 conda 环境 `langchain`），
> 生成侧指标需在 `.env` 配置 `DASHSCOPE_API_KEY`。

```bash
# 完整评测（首次自动生成合成语料；含 LLM 裁判）
python evaluation/run_eval.py

# 只评测检索侧（不消耗生成 token，无需强裁判）
python evaluation/run_eval.py --no-judge

# 自定义 golden set 与 K
python evaluation/run_eval.py --golden path/to/golden.json --k 1 3 5 10 --top-k 5
```

报告输出到 `evaluation/reports/report_<时间戳>.{json,md}`。

---

## 6. 结果如何解读

- **检索强、生成弱**（Recall/nDCG 高，但 Faithfulness/Answer Relevance 低）：
  问题多在提示词或生成模型，考虑调 prompt、换更强对话模型。
- **检索弱**（Recall@5 偏低）：调 `CHUNK_SIZE/OVERLAP`、`top-k`、BM25/向量融合权重，或换嵌入模型。
- **Context Precision 低**：上下文噪声大，可减小 top-k 或加重排序（如 cross-encoder）。
- **Context Recall 低但 Recall@K 高**：分块切碎了关键信息，考虑增大 chunk 或调整分隔符。
- **拒答正确率低**：无检索拒答策略未生效，检查 `graph.py` 的兜底逻辑与系统提示词。

> ⚠️ 简历/汇报中引用的数字，请以本框架在**固定 golden set**上实测的报告为准，
> 并注明评测集规模、K 值、模型版本与评测时间。
