# RefMind 架构与降级策略

## 1. 设计目标

RefMind 的核心约束是“答案必须能回到用户上传的文献证据”。multi-agent、插件和 MCP
都是可选增强层，不能改变以下基线：无文献证据时拒答；增强失败时保留上一阶段的有效结果；
外部 MCP 内容默认不参与答案生成。

## 2. 入库流程

```text
批量上传 PDF
  → 按 PDF 建立独立任务与临时文件
  → 最多 32 篇文档进入阶段流水线
  → 每篇在同目录暂存并原子替换
  → 解析闸门：before_parse → MinerU（失败则 PyMuPDF）→ after_parse
      并发上限 `PDF_MAX_PARALLEL_DOCUMENTS`
  → 图片提取 → 共享线程池并行生成去重后的图片摘要
      全局上限 `IMAGE_SUMMARY_MAX_WORKERS`
  → 归一化 layout blocks → layout-aware 切分 → before_ingest
      不同文档并行，单篇内部保持阅读顺序
  → 并行分支
      ├─ 共享线程池分批 Embedding（`EMBEDDING_MAX_PARALLEL_BATCHES`）
      └─ 生成文档中文摘要（`DOCUMENT_SUMMARY_MAX_WORKERS`）
  → 仅 Chroma upsert 按 group 串行提交
  → SQLite 状态提交为 ready
  → after_ingest 插件 hook
```

PDF、解析 JSON、Chroma 与 SQLite 无法共享单一事务，因此入库服务采用补偿事务：提交
`ready` 前任一步骤失败，都会按 `doc_id` 删除本次可能写入的向量、解析文件、上传文件和
数据库记录。每个源 PDF 的路径包含 `doc_id`，同名重传会先建立新版本，成功后才删除旧版。
解析 JSON 使用同目录临时文件加 `os.replace`，避免进程中断留下半个 JSON。

若进程在 Chroma 写入后被强杀，Python 异常处理来不及执行；应用启动时会扫描所有非
`ready` 记录并按 `doc_id` 继续清理。向量清理失败则保留 `cleanup_failed` 记录与文件，
避免丢掉后续重试所需的索引线索。删除同样使用 `deleting → cleanup_failed/删除记录` 的
可重试状态，而不是吞掉错误后宣称成功。

批量上传采用分阶段流水线，单篇任务仍保留上述完整状态机。外层最多允许 32 篇文档处于
飞行状态，避免前几篇等待图片或远程 API 时让 MinerU 空闲；图片摘要、Embedding 批次和
文档摘要使用三个模块级共享线程池，不会为每篇文档无限创建嵌套线程。切分在不同文档间
并行，单篇内部仍按阅读顺序执行，因为该阶段通常远快于 MinerU 和模型调用，强行拆散会破坏
章节上下文且增加合并开销。

四个并发配置都是进程级上限，多个 Streamlit 会话共享，而不是每个浏览器分别拥有一份。
`PDF_MAX_PARALLEL_DOCUMENTS` 只限制解析槽；传给单个批次的更小值还会通过本地信号量进一步
收紧。远程 Embedding 已移出 Chroma 写锁，同一 `group_id` 仅最终 upsert/delete 串行提交，
不同文献库互不阻塞。SQLite 每个 worker 使用独立连接，并启用 WAL 与 30 秒 busy timeout。
worker 只写入线程安全进度队列，Streamlit 控件始终由主线程更新。

同名旧文档不会在 worker 中立即删除：所有新版本任务结束后，协调线程只对成功提交的任务
串行清理旧版本。单篇解析或 Embedding 失败会回滚该篇，但不会取消批次中的其他任务；单张
图片摘要或文档摘要失败属于可选增强降级，正文仍可提交为 ready。解析默认并行数为 2，其余
三个模型阶段默认并发数为 4；使用单张本地 GPU MinerU、OCR 或内存有限时，解析建议先从
2～3 开始，不宜盲目调高。

设置页保留上一次批量入库的墙钟时间、各文档阶段耗时以及解析/Chroma 等待时间。Embedding
与文档摘要、不同文档之间会发生重叠，因此这些阶段数字用于定位瓶颈，不能相加后与批次墙钟
直接比较。同库删除通过进程内生命周期闸门等待活跃入库收敛，避免删除 Chroma 目录时被并发
任务重新创建；共享目录仍按单应用进程部署设计，多进程写入需增加跨进程锁。

### 2.1 Layout-aware 论文切分

### 2.2 图片摘要索引与多模态回答

`ingestion` 在 MinerU/PyMuPDF 完成版面解析后，依据 figure 的 `bbox` 裁剪页面（没有 bbox 时提取嵌入位图），将原图写入 `DOCSTORE_DIR/doc_<doc_id>/images/`。图片文件不进入 Chroma；入库时由 `qwen3.5-omni-plus-2026-03-15` 生成克制的结构化视觉摘要，摘要、图注、页码及受控的 `image_path` 一起作为 `figure` 文本块写入索引。

检索流程先按摘要召回和重排。仅当最终证据块含有 `image_path` 时，回答层才校验路径仍位于 docstore、检查尺寸上限、Base64 编码原图，并将不超过 `IMAGE_MAX_PER_ANSWER` 张图片附到全模态模型消息中。图片摘要服务不可用不会中断文本入库；此时会保留图注或明确的降级标记。删除或入库回滚会同步清理该 `doc_id` 的 docstore 目录。

解析结果保留旧版 `markdown/pages/tables`，并新增 `schema_version=2` 与 `blocks`。MinerU
提供的阅读顺序优先于 bbox 推断；标题会开启新的章节，连续正文块只在同一章节内合并，
表格、公式和图注保持原子语义单元，超过 `LAYOUT_CHUNK_MAX_CHARS` 才进行二次递归切分。
下游 metadata 包含 `content_type`、`page_start/page_end`、`section_path`、`block_ids`、
`bbox` 和布局置信度。列表字段会序列化为 JSON 字符串，以满足 Chroma 仅接受标量 metadata
的限制。没有 `blocks` 的历史解析文件继续使用逐页切分，不要求迁移已有数据。

PyMuPDF 回退只能提供低置信度的逐页版面块，不能可靠恢复双栏阅读顺序与标题层级；需要高质量
论文解析时应安装 MinerU。结构化路径不会再同时索引 `pages` 和 `tables`，从源头避免表格重复召回。

## 3. 问答流程

```text
问题 + 相关会话历史
  → memory_retrieve：按 user_id + group_id 召回活跃长期记忆
  → PlanningAgent：用研究/偏好语境补全查询，复杂问题拆为 ≤3 个检索查询
  → ParallelRetrievalAgent：有界线程池并发执行 BM25 + Chroma 混合召回
  → 按 chunk_id / 文档位置 / 内容稳定去重
  → 对合并候选统一 rerank + 上下文压缩
  → 可选 EvidenceReviewAgent：只筛证据，不生成答案
  → 原 RAG prompt 生成草稿
  → 可选 AnswerReviewAgent：只依据证据做最小修正
  → memory_extract：只从本轮用户消息提取原子候选
  → memory_update：过滤、合并或失效冲突事实后写入 SQLite
  → 写入会话记忆并返回答案、证据、查询与降级诊断
```

基础 LangGraph 为 `memory_retrieve → retrieve → generate → memory_extract → memory_update`。
关闭 `MULTI_AGENT_ENABLED` 时直接走基础图；启用时 multi-agent 复用相同的记忆节点逻辑。
规划、并发检索、后处理、审校或记忆增强发生异常时会回到单查询/原草稿路径。

### 3.1 三层存储与记忆治理

| 层级 | 内容 | 存储/检索 |
|---|---|---|
| 会话记忆 | 单个 session 的原始消息 | SQLite `messages` + 滑动窗口 |
| 用户长期记忆 | 偏好、研究方向、背景、术语、任务与重要情景 | SQLite `long_term_memories` + embedding |
| 文献知识 | 论文正文、公式、表格与图片摘要 | Chroma + BM25 |

长期记忆检索固定使用 `user_id + group_id + is_active=1`，不会进入论文召回池。生成 Prompt
将其渲染在 `【用户长期记忆】`，论文片段单独渲染在 `【论文检索证据】`；系统规则禁止以
用户记忆支撑论文结论或引用。

候选写入采用保守策略：只分析用户原话；要求原子化、用户中心、达到重要度与置信度阈值；
论文数值、公式、结论、寒暄和一次性指令由提取提示与结构校验共同过滤。精确哈希或高相似
候选会合并并增强权重；相同 `memory_key` 的新事实会插入新版本并将旧版本软失效，通过
`superseded_by` 保留审计链。

权重按最后访问时间指数衰减，默认语义记忆半衰期 180 天、情景记忆 45 天。情景记忆默认
180 天到期；语义/情景记忆分别在 730/180 天未使用且有效权重低于 0.15 时软归档。召回会
增加 `access_count` 并刷新 `last_accessed_at`。这些参数均可由 `.env` 调整。

## 4. 角色边界

| 角色 | 可以做 | 不可以做 |
|---|---|---|
| PlanningAgent | 生成少量互补检索式 | 回答问题、引入题外目标 |
| ParallelRetrievalAgent | 并发召回、稳定去重 | 修改全局索引、无限创建线程 |
| EvidenceReviewAgent | 按原问题筛掉弱相关片段 | 生成新事实 |
| AnswerReviewAgent | 用现有证据修正草稿 | 使用外部知识补写答案 |

编排器只依赖 `invoke`/callable 协议，不依赖某一家模型 SDK；当前继续复用项目既有的
OpenAI-compatible DashScope 模型工厂和熔断器。

并行检索使用有界线程池并设请求级等待上限。超时后停止等待未完成 future 并返回降级状态；
Python 无法强杀已经运行的线程，因此自定义检索器和插件仍必须为底层网络请求配置超时。

## 5. 插件边界

插件管理器支持显式注册、`REFMIND_PLUGIN_MODULES` 模块发现与 `refmind.plugins` entry
point。每个 hook 的异常独立记录，后续插件和核心流程继续执行。解析、分块、检索与生成
边界还会校验插件返回类型；不兼容返回值会被忽略。

同步插件在 multi-agent worker 中由管理器串行调用；异步 hook 的并发状态需插件自行保护。
插件代码与应用进程拥有相同权限，只应安装和启用可信插件。详细接口见
[plugins-and-mcp.md](./plugins-and-mcp.md)。

## 6. MCP 信任边界

MCP 客户端是可选依赖，支持 stdio 和 Streamable HTTP。服务必须在
`REFMIND_MCP_SERVERS` 中显式声明。`MCPContextProvider` 获取的内容默认
`allow_answer_use=False`，所以 `render_for_answer()` 返回空字符串；这防止未知服务绕过
“仅依据上传文献回答”的系统规则。只有业务代码完成来源、权限与质量审查后，才能显式放行。

## 7. 故障降级矩阵

| 故障 | 行为 |
|---|---|
| MinerU 不可用/超时/空结果 | 回退 PyMuPDF；扫描件仍无文本则给出明确错误 |
| 入库中途失败 | 按 `doc_id` 补偿；清理失败保留 `cleanup_failed` 记录 |
| 入库时进程强杀 | 下次应用启动扫描非 `ready` 记录并继续清理 |
| 并行批次中单篇 PDF 失败 | 仅回滚该篇；其他 PDF 继续，旧版本保持可用 |
| 规划模型失败 | 使用原问题单查询 |
| 某个子查询失败 | 保留其他子查询结果；全空时重试原问题 |
| 原问题检索抛出基础设施异常 | 返回“检索服务暂不可用”，不伪装成“文献无证据” |
| rerank/压缩失败 | 各组件内部降级；编排层仍保留原始候选 |
| 证据/答案审校失败 | 保留审校前证据或草稿 |
| 主 LLM 熔断 | 使用备选模型；half-open 仅放行一个探测 |
| 插件异常/返回类型错误 | 记录诊断并保留核心值 |
| MCP 未安装/未配置/连接失败 | 返回不可用探测结果，核心 RAG 不受影响 |
| 长期记忆提取/嵌入/SQLite 写入失败 | 跳过本轮记忆增强，论文问答继续完成 |
