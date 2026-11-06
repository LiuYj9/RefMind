# RefMind 分阶段评测

当前登记的是**草案库存**：284个主问题（254开发、原30同域保留历史回归）、72开发外检任务、4组四轮开发对话、2个开发结构页，人工审定数全部为0。尚未达到600/120/30/120，不构成正式验收。新增CS/ML的160题来自75篇实际PDF匹配的QASPER训练论文，均已用于建设、全部保持开发集；相对CS开发配额多40题，仅作为候选库存，不转作盲测。公开基准可能出现在模型训练中，不能视为全新未见题。新材料/超导24题来自2篇固定版本全文；另有来源候选元数据，不计成题数。

旧 `real_v1.json` 原文件与70/30划分保持不变。2026-11-05一次只读文档搜索误覆盖旧JSON，输出出现旧保留题片段；没有执行这些问题，也没有据其调参。迁移v3的顶层 `independence_audit` 如实记录旧30的暴露，不能再算作下一轮全新盲验。`--independent-acceptance` 会拒绝开发题、旧暴露保留题及无v3审计的旧格式；人工批准不能恢复独立性。普通运行即使未来人工审定，也明确标 `historical_regression`、`official=false`、`independent_eligible=false`。后续需补新家族完成独立验收；当前独立同域/跨域保留候选均为0。

当前使用的数据文件由 `datasets/construction_v3.json` 的 `dataset_files` 唯一登记；早期10/16题采集中间文件不能叠加计数。公共PDF、来源响应和运行产物位于被忽略的 `evaluation/workspaces/`，不随仓库分发。QASPER问题/答案按CC BY 4.0记录归属；PDF各自的许可单独登记，未知许可不会标成已确认。

## 科研流程与可复现实验

`--pipeline research` 调用与产品相同的 `answer_research_question / retrieve_research_evidence`；默认 `legacy` 保留旧基线。两条路径报告独立。`--set KEY=VALUE` 在隔离子进程加载配置前注入，并核对有效值；只接受白名单科研参数，不能写密钥或存储路径。

```powershell
.\.venv\Scripts\python.exe -B -m evaluation.run_eval --mode prepare --golden evaluation/datasets/public_materials_hts_dev_v3.json --pipeline research --split dev --allow-draft --allow-api --workspace evaluation/workspaces/materials-research-dev
.\.venv\Scripts\python.exe -B -m evaluation.run_eval --mode answer --golden evaluation/datasets/public_materials_hts_dev_v3.json --pipeline research --split dev --allow-draft --allow-api --workspace evaluation/workspaces/materials-research-dev --set CONTEXT_MAX_TOKENS=6000 --set MODEL_CONTEXT_TOKENS=32768 --set ANSWER_TOKEN_RESERVE=4096 --set MULTI_AGENT_CONFIRM_REFUSAL=false
```

切分/嵌入配置变化必须重建独立工作目录。检索/重排/压缩/审核旋钮可使用同一固定索引，报告记录实际有效值。回答的主上下文分数使用 `trace.stages.generation_context`（实际送进生成器）；`packed`是审核前装配，`final_evidence`是最终引用，均单独报告，不能替代生成上下文。阶段缺失留空，不补造。

## 外检、多轮、结构的独立执行器

```powershell
# 两个真实Crossref DOI响应的冻结重放：调用生产题录适配器，零模型/零网络
.\.venv\Scripts\python.exe -B -m evaluation.run_suite --track external --golden evaluation/workspaces/public-source-v3/external_lookup_subset.json --replay evaluation/workspaces/public-source-v3/external_lookup_replay.json --backend provider_replay --allow-draft
# 两个实际PDF页的本地解析诊断；不调用MinerU或模型
.\.venv\Scripts\python.exe -B -m evaluation.run_suite --track structure --golden evaluation/datasets/public_structure_dev_v3.json --backend parser --parser pymupdf --allow-draft
# 对话先用run_eval prepare准备对应题集，然后运行整组四轮；此命令可能计费
.\.venv\Scripts\python.exe -B -m evaluation.run_suite --track dialogue --golden evaluation/datasets/public_dialogues_dev_v3.json --backend production --pipeline research --workspace evaluation/workspaces/dialogue-dev --allow-draft --allow-api
```

`recorded`后端还支持固定输出文件，严格绑定dataset SHA；标为重放，不宣称新运行整条生产链路。外检qrels区分已判/未判相关性，未判结果不冒充负例；题录、摘要、身份核验的PDF下载、同哈希解析、全文证据分别报告。`identity_status=unknown`不算全文成功。72外检开发草案为24篇论文分别提出身份定位、摘要边界、全文证据三种任务；摘要支撑判断仍待人工，未编造分数。全文证据到场不等于答案正确。对话按四轮全通过计算整组率，使用独立会话，报告状态/文献范围/证据行为；还不是通用语义答案正确率。结构指标区分对象位置/顺序/公式字面匹配；部分区域标注仅计已标对象召回，不给全页precision/F1，不声称表格单元格或公式语义等价正确率。

## 雪蝉999的首批人工复核包

`review_packets/2026-11-05_xuechan999/manifest.json` 登记6份开发复核包，共332个审查单元。每份 `*.tasks.csv` 可逐题记录检查项，`*.review.json` 可填写 `reviewed_at` 与每题 `status/comment/patches`。JSON是正式导回文件，CSV仅作任务清单。指派复核人不代表已审定或具备某个领域专长；未复核项保留pending。原30保留题未导出给本轮开发复核。

```powershell
.\.venv\Scripts\python.exe -B -m evaluation.review export --golden evaluation/datasets/public_materials_hts_dev_v3.json --output evaluation/review_packets/materials.review.json --task-table evaluation/review_packets/materials.tasks.csv --assigned-reviewer 雪蝉999
.\.venv\Scripts\python.exe -B -m evaluation.review apply --golden evaluation/datasets/public_materials_hts_dev_v3.json --decisions evaluation/review_packets/2026-11-05_xuechan999/public_materials_hts_dev_v3.review.json --output evaluation/datasets/public_materials_hts_reviewed_v3.json
```

导回要求原题集SHA一致、具名及日期完整、逐题明确决策；本地正例批准还须完成最终类别与事实槽。导回创建新版本和审计文件，不能覆盖原题集或已有输出。示例导回命令需等人完成复核后执行，当前未执行真实审批。

## V3：科研证据契约与建设进度

新版增加 `dev / id_holdout / ood_holdout`、独立 `corpus_scope`、文献家族和事实槽。
`holdout` 是 `id_holdout` 的兼容别名。开发题的检索范围只能含开发来源；验收题可使用冻结全库作干扰，
但正向标准证据的文献家族必须与问题划分相同。划分题目不再隐式等于裁剪检索库。

旧 `datasets/real_v1.json` 保持原字节；`real_v3_draft.json` 只迁移实际100题，保留70/30。
原标签仍是待人审草案，未自动生成事实槽，也没有把所有旧事实题冒充数值题、条件题冒充总结题。

`datasets/construction_v3.json` 是建设配额，不是已完成数据：600个主问题、120个外检任务、
30组固定四轮会话、120个结构标注PDF页。主问题最终类别配额为数值100、方法70、表格80、
公式60、图50、跨论文70、总结50、无答案120。不要用翻译、标点、provider或重复运行增加题数。

```powershell
# 显示实际题数、人审数和缺口；不读取模型或调用外部服务
.\.venv\Scripts\python.exe -B -m evaluation.construction status
# 仅验证结构、来源字节和摘录；不运行任何保留题
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode validate --golden evaluation\datasets\real_v3_draft.json
# 明确允许开发草案诊断，报告标 draft_diagnostic，不能当正式质量验收
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode retrieval --backend bm25 --golden evaluation\datasets\real_v3_draft.json --split dev --allow-draft
```

任何正式运行需要标注 `human_approved`，并提供具名、日期完整的 `human_review`；本地正例还需要人工审定的事实槽。
`--unlock-holdout` 只解锁选择，不跳过人审。`--allow-draft` 仅对开发集有效，不能绕过任一验收集。
历史v1/v2诊断接口仍能读取旧文件，但CLI实际运行同样要求显式草案选择或完整人审。

事实槽用 `fact_slots[].evidence_sets` 表达证据依赖。例如 `[["e1","e2"],["e3"]]` 表示：
e1与e2同时具备，或者单独具备e3，才覆盖该事实槽；所有required槽覆盖才算该题证据齐全。
文本证据要求正确文献、物理页和实际上下文中的原文摘录同时匹配，区分mA/MA，不把整页命中当作数值和条件完整。
旧迁移题没有事实槽，指标显示null和已标注分母，不返回满分。
该指标衡量证据是否到场，不等于答案数值正确率；回答仍需独立人工或裁判核对。

报告记录供应链实际提供的trace、阶段证据指标、模型预算和异常状态；未提供的阶段明确缺测，不能伪造。
本地RAG runner拒绝把外检、多轮和结构任务按PDF页Recall套算；它们必须使用各自适配器。
visual证据需要区域、坐标系、渲染哈希及人工观察，原生PDF文本检验不会假装验证图像中的事实。

新增来源和题目应先登记家族/来源/许可并冻结文件哈希。模型草拟题目始终保持draft，只有人类核对后才能批准。
跨域保留域是机械热流、化工过程；开发期不通过其题干或答案调参。

阶段 0 建立评测工具和数据基线，不接入新供应商、不调用付费模型。原有 6 题合成数据保留为冒烟样例；真实数据见 `datasets/real_v1.json`。

## 运行环境

从项目根目录使用 `.venv\Scripts\python.exe`。当前实测解释器为 Python 3.13.12；不使用裸 `python` 或其他 Conda 环境代替。

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m pip check
```

## 五种独立模式

默认运行 `validate`。`--no-judge` 现为 `retrieval` 的兼容别名，不再隐式生成答案。新的入口不会自动入库、创建合成 PDF 或重新建立索引。

| 模式 | 工作 | 模型调用 |
|---|---|---|
| validate | 检查标注结构、PDF SHA256、物理页码及证据摘录 | 无 |
| prepare | 显式准备独立评测索引 | 可能产生解析／嵌入费用 |
| retrieval | 只检索并计算证据页指标 | 生产后端的查询嵌入可能计费 |
| answer | 运行真实问答，保存实际上下文和答案 | 可能计费，不运行裁判 |
| judge | 只读取已保存的答案与上下文评分 | 可能计费，不重新检索／生成 |

### 阶段 0：零 API 费用

```powershell
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode validate --golden evaluation\datasets\real_v1.json
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode retrieval --backend bm25 --golden evaluation\datasets\real_v1.json --split dev --allow-draft
```

`bm25` 是逐 PDF 页的词法诊断基线，不等于 RefMind 生产混合检索。它没有向量搜索、重排、生成或拒答能力，不能据此宣称答案正确率、幻觉率或线上性能。报告的耗时仅为已加载词法索引上的检索调用，不包含解析和建索引。

### 后续阶段：显式允许付费操作

以下命令不属于阶段 0 验收运行内容。只有主动增加 `--allow-api` 才能执行。

```powershell
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode prepare --golden evaluation\datasets\real_v1.json --workspace evaluation\workspaces\dev-v1 --allow-api --allow-draft
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode retrieval --golden evaluation\datasets\real_v1.json --workspace evaluation\workspaces\dev-v1 --allow-api --allow-draft
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode answer --golden evaluation\datasets\real_v1.json --workspace evaluation\workspaces\dev-v1 --allow-api --allow-draft
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode judge --input evaluation\reports\answer_RUN_ID.json --allow-api --allow-draft
```

准备时为每次实验创建专用工作目录，只接受 `evaluation/workspaces/` 下的子目录，不覆盖已经准备好的目录。检索和回答必须显式使用已有快照，语料、划分、嵌入或切分配置不一致时拒绝运行，不静默重建。

生产后端在独立子进程中先设置数据库、Chroma、上传、解析及图片路径，再导入应用。不会修改父进程 settings、`.env` 或日常文献库；禁用用户记忆、外部学术检索、第三方插件和图片入库摘要。问答保留现有生产多查询、重排及压缩配置，并把有效配置写进报告。

失败及中断的评测目录保留供排查，不自动删除日常文件。运行日志位于评测工作目录，目录整体不进入 Git。准备失败后显式使用新的实验目录重试。

## 数据与保留集

真实题集覆盖 10 份已上传论文：事实 30、跨论文比较 20、表格及邻近上下文 10、术语 10、无答案 20、条件与不可直接比较 10。按文献划分 70 题开发集与 30 题保留集。

题集为 **AI 助手编写、已核对来源的标注草案，尚未获得人工审定**。证据摘录自动校验只能证明文本出现在该页，不能证明标准答案和拒答标签一定正确。人审流程见 `datasets/ANNOTATION_GUIDE.md`。

`validate` 可以检查全部标注，但不运行任何问题；检索／回答默认只运行开发集。选择 `--split holdout` 或 `all` 需要显式 `--unlock-holdout`，该选项仅用于最终发布验收。开发期不得用保留集分数调参。被查看并用于修复的保留题应转入回归集，补充新的独立验收题。

PDF 原文件仍在本机 `data/uploads/`，没有复制进版本库。换机时需按清单恢复同字节文件；缺少原文件或 SHA256 改变会导致校验失败，不会自动下载替代版本。

## 指标口径与失败统计

- 证据位置按 1 起始的 PDF 物理页标注，不使用论文印刷页。
- Recall@K 计算前 K 个结果覆盖的唯一证据页比例。
- MRR 计算第一个正确证据结果的倒数排名；缺少有效页码不算命中。
- nDCG@K 使用二值新增证据页增益；重复页／重复范围不重复加分，理想排序使用证据页数。跨页结果至多贡献一次排序增益。
- 原始召回与答案最终上下文分别报告，不能互相代替。
- 服务异常、空答案和裁判失败显式标记，不能计为成功拒答。无法结构化判断的拒答状态保留为 unknown，不通过关键词猜测。
- 每项均报告 eligible／scored 数量；缺测显示 null，部分失败的报告标记 partial，不得当作完整通过。
- 阶段 0 不设置 RAG 效果发布门槛，真实模型基线与质量门槛从后续阶段建立。

JSON 保存逐题结果与上下文，Markdown 提供可读摘要。报告包括运行标识、北京时间、数据集 SHA256、实现文件 SHA256、环境、模式和配置；报告及工作目录都是本地运行产物。

## 验收边界

阶段 0 的工程验收包括完整离线回归、指标手算反例、模式边界、子进程存储隔离、题集原文校验和开发集离线诊断。题集人工审定、真实付费模型效果和完整生产后端 API 烟测分别明确列为未完成项，不用 mock 结果代替。
