# V3候选冻结与交接（2026-11-05）

当前冻结的是评测工具和真实待审库存，不是600题建成、人工审定通过或RAG效果验收通过。原 `real_v1.json` 字节、原70/30划分均保留；本代理没有执行保留题或调用计费模型。

后续窄修已保留组件实际返回的证据审查、警告、拒答阶段、任务描述和审核状态，并扩大报告依赖指纹；评测回归更新为67项通过。第一次冻结保留为历史，新冻结为 `candidate_freeze_v3_1.json`。根任务20开发题真实成本试跑已完成技术运行，旧报告五类审查字段各缺20条，未回填；详细边界见 `REPORT_METADATA_CHANGELOG.md`。

20题试跑索引早于最终document_processor/provenance缓存身份修复，只代表当时实现与原snapshot。旧report/snapshot保持原字节，不把该成绩计到最终业务版本；新冻结另记最终42项依赖哈希。

## 实际数据与缺口

| 轨道 | 目标 | 实际草案 | 开发/同域/跨域 | 人工批准 | 原始数量缺口 |
|---|---:|---:|---|---:|---:|
| 主问题 | 600 | 284 | 254/30/0 | 0 | 316 |
| 外检任务 | 120 | 72 | 72/0/0 | 0 | 48 |
| 四轮对话组 | 30 | 4（16轮） | 4/0/0 | 0 | 26组 |
| 结构PDF页 | 120 | 2 | 2/0/0 | 0 | 118页 |

主问题由原100、新增QASPER实际独立160题（75篇PDF）、材料/超导24题（2篇PDF）组成。新增160题全部是建设开发库存，不能抽40题回填独立验收。CS开发配额120题，现库存超额40；材料开发12/120、同域0/40；超导开发82/120，原同域30已列历史回归。按领域配额扣除CS超额及原30历史回归后，最多214题开发候选可供后续挑选，尚需146题配额匹配开发候选和240题全新独立保留候选。上述只是候选数量，人工正式批准数仍为0。论文家族独立性和最终抽样仍需人工核对，75个CS家族不能冒充原约60家族设计已经完成。

外检72任务来自24篇真实固定全文，分别提出身份定位、摘要边界、全文证据三个不同任务。全文问题可复用主轨道科学需求，不另加主问题计数。已知目标qrels不代表全网相关性已穷尽，摘要是否足够仍待人审，没有编造支持标签。结构2页是部分区域草案，不是完整页对象标注；只计已标对象召回/顺序，precision/F1为空，不声称表格单元格或公式语义等价质量。

正式登记只有 `datasets/construction_v3.json` 的6个dataset_files。早期10题/16题及候选元数据文件作为采集历史保留，不叠加计数。公开基准可能被模型训练见过；QASPER许可、来源、变更说明及该风险均在provenance记录。原文PDF、渲染区域、外部响应和报告放在本地忽略的workspaces/reports内，不随仓库分发。

## 旧保留集暴露审计

2026-11-05，一次定位阶段文档的只读搜索误包含旧评测JSON，工具输出暴露旧保留题片段。没有执行这些问题，没有据其调参，没有改原文件或70/30划分。迁移v3的顶层 `independence_audit` 记录了事件、受影响家族/划分及30题数量，不再复制题干。

这30题仅保留历史回归身份，不能计为下一轮全新盲验。`--independent-acceptance` 在检索/生成前拒绝开发题、暴露的历史保留题和旧非v3格式。人审批准不能恢复独立性；普通已审定历史运行仍标 `historical_regression`、`official=false`、`independent_eligible=false`。真实CLI已验证旧30被拦截，未触发查询。下一轮同域/跨域独立验收候选当前均为0，需要新家族补足。

## 阶段2.0—2.9执行状态

| 阶段 | 已可执行 | 本次已执行的证据 | 尚未执行/未通过的范围 |
|---|---|---|---|
| 2.0 数据与测量基础 | v3加载/验证、家族与划分隔离、事实槽AND/OR、专项指标、人审导入与审计 | 64项离线评测回归通过；6份登记草案来源哈希/物理页/原文摘录验证；库存统计 | 600/120/30/120建成、人类审定、类别和家族最终抽样 |
| 2.1 冻结基线 | 数据/实现SHA、索引快照、legacy/research分开报告、开发草案标识 | 原real_v1哈希不变；当前候选冻结清单；开发BM25诊断可复现；两个真实Crossref响应已冻结 | 完整科研流程的正式质量与成本基线；冻结基线不等于效果合格 |
| 2.2 解析、切分与索引 | prepare隔离索引、解析/切分实现指纹、结构对象指标 | 两页真实PDF的PyMuPDF局部结构诊断；实际PDF来源字节验证 | MinerU真机结构质量比较；表示与chunk大小分离消融；全量索引效果评测 |
| 2.3 召回与重排 | 显式legacy/research调用、原始/重排候选分段证据指标、白名单参数有效值核验 | 手算/反例测试及原70开发页级BM25诊断 | research真实多域召回/数值单位/中英/图表专项、重排后端与候选数消融；BM25页诊断不能代替生产混合召回 |
| 2.4 上下文组织 | packed、generation_context、final_evidence分段；上下文事实覆盖用实际生成输入 | 装配有证据但生成上下文为空的反例验证；不以审核前packed或最终引用替代 | 各token预算下覆盖、丢失率、延迟和答案正确性的真实实验 |
| 2.5 生成、拒答与引用 | 结构化拒答/错误分离、实际上下文保存、saved-answer独立裁判入口 | 离线状态/异常/证据页与事实槽反例；没有用模拟输出当真实回答 | 本代理未调用真实生成/裁判；拒答边界、数值条件正确率、引用支持率和正式质量门槛 |
| 2.6 外部论文发现 | external独立runner、冻结provider HTTP重放、qrels0/1/2、未判相关性留空 | 两个真实Crossref DOI响应经生产discover_papers适配器重放，已知目标Recall@1=1，零网络/模型 | 72任务完整外检运行、宽泛探索排序、未判结果人审及120任务建成；两例题录定位不代表发现质量普遍通过 |
| 2.7 全文获取与研读 | fetch身份、固定文件SHA、同SHA解析、全文证据到场分阶段指标；unknown身份不能算成功 | 真实公开PDF下载、固定版本/哈希及页证据；全文计分反例 | 生产discover→fetch_public_pdf→parse→research answer整链真实执行、成功率/限时/外部边界验收；来源下载不能冒充该整链通过 |
| 2.8 多模态与多轮 | 4轮组级runner、隔离会话与BaseMessage历史、显式范围切换；区域/公式字面指标 | 4组真实开发脚本、2页图像区域实际渲染与部分目视核对；会话重置/组失败反例 | 4组真实模型会话、图像语义/曲线趋势/公式语义/表格单元格评测；30组与120页建成 |
| 2.9 独立验收与试运行 | 独立验收门禁、具名人审、分阶段报告、运行时有效参数、隔离存储 | 67离线测试；暴露30的真实CLI前置拒绝；根任务20开发题技术运行完成且审查元数据缺测已留记录 | 全新独立保留集、人审通过、正式验收；20开发成本试跑不算盲验或本代理模型运行 |

## 交付文件归属

仅编辑evaluation目录以及两份评测测试，没有修改业务RAG、settings、graph、reranker或用户日常库。

- 修改：`evaluation/dataset.py`、`evaluator.py`、`run_eval.py`、`worker.py`、`README.md`。
- 新增契约/统计/指标：`evaluation/v3_contract.py`、`construction.py`、`runtime.py`、`metrics/fact_evidence.py`。
- 新增专项与人工工具：`evaluation/suites.py`、`run_suite.py`、`review.py`。
- 新增真实来源采集/草案构建：`evaluation/public_sources.py`、`build_public_seeds.py`、`build_external_drafts.py`。
- 新增回归：`tests/test_evaluation_v3.py`、`tests/test_evaluation_suites.py`。
- 数据、人工包与冻结：`evaluation/datasets/`新增草案及清单、`evaluation/review_packets/2026-11-05_xuechan999/`、本状态文档及`evaluation/candidate_freeze_v3.json`。

索引快照记录解析/切分配置和实现哈希。`--set`仅允许白名单RAG参数，子进程加载settings之前注入，核验请求与有效值一致；密钥、存储路径、插件配置不可注入。实现/索引不匹配时显式失败，不静默复用或重建。

## 人工复核包与导回

`review_packets/2026-11-05_xuechan999/manifest.json` 指派“雪蝉999”，不宣称领域专长或复核完成。共332个开发审查单元（254主问题、72外检、4对话组、2结构页），旧30历史回归未导出。每份`*.tasks.csv`是可填写检查表；`*.review.json`含逐题pending决策，JSON是正式导回依据。

先在JSON填写reviewed_at，以及真正审过的题目status/comment/patches；未审项保留pending。本地正例批准必须有最终类别与事实槽。随后运行：

```powershell
.\.venv\Scripts\python.exe -B -m evaluation.review apply --golden evaluation/datasets/public_materials_hts_dev_v3.json --decisions evaluation/review_packets/2026-11-05_xuechan999/public_materials_hts_dev_v3.review.json --output evaluation/datasets/public_materials_hts_reviewed_v3.json
```

工具校验原题集SHA，创建新的题集和review-audit文件，不覆盖原文件。当前没有执行真实批准。已生成人审包，名字只是指派，不等于任何题目成为human_approved。

## 已执行报告

- 审计后当前SHA的原70开发BM25页诊断：`evaluation/reports/retrieval_134dfbf15f8242b2a954861fae24a6d4.json`。仅词法页级开发诊断，不能代替生产混合检索效果。
- 外检冻结题录两例：`evaluation/reports/external_fb4fb391cd9745feb6db75088c4d2b71.json`。零网络/模型，定位结果不等于全文成功。
- 局部结构两页：`evaluation/reports/structure_4a44e1b1cb86410fbfdf0f99fb850ab2.json`。零模型，PyMuPDF回退解析；不作为MinerU质量结果。
- 64项评测离线回归包含事实AND/OR、单位大小写、错页/错文、阶段证据丢失、源范围隔离、未知外检相关性、四轮组指标、部分结构标注、人工导回审计、运行时参数消费、历史暴露独立性门禁等。

后续任何数据/标注/实现变化均应生成新的冻结记录并重验对应层级；不能以本记录声称上线、部署或正式科学质量已通过。
