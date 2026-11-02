# RefMind 分阶段评测

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
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode retrieval --backend bm25 --golden evaluation\datasets\real_v1.json --split dev
```

`bm25` 是逐 PDF 页的词法诊断基线，不等于 RefMind 生产混合检索。它没有向量搜索、重排、生成或拒答能力，不能据此宣称答案正确率、幻觉率或线上性能。报告的耗时仅为已加载词法索引上的检索调用，不包含解析和建索引。

### 后续阶段：显式允许付费操作

以下命令不属于阶段 0 验收运行内容。只有主动增加 `--allow-api` 才能执行。

```powershell
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode prepare --golden evaluation\datasets\real_v1.json --workspace evaluation\workspaces\dev-v1 --allow-api
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode retrieval --golden evaluation\datasets\real_v1.json --workspace evaluation\workspaces\dev-v1 --allow-api
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode answer --golden evaluation\datasets\real_v1.json --workspace evaluation\workspaces\dev-v1 --allow-api
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode judge --input evaluation\reports\answer_RUN_ID.json --allow-api
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
