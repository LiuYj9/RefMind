# 阶段1：OpenAI接入与基线复现

默认保持原有DashScope配置。密钥只保存在本机.env，不进入Git。CHAT_PROVIDER、EMBEDDING_PROVIDER、VISION_PROVIDER分别控制文本、嵌入和视觉用途；设置为openai才启用相应OpenAI适配。

文本模型固定为gpt-5.4-mini-2026-03-17，嵌入固定为text-embedding-3-small。OpenAI嵌入仅允许在带.refmind-eval标记且Chroma路径位于该目录内的独立工作区使用。设置页尚未提供完整供应商管理，使用环境变量配置。

## 预算与用量

账本为data/openai_usage.sqlite3，使用SQLite事务预占。默认每日2美元、累计10美元；本次用户授权总额10美元，实验进程每日上限设为10美元，累计仍为10美元。未知用量保留预算占用，正常响应按API usage结算。缓存折扣未扣除，费用为标准单价估算，不替代官方账单。白名单外模型禁止调用，SDK与底层重试为0，不自动跨供应商降级。

## PowerShell复现

以下操作向OpenAI发送指定开发集内容并产生费用；先取得资料使用授权。不要运行保留集。

```powershell
$env:CHAT_PROVIDER='openai'
$env:EMBEDDING_PROVIDER='openai'
$env:OPENAI_TOTAL_BUDGET_USD='10'
$env:OPENAI_DAILY_BUDGET_USD='10'
$env:USE_FALLBACK_PARSER='true'
$env:MULTI_AGENT_ENABLED='false'
$env:RERANK_ENABLED='false'
$env:CONTEXT_COMPRESSION_ENABLED='false'
$env:REFMIND_EVAL_TEXT_ONLY='true'
$env:REFMIND_EVAL_TIMEOUT_SECONDS='7200'
$env:PYTHONIOENCODING='utf-8'
# 只有本机确实有该代理时设置；直连可省略。
$env:OPENAI_PROXY='http://127.0.0.1:7890'

.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode prepare --golden evaluation\datasets\real_v1.json --workspace evaluation\workspaces\NEW_DEV_RUN --allow-api
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode retrieval --golden evaluation\datasets\real_v1.json --workspace evaluation\workspaces\NEW_DEV_RUN --allow-api
.\.venv\Scripts\python.exe -B evaluation\run_eval.py --mode answer --golden evaluation\datasets\real_v1.json --workspace evaluation\workspaces\NEW_DEV_RUN --allow-api
.\.venv\Scripts\python.exe -B scripts\score_phase1.py evaluation\reports\answer_RUN_ID.json --allow-api
```

首次构建必须使用新目录，不覆盖已有快照。评分只读取dev答案，固定联合评分规则、同模型、2个工作线程，保存逐题理由及规则哈希，不重新生成答案。

## 已知边界

- 纯文本开关必须开启，否则既有图文问答可能进入原视觉供应商分支。
- 固定拒答去除参考来源后精确判断；混合回答/拒答保留待复核。
- 同步invoke和stream已验收；异步直连未验收。
- 计费传输层先完整读取响应，因此stream首字延迟会增加，不宣传首字性能。
- 本次未显式设置OpenAI temperature，使用接口默认值，不保证确定性。
- 同模型自动裁判不是人工认证；题集仍待人类审定。
- 本说明不授权进入阶段2，不自动提交或发布。
