# 分块规则

普通正文沿用 LangChain RecursiveCharacterTextSplitter，遇页码变化或标题则结束当前组合。
表格和独立公式保持完整，不受正文 1800 字符切分上限约束；is_atomic/oversized 标记状态。
公式仅与紧邻且符合提示词、长度和页距条件的下一段解释合并；保持 source_locations、页码范围，不重复索引解释。
旧表格输入路径同样保持完整。引用精度为页或已知页码范围，不补造上游未知定位。

新规则只影响重新分块的文献，现有索引不会自动迁移。
当前上下文打包仍可能截断，超大原子块也可能超过嵌入服务输入上限；本次仅保证分块输出完整，未宣称端到端完整性或召回提升。

测评详见 evaluation/reports/chunking_20261103/分块改进测评报告.md，复跑入口 scripts/eval_chunking.py。
