# MinerU 后处理规则（2026-11-03）

本次按本机 MinerU 3.2.3 的 content_list V1 输出核对，未升级 MinerU。
已核对安装源码中的 make_blocks_to_content_list 与 table_merge：上游已处理表格正文、标题、脚注关联和跨页表格。
最新在线文档可能对应不同主版本，不能直接当作本机字段契约。

## 保留必要适配

- 统一块类型、已知页码和块标识；使用 text_level、list_items、image_caption、image_footnote。
- 表格直接使用上游 HTML/Markdown，包含原始 rowspan/colspan、空单元格、标题和脚注。
- blocks 的 mineru_source、表格的 source_items 保存原始内容项，方便追溯；这些字段只留在解析 JSON，不额外进入向量文本。
- 保留有效文字检查、解析失败回退、原子保存。

## 删除推断

移除列数/表头相似度/位置打分、首行猜表头、空值向下填充、重复表头删除。
不再把无法定位的全文 Markdown 当作第 1 页；有效内容缺少页码时触发 PyMuPDF 回退。

## 唯一的续表补充

只在两个独立表格满足全部条件时关联：内容列表中紧邻、页码相邻、明确且相同的阿拉伯数字表号、后一表标题包含 continued/cont./续表等标记。
原始片段、重复表头、脚注全部保留；记录 merge_method=caption_continuation。
其余表格原样保留，记录 merge_method=mineru；该值表示沿用上游，不表示已确认其跨页合并状态。
不支持用罗马数字表号推断续表；证据不足时保持分离。

## 限制与验证

- 上游已合并的表格只作为一个内容项接收，不重新拆分。当前 content_list 若只提供起始 page_idx，就只能保留该定位，不能据此确定跨页结束页。
- layout_confidence 仍是既有解析路径标签，不是准确率或模型置信概率。
- 旧解析 JSON 和索引未改写；新规则只对后续重新解析生效。
- 保留 HTML 可能增加文本长度；尚未证明召回提升，需要在后续授权的开发集测评中验证。
- 2026-11-03：解析专项 17 项通过（新增 10 项），完整离线回归 167 项通过；包括表格通过 parsed_to_documents 后仍保留空值、合并单元格和脚注。
- 未运行真实 MinerU 模型解析或付费 API；没有使用保留集，也没有重建知识库。
