import type { Evidence, TraceDocument, AnswerTrace } from './types'

export const stageLabels: Record<string, string> = {
  bm25: '关键词召回', vector: '向量召回', fused: '排名融合', direct_merged: '多查询候选去重',
  expanded: '扩展候选', raw: '检索输出', raw_candidates: '融合候选',
  reranked: '重排', compressed: '上下文压缩', selected: '选定证据',
  extractive_views: '摘录压缩视图', packed: '预算装包', evidence_review_input: '证据审查输入', evidence_review_output: '证据审查输出',
  evidence_guard_output: '证据范围检查', generation_budget_output: '生成前预算检查',
  generation_context: '生成上下文', final_evidence: '最终证据',
}
export const timingLabels: Record<string, string> = {
  retrieval: '检索', rerank: '重排', compression: '压缩', pack: '压缩与装包',
  evidence_review: '证据审查', generation: '答案生成', answer_review: '答案审校', answer_total: '问答总耗时',
}
export function finite(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value) }
export function metric(value: unknown, missing = '未采集'): string {
  return finite(value) ? Number(value.toPrecision(7)).toString() : missing
}
export function compressionStatus(compression?: AnswerTrace['compression']): string {
  if (!compression) return '未记录压缩明细'
  if (compression.status === 'fallback_original') return '压缩调用失败，已保留原始片段'
  if (compression.status) return `压缩状态：${compression.status}`
  return compression.enabled === false ? '本轮压缩未启用' : compression.enabled === true ? '本轮压缩已启用' : '压缩启用状态未记录'
}
export function documentEvidence(document: TraceDocument): Evidence | undefined {
  const text = typeof document?.text === 'string' ? document.text : document?.page_content
  if (typeof text !== 'string' || !document.metadata || typeof document.metadata !== 'object') return
  return { text, metadata: document.metadata }
}
export function evidenceLabel(evidence: Evidence): string {
  const meta = evidence.metadata
  return String(meta.paper_title || meta.filename || meta.chunk_id || '未命名来源')
}
export function scoreRows(evidence: Evidence) {
  const meta = evidence.metadata
  const bm25 = meta.retrieval_channels?.bm25, vector = meta.retrieval_channels?.vector, rrf = meta.retrieval_fusion
  return [
    { label: 'BM25 名次', value: metric(bm25?.rank, '未记录') },
    { label: 'BM25 原始分', value: metric(bm25?.score ?? bm25?.raw_score) },
    { label: '向量名次', value: metric(vector?.rank, '未记录') },
    { label: '向量距离', value: metric(vector?.distance) },
    { label: 'RRF 名次', value: metric(rrf?.rank, '未记录') },
    { label: 'RRF 融合分', value: metric(rrf?.score ?? rrf?.rrf_score) },
    { label: '重排分', value: metric(meta.rerank_score) },
  ]
}
export function scoreGroups(evidence: Evidence) {
  const runs = Array.isArray(evidence.metadata.retrieval_runs)
    ? evidence.metadata.retrieval_runs.filter((run: unknown) => run && typeof run === 'object') : []
  if (!runs.length) return [{ label: '本次检索记录', note: '', rows: scoreRows(evidence) }]
  return [
    ...runs.map((run: any) => ({
      label: `查询 ${String(run.query_id || '身份未记录')}${run.text ? ` · ${run.text}` : ''}`,
      note: run.rrf_score_source === 'derived_from_observed_ranks' ? 'RRF 分由本查询已记录名次计算；各查询的分数不相加。' : '',
      rows: scoreRows({ text: evidence.text, metadata: {
        retrieval_channels: run.channels,
        retrieval_fusion: { ...run.fusion, rrf_score: run.rrf_score ?? run.fusion?.rrf_score },
      } }).slice(0, 6),
    })),
    { label: '本轮重排', note: '', rows: [{ label: '重排分', value: metric(evidence.metadata.rerank_score) }] },
  ]
}
export function budgetRows(packing?: AnswerTrace['packing']) {
  const labels: Record<string, string> = {
    evidence_budget: '证据 token 预算', model_context_tokens: '模型上下文上限', output_reserve: '回答预留 token',
    evidence_tokens: '证据 token 估计', used_tokens: '已用 token 估计', input_tokens: '输入 token 估计',
    estimated_input_tokens: '输入 token 估计', total_tokens: '总 token 估计',
  }
  return Object.entries(labels).filter(([key]) => finite(packing?.[key])).map(([key, label]) => ({ label, value: metric(packing![key]) }))
}
