export interface User { id: number; username: string; is_admin: boolean | number; csrf: string }
export interface Library { id: number; title: string; kind: 'private' | 'shared'; role: 'owner' | 'editor' | 'viewer' }
export interface Session { id: number; group_id: number; name: string; created_at?: string }
export interface DocumentRow { id: number; group_id: number; paper_title?: string; filename: string; library_index?: number; summary?: string; status: string; num_chunks?: number; created_at?: string }
export interface PaperLocation { url: string; landing_url: string; is_oa?: boolean; license?: string; version?: string; provider?: string }
export interface Paper { paper_id: string; title: string; abstract: string; authors: string[]; year?: number; venue?: string; doi?: string; providers: string[]; locations: PaperLocation[]; citation_count?: number; is_retracted?: boolean; _search_id?: string }
export interface SelectedSpan { start: number; end: number; quote: string; reason?: string }
export interface Evidence { text: string; metadata: Record<string, any> & { evidence_view?: string; selected_spans?: SelectedSpan[] | string } }
export interface TraceDocument { text?: string; page_content?: string; metadata?: Evidence['metadata'] }
export interface TraceStage { documents?: TraceDocument[]; count?: number; status?: string; reason?: string; score_type?: string; invoked?: boolean }
export interface CompressionRecord { chunk_id?: string; identity?: { chunk_id?: string | number }; status?: string; reason?: string; before_chars?: number; after_chars?: number; selected_span_count?: number }
export interface AnswerTrace {
  pipeline?: string
  queries?: Array<{ query_id?: string; text?: string; status?: string }>
  stages?: Record<string, TraceStage>
  compression?: { enabled?: boolean; scorer?: string; status?: string; reason?: string; elapsed_seconds?: number; coverage_limit?: Record<string, unknown>; records?: CompressionRecord[] }
  packing?: Record<string, unknown>
  timings?: Record<string, number>
  scope?: Record<string, unknown>
}
export interface AnswerResult { answer: string; evidence?: Evidence[]; warnings?: string[]; degraded?: boolean; rejection_stage?: string; service_failed?: boolean; session_id?: number; mode?: string; trace?: AnswerTrace }
export interface Message { id: number; role: string; content: string; created_at?: string; result?: AnswerResult }
export interface Asset { id: string; session_id: number; doc_id?: number; paper: Paper; status: string; expires_at?: number; identity_status?: string; warnings?: string[]; version?: string; sha256?: string; coverage?: Record<string, any> }
export interface Task { id: string; kind: string; status: string; stage?: string; progress?: number; result: any; error?: string; created_at?: number; updated_at?: number }
export interface RuntimeConfig { external_search_enabled: boolean; chat_provider: string; embedding_provider: string; model_configured: boolean; temporary_retention_days: number; max_pdf_mb: number }
export const terminal = (status: string) => ['completed', 'failed', 'cancelled', 'interrupted', 'partial', 'needs_review'].includes(status)
export const taskLabels: Record<string, string> = { search: '文献检索', reading: '获取全文', answer: '科研问答', import: '加入文献库', upload: '上传入库', legacy_reindex: '索引维护' }
export const statusLabels: Record<string, string> = { ready: '已就绪', queued: '排队中', running: '进行中', completed: '已完成', failed: '未完成', cancelled: '已取消', interrupted: '已中断', partial: '部分可用', needs_review: '需要确认', expired: '已过期', downloaded: '已获取', parsing: '解析中', indexing: '建立索引', pending: '等待处理', maintenance_pending: '等待离线维护', reindexing: '重建待恢复', reindex_failed: '重建未完成' }
export function safeLink(value?: string): string | undefined { if (!value) return; try { const url = new URL(value); return ['https:', 'http:'].includes(url.protocol) && !url.username && !url.password ? url.href : undefined } catch { return } }
