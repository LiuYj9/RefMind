import type { AnswerResult, Evidence, SelectedSpan } from '../src/types'

const legacy: AnswerResult = { answer: 'A legacy answer' }
const span: SelectedSpan = { start: 0, end: 3, quote: '😀AB', reason: 'query_related' }
const compressed: Evidence = { text: '😀AB', metadata: { evidence_view: 'extractive_text', selected_spans: [span] } }
const traced: AnswerResult = { answer: 'Answer', evidence: [compressed], trace: {
  queries: [{ query_id: 'q1', text: 'original question', status: 'completed' }],
  stages: { reranked: { documents: [compressed], count: 1, score_type: 'rerank_score' } },
  compression: { enabled: true, records: [{ chunk_id: 'one', status: 'kept' }] },
  timings: { rerank: 0.2 },
} }
// @ts-expect-error: span offsets are numeric Unicode code-point offsets.
const invalidSpan: SelectedSpan = { start: '0', end: 3, quote: 'abc' }
// @ts-expect-error: unknown timing strings cannot be presented as measured seconds.
const invalidTrace: AnswerResult = { answer: '', trace: { timings: { rerank: 'fast' } } }
void [legacy, traced, invalidSpan, invalidTrace]
