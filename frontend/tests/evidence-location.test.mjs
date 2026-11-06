import test from 'node:test'
import assert from 'node:assert/strict'
import { findQuoteMatches, isEvidencePage, selectedSpans, sourceBBox, resolveCitation } from '../src/evidenceLocation.ts'
import { scoreRows, scoreGroups, budgetRows, compressionStatus, documentEvidence, stageLabels } from '../src/tracePresentation.ts'

const base = 'https://refmind.example/read'
const evidence = (overrides = {}) => ({ text: 'Source passage', metadata: { group_id: 2, doc_id: 4, page: 8, paragraph_index: 1, chunk_id: 'first', bbox: [10, 20, 100, 120], ...overrides } })

test('selected spans use Unicode code points, not UTF-16 units', () => {
  const item = evidence({ evidence_view: 'extractive_text', selected_spans: [{ start: 3, end: 6, quote: '😀AB', reason: 'relevant' }] })
  assert.equal(selectedSpans(item).length, 1)
  item.metadata.selected_spans[0].end = 7
  assert.deepEqual(selectedSpans(item), [])
  assert.deepEqual(selectedSpans(evidence()), [])
})
test('serialized spans validate shape before using quotes', () => {
  assert.equal(selectedSpans(evidence({ evidence_view: 'extractive_text', selected_spans: '[{"start":0,"end":3,"quote":"abc"}]' })).length, 1)
  assert.deepEqual(selectedSpans(evidence({ evidence_view: 'extractive_text', selected_spans: 'invalid' })), [])
})
test('quotes match across PDF text spans with exact character boundaries', () => {
  const found = findQuoteMatches(['Prefix induced ', 'eddy current', ' tail'], ['induced eddy current'])
  assert.equal(found.matches.length, 1)
  assert.deepEqual(found.matches[0].start, { item: 0, offset: 7, end: 8 })
  assert.equal(found.matches[0].end.item, 1)
  assert.equal(found.matches[0].end.end, 12)
})
test('NFKC ligatures and astral characters keep DOM UTF-16 offsets', () => {
  const found = findQuoteMatches(['😀ﬁeld loss'], ['field'])
  assert.equal(found.matches[0].start.offset, 2)
  assert.equal(found.matches[0].end.end, 6)
})
test('full independent quotes are used, not a compressed concatenation or prefix', () => {
  const text = 'a'.repeat(95) + 'different ending'
  const result = findQuoteMatches([text, 'second retained sentence'], ['a'.repeat(95) + 'wrong ending', 'second retained sentence'])
  assert.equal(result.matches.length, 1)
  assert.equal(result.missing.length, 1)
})
test('ambiguous repeated quotes are not falsely located to the first occurrence', () => {
  const found = findQuoteMatches(['same quote same quote'], ['same quote'])
  assert.equal(found.matches.length, 0)
  assert.deepEqual(found.ambiguous, ['same quote'])
})
test('bbox fallback only applies on the recorded source page and must be valid', () => {
  assert.equal(isEvidencePage(evidence({ page_start: 8, page_end: 9 }), 9), true)
  assert.equal(sourceBBox(evidence({ page_start: 8, page_end: 9 }), 9), undefined)
  assert.deepEqual(sourceBBox(evidence(), 8), [10, 20, 100, 120])
  assert.equal(sourceBBox(evidence({ bbox: [100, 20, 10, 120] }), 8), undefined)
})
test('local citations distinguish chunks on the same paper page', () => {
  const first = evidence(), second = evidence({ chunk_id: 'second', paragraph_index: 2 })
  const resolved = resolveCitation('?open_citation=g2-d4-p8-r2', [first, second], base)
  assert.equal(resolved.evidence, second)
  assert.equal(resolved.exact, true)
})
test('resource chunk_id takes precedence; paragraph is used when chunk_id is absent', () => {
  const first = evidence({ resource_id: 'resource' }), second = evidence({ resource_id: 'resource', chunk_id: 'second', paragraph_index: 2 })
  assert.equal(resolveCitation('?open_resource=resource&page=8&chunk_id=second&paragraph=1', [first, second], base).evidence, second)
  assert.equal(resolveCitation('?open_resource=resource&page=8&paragraph=2', [first, second], base).evidence, second)
})
test('missing precise identities and ambiguous old links degrade to the page, never another quote', () => {
  const items = [evidence(), evidence({ chunk_id: 'second', paragraph_index: 2 })]
  for (const href of ['?open_citation=g2-d4-p8-r99', '?open_citation=g2-d4-p8-r0']) {
    const result = resolveCitation(href, items, base)
    assert.equal(result.exact, false)
    assert.equal(result.evidence.text, '')
    assert.equal(result.evidence.metadata.bbox, undefined)
    assert.equal(result.page, 8)
  }
  assert.equal(resolveCitation('?open_citation=g2-d4-p8-r0', [items[0]], base).evidence, items[0])
})
test('raw scores and distances are never derived from ranks; zero remains a valid score', () => {
  const rows = scoreRows(evidence({ retrieval_channels: { bm25: { rank: 1 }, vector: { rank: 2, score: 0.7 } }, retrieval_fusion: { rank: 1, c: 60, weights: [0.5, 0.5] }, rerank_score: 0 }))
  const values = Object.fromEntries(rows.map(row => [row.label, row.value]))
  assert.equal(values['BM25 原始分'], '未采集')
  assert.equal(values['向量距离'], '未采集')
  assert.equal(values['RRF 融合分'], '未采集')
  assert.equal(values['重排分'], '0')
})
test('trace accepts either serialized document shape and exposes only known finite budget values', () => {
  assert.equal(documentEvidence({ page_content: 'text', metadata: {} }).text, 'text')
  assert.equal(documentEvidence({ metadata: {} }), undefined)
  assert.deepEqual(budgetRows({ evidence_budget: 4000, model_context_tokens: Infinity, unknown_secret: 'hidden' }), [{ label: '证据 token 预算', value: '4000' }])
})
test('multiquery ranks stay attached to each observed query without summing scores', () => {
  const groups = scoreGroups(evidence({ rerank_score: 0.3, retrieval_runs: [
    { query_id: 'q1', text: 'question', channels: { bm25: { rank: 2 } }, fusion: { rank: 3 }, rrf_score: 0.008, rrf_score_source: 'derived_from_observed_ranks' },
    { query_id: 'q2', text: 'angle', channels: { vector: { rank: 1 } }, fusion: { rank: 1 }, rrf_score: 0.016 },
  ] }))
  assert.equal(groups.length, 3)
  assert.equal(groups[0].rows.find(row => row.label === 'RRF 融合分').value, '0.008')
  assert.equal(groups[1].rows.find(row => row.label === 'RRF 融合分').value, '0.016')
  assert.match(groups[0].note, /已记录名次计算/)
  assert.equal(groups[2].rows[0].value, '0.3')
})
test('actual extraction stage and failed compression are shown without claiming success', () => {
  assert.equal(stageLabels.extractive_views, '摘录压缩视图')
  assert.equal(compressionStatus({ enabled: true, status: 'fallback_original' }), '压缩调用失败，已保留原始片段')
  assert.equal(compressionStatus({ enabled: false }), '本轮压缩未启用')
})
