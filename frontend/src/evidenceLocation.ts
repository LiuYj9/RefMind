import type { Evidence, SelectedSpan } from './types'

export interface TextPosition { item: number; offset: number; end: number }
export interface QuoteMatch { quote: string; start: TextPosition; end: TextPosition }
export interface QuoteSearch { matches: QuoteMatch[]; ambiguous: string[]; missing: string[] }

function decoded(value: unknown): unknown {
  if (typeof value !== 'string') return value
  try { return JSON.parse(value) } catch { return undefined }
}

/** Offsets use Python/Unicode code points, never JavaScript UTF-16 indices. */
export function selectedSpans(evidence?: Evidence | null): SelectedSpan[] {
  if (evidence?.metadata.evidence_view !== 'extractive_text') return []
  const raw = decoded(evidence.metadata.selected_spans)
  if (!Array.isArray(raw)) return []
  return raw.filter((span): span is SelectedSpan => !!span && Number.isInteger(span.start)
    && Number.isInteger(span.end) && span.start >= 0 && span.end > span.start
    && typeof span.quote === 'string' && !!span.quote.trim()
    && Array.from(span.quote).length === span.end - span.start)
}

export function isEvidencePage(evidence: Evidence | null | undefined, page: number): boolean {
  if (!evidence) return false
  const start = Number(evidence.metadata.page_start || evidence.metadata.page)
  const end = Number(evidence.metadata.page_end) || start
  return Number.isInteger(start) && start > 0 && start <= page && page <= end
}

export function sourceBBox(evidence: Evidence | null | undefined, page: number): number[] | undefined {
  const meta = evidence?.metadata
  if (!meta || !isEvidencePage(evidence, page)) return
  // A chunk-wide rectangle is only known for its source page, not every page in its range.
  if (Number(meta.page_start || meta.page) !== page) return
  const value = decoded(meta.bbox)
  if (Array.isArray(value) && value.length === 4 && value.every(Number.isFinite)
    && value[2] > value[0] && value[3] > value[1]) return value
}

function normalized(text: string): string {
  return Array.from(text).map(char => char.normalize('NFKC').toLowerCase()).join('').replace(/[\s\u00ad]+/gu, '')
}

/** Search full quotes; retain UTF-16 boundaries for accurate DOM Range highlighting. */
export function findQuoteMatches(items: string[], quotes: string[]): QuoteSearch {
  let joined = ''
  const positions: TextPosition[] = []
  items.forEach((text, item) => {
    let offset = 0
    for (const char of text) {
      const value = normalized(char)
      joined += value
      for (let i = 0; i < value.length; i++) positions.push({ item, offset, end: offset + char.length })
      offset += char.length
    }
  })
  const result: QuoteSearch = { matches: [], ambiguous: [], missing: [] }
  for (const quote of [...new Set(quotes)]) {
    const needle = normalized(quote)
    const start = needle.length ? joined.indexOf(needle) : -1
    if (start < 0) { result.missing.push(quote); continue }
    if (joined.indexOf(needle, start + 1) >= 0) { result.ambiguous.push(quote); continue }
    result.matches.push({ quote, start: positions[start]!, end: positions[start + needle.length - 1]! })
  }
  return result
}

function pageOnly(evidence: Evidence, page: number): Evidence {
  const metadata: Evidence['metadata'] = { ...evidence.metadata, page, page_start: page, page_end: page, citation_resolution: 'page_only' }
  for (const key of ['bbox', 'source_locations', 'selected_spans', 'evidence_view', 'chunk_id', 'paragraph_index', 'evidence_id']) delete metadata[key]
  return { text: '', metadata }
}

/** Never substitute a neighbouring chunk for a missing precise citation target. */
export function resolveCitation(href: string, sources: Evidence[], base: string): { evidence: Evidence; page: number; exact: boolean } | undefined {
  let url: URL
  try { url = new URL(href, base) } catch { return }
  const resource = url.searchParams.get('open_resource')
  const target = url.searchParams.get('open_citation')
  const chunk = url.searchParams.get('chunk_id')
  let page = Number(url.searchParams.get('page')) || 0
  let paragraph = Number(url.searchParams.get('paragraph') || url.searchParams.get('paragraph_index')) || 0
  let same: Evidence[] = []
  if (resource) same = sources.filter(item => String(item.metadata.asset_id || item.metadata.resource_id || '') === resource)
  else if (target) {
    const match = /^g(\d+)-d(\d+)-p(\d+)-r(\d+)$/.exec(target)
    if (!match) return
    page = Number(match[3]); paragraph = Number(match[4])
    same = sources.filter(item => Number(item.metadata.group_id) === Number(match[1]) && Number(item.metadata.doc_id) === Number(match[2]))
  }
  if (!same.length) return
  page ||= Number(same[0]!.metadata.page_start || same[0]!.metadata.page) || 1
  const onPage = same.filter(item => isEvidencePage(item, page))
  const exact = onPage.filter(item => chunk ? String(item.metadata.chunk_id || '') === chunk
    : !paragraph || Number(item.metadata.paragraph_index) === paragraph)
  if (exact.length === 1) return { evidence: exact[0]!, page, exact: true }
  return { evidence: pageOnly(onPage[0] || same[0]!, page), page, exact: false }
}
