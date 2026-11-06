import type { DocumentRow } from './types'

export type LibraryStatusFilter = 'all' | 'ready' | 'processing' | 'needs_attention'
const processingStates = new Set(['pending', 'queued', 'parsing', 'indexing', 'deleting'])

export function documentStatusGroup(status: string): Exclude<LibraryStatusFilter, 'all'> {
  if (status === 'ready') return 'ready'
  return processingStates.has(status) ? 'processing' : 'needs_attention'
}

export function matchesDocumentStatus(document: DocumentRow, filter: string): boolean {
  if (filter === 'all') return true
  if (filter === 'processing' || filter === 'needs_attention') return documentStatusGroup(document.status) === filter
  return document.status === filter
}
