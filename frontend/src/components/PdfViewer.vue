<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { ArrowLeft, ArrowRight, ZoomIn, ZoomOut, Document, Download } from '@element-plus/icons-vue'
import * as pdfjs from 'pdfjs-dist'
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import type { Evidence } from '../types'
import { findQuoteMatches, isEvidencePage, selectedSpans, sourceBBox } from '../evidenceLocation'
import type { QuoteMatch } from '../evidenceLocation'
pdfjs.GlobalWorkerOptions.workerSrc = workerUrl
const props = defineProps<{ url?: string; title?: string; page?: number; evidence?: Evidence | null }>()
const emit = defineEmits<{ page: [page: number] }>()
const canvas = ref<HTMLCanvasElement>(), textHost = ref<HTMLDivElement>(), scrollHost = ref<HTMLDivElement>()
const current = ref(1), pages = ref(0), zoom = ref(1), loading = ref(false), error = ref('')
const location = ref<'none' | 'excerpt' | 'source' | 'page'>('none'), matchedQuotes = ref(0), quoteCount = ref(0)
const located = computed(() => location.value === 'excerpt' || location.value === 'source')
const locationLabel = computed(() => location.value === 'excerpt'
  ? `已匹配摘录${quoteCount.value > 1 ? `（${matchedQuotes.value}/${quoteCount.value} 段）` : ''}`
  : location.value === 'source' ? '已定位来源区域' : props.evidence ? '仅定位到页' : '原文可选中、复制与缩放')
const rendering = ref(false)
const dimensions = ref({ width: 0, height: 0 }), boxes = ref<{ left: number; top: number; width: number; height: number }[]>([])
let pdf: any = null, loadingTask: any = null, renderTask: any = null, textTask: any = null, generation = 0, renderRevision = 0, observer: ResizeObserver | null = null, resizeTimer: ReturnType<typeof setTimeout>
const pageStyle = computed(() => ({ width: `${dimensions.value.width}px`, height: `${dimensions.value.height}px` }))
async function load() {
  const token = ++generation
  renderTask?.cancel(); textTask?.cancel(); await loadingTask?.destroy().catch(() => {})
  pdf = null; pages.value = 0; error.value = ''; boxes.value = []; location.value = 'none'
  if (!props.url) return
  loading.value = true
  try {
    loadingTask = pdfjs.getDocument({ url: props.url, withCredentials: true })
    const loaded = await loadingTask.promise
    if (token !== generation) { await loaded.destroy(); return }
    pdf = loaded; pages.value = pdf.numPages; current.value = Math.min(Math.max(props.page || 1, 1), pages.value)
    await nextTick(); await render()
  } catch (e) { if (token === generation) error.value = '原文暂时不可用，可能已过期或访问权限发生变化。' }
  finally { if (token === generation) loading.value = false }
}
async function render() {
  if (!pdf || !canvas.value || !textHost.value || !scrollHost.value) return
  const token = generation, revision = ++renderRevision
  rendering.value = true
  renderTask?.cancel(); textTask?.cancel(); location.value = props.evidence ? 'page' : 'none'; boxes.value = []; matchedQuotes.value = 0; quoteCount.value = 0
  try {
    const page = await pdf.getPage(current.value)
    if (revision !== renderRevision || token !== generation) return
    const natural = page.getViewport({ scale: 1 })
    const surface = getComputedStyle(scrollHost.value)
    const width = Math.max(160, scrollHost.value.clientWidth - parseFloat(surface.paddingLeft) - parseFloat(surface.paddingRight) - 2)
    const viewport = page.getViewport({ scale: Math.min(width / natural.width, 1.6) * zoom.value })
    dimensions.value = { width: viewport.width, height: viewport.height }
    const ratio = window.devicePixelRatio || 1
    canvas.value.width = Math.ceil(viewport.width * ratio); canvas.value.height = Math.ceil(viewport.height * ratio)
    canvas.value.style.width = `${viewport.width}px`; canvas.value.style.height = `${viewport.height}px`
    renderTask = page.render({ canvas: canvas.value, canvasContext: canvas.value.getContext('2d')!, viewport, transform: ratio === 1 ? undefined : [ratio, 0, 0, ratio, 0, 0] })
    await renderTask.promise
    if (token !== generation || revision !== renderRevision) return
    textHost.value.replaceChildren(); textHost.value.style.setProperty('--scale-factor', String(viewport.scale)); textHost.value.style.setProperty('--total-scale-factor', String(viewport.scale))
    textTask = new pdfjs.TextLayer({ textContentSource: await page.getTextContent(), container: textHost.value, viewport })
    await textTask.render()
    if (token !== generation || revision !== renderRevision) return
    const meta = props.evidence?.metadata || {}
    const useSourceRegion = () => {
      const bbox = sourceBBox(props.evidence, current.value)
      if (!bbox) return false
      let coords: number[]
      if (meta.bbox_coordinate_system === 'normalized_1000') coords = [bbox[0] * viewport.width / 1000, bbox[1] * viewport.height / 1000, bbox[2] * viewport.width / 1000, bbox[3] * viewport.height / 1000]
      else coords = viewport.convertToViewportRectangle([page.view[0] + bbox[0], page.view[3] - bbox[1], page.view[0] + bbox[2], page.view[3] - bbox[3]])
      const [x0, y0, x1, y1] = coords
      const left = Math.max(0, Math.min(x0!, x1!)), top = Math.max(0, Math.min(y0!, y1!))
      const right = Math.min(viewport.width, Math.max(x0!, x1!)), bottom = Math.min(viewport.height, Math.max(y0!, y1!))
      if (right <= left || bottom <= top) return false
      boxes.value = [{ left, top, width: right - left, height: bottom - top }]; location.value = 'source'; return true
    }
    const matchQuotes = (quotes: string[]) => {
      if (!textHost.value || !isEvidencePage(props.evidence, current.value)) return false
      const spans = Array.from(textHost.value.querySelectorAll('span')).filter(span => span.textContent?.trim() && !span.querySelector('span'))
      const found = findQuoteMatches(spans.map(span => span.textContent || ''), quotes)
      const resolved = found.matches.map(match => quoteRectangles(match, spans, textHost.value!)).filter(rectangles => rectangles.length)
      const rectangles = resolved.flat()
      if (!rectangles.length) return false
      boxes.value = rectangles; matchedQuotes.value = resolved.length; location.value = 'excerpt'; return true
    }
    if (meta.evidence_view === 'extractive_text') {
      const quotes = selectedSpans(props.evidence).map(span => span.quote)
      quoteCount.value = new Set(quotes).size
      // Compression offsets are chunk coordinates, not positions in the PDF text layer.
      // Search each selected quote separately before considering the original coarse rectangle.
      if (!matchQuotes(quotes)) useSourceRegion()
    } else if (!useSourceRegion() && props.evidence?.text) {
      const quote = Array.from(props.evidence.text).slice(0, 90).join('')
      if (quote.replace(/\s/g, '').length >= 12) { quoteCount.value = 1; matchQuotes([quote]) }
    }
    scrollHost.value.scrollTop = Math.max(0, (boxes.value[0]?.top || 0) - 90)
  } catch (e: any) { if (!['RenderingCancelledException', 'AbortException'].includes(e?.name) && token === generation && revision === renderRevision) error.value = '这一页未能正常渲染，请重试或下载原文。' } finally { if (revision === renderRevision) rendering.value = false }
}
function quoteRectangles(match: QuoteMatch, spans: HTMLElement[], host: HTMLElement) {
  function point(element: HTMLElement | undefined, offset: number): { node: Text; offset: number } | undefined {
    if (!element) return
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT)
    let node: Node | null, remaining = offset
    while ((node = walker.nextNode())) {
      if (remaining <= (node.textContent?.length || 0)) return { node: node as Text, offset: remaining }
      remaining -= node.textContent?.length || 0
    }
  }
  const start = point(spans[match.start.item], match.start.offset), end = point(spans[match.end.item], match.end.end)
  if (!start || !end) return []
  const range = document.createRange(); range.setStart(start.node, start.offset); range.setEnd(end.node, end.offset)
  const origin = host.getBoundingClientRect()
  return Array.from(range.getClientRects()).filter(rect => rect.width > 0 && rect.height > 0).map(rect => ({
    left: rect.left - origin.left, top: rect.top - origin.top, width: rect.width, height: rect.height,
  }))
}
function changePage(value: number) { current.value = Math.min(Math.max(value, 1), pages.value); emit('page', current.value); render() }
watch(() => props.url, load)
watch(() => props.page, value => { if (value && pdf && value !== current.value) changePage(value) })
watch(() => props.evidence, () => render())
watch(zoom, () => render())
onMounted(() => { load(); observer = new ResizeObserver(() => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 150) }); if (scrollHost.value) observer.observe(scrollHost.value) })
onBeforeUnmount(() => { generation++; clearTimeout(resizeTimer); observer?.disconnect(); renderTask?.cancel(); textTask?.cancel(); loadingTask?.destroy().catch(() => {}) })
</script>
<template>
  <section class="pdf-viewer">
    <div class="pdf-toolbar"><div class="pdf-title"><el-icon><Document /></el-icon><span :title="title">{{ title || '原文阅读器' }}</span></div><a v-if="url" :href="url" target="_blank" rel="noopener" aria-label="打开原文" class="icon-link"><el-icon><Download /></el-icon></a></div>
    <div v-if="url" class="pdf-controls"><div><el-button text circle :disabled="current <= 1" aria-label="上一页" @click="changePage(current - 1)"><el-icon><ArrowLeft /></el-icon></el-button><span class="page-number">{{ current }} <span>/ {{ pages || '—' }}</span></span><el-button text circle :disabled="current >= pages" aria-label="下一页" @click="changePage(current + 1)"><el-icon><ArrowRight /></el-icon></el-button></div><div><el-button text circle aria-label="缩小" :disabled="zoom <= .6" @click="zoom -= .1"><el-icon><ZoomOut /></el-icon></el-button><span class="zoom-label">{{ Math.round(zoom * 100) }}%</span><el-button text circle aria-label="放大" :disabled="zoom >= 4" @click="zoom += .1"><el-icon><ZoomIn /></el-icon></el-button></div></div>
    <div ref="scrollHost" class="pdf-scroll" v-loading="loading"><div v-if="error" class="pdf-empty"><el-icon><Document /></el-icon><h3>原文未能打开</h3><p>{{ error }}</p><el-button @click="load">重新加载</el-button></div><div v-else-if="!url" class="pdf-empty"><div class="paper-illustration"><i></i><i></i><i></i><i></i></div><h3>让每个结论，都有出处</h3><p>选择一篇文献，或点击回答中的引用。<br>对应原文将在这里打开。</p></div><div v-show="url && !error" class="pdf-page" :style="pageStyle"><canvas ref="canvas"></canvas><div ref="textHost" class="textLayer"></div><div v-for="(box, i) in boxes" :key="i" class="evidence-box" :style="{ left: `${box.left}px`, top: `${box.top}px`, width: `${box.width}px`, height: `${box.height}px` }"></div></div></div>
    <div v-if="url && !error" class="pdf-location-status" role="status" aria-live="polite"><span :class="['tiny-dot', { muted: !located }]"></span>{{ rendering ? '正在定位页面与证据…' : locationLabel }}</div>
  </section>
</template>
