<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Search, ArrowRight, Document, Reading, Link, Filter, Close, Select } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { useWorkspace } from '../store'
import { post, request, showError } from '../api'
import { terminal, safeLink } from '../types'
import type { Paper, Task } from '../types'
const ws = useWorkspace(), route = useRoute(), router = useRouter()
const query = ref(ws.discoveryQuery), provider = ref('auto'), searchLimit = ref(20), yearFrom = ref<number>(), yearTo = ref<number>(), oaOnly = ref(false), advanced = ref(false)
const selected = ref<Record<string, Paper>>({}), focused = ref(''), chosenLocations = ref<Record<string, number>>({}), requesting = ref(false), error = ref(''), nextCursor = ref<string | null>(null), warnings = ref<string[]>([]), append = ref(false), appliedTask = ref('')
const active = computed(() => ws.tasks.find(t => t.id === ws.discoveryTaskId))
const busy = computed(() => requesting.value || !!active.value && !terminal(active.value.status))
const providerLabels: Record<string, string> = { semantic_scholar: 'Semantic Scholar', openalex: 'OpenAlex', crossref: 'Crossref' }
const providerStates: Record<string, string> = { running: '检索中', completed: '已返回', failed: '暂不可用', timeout: '已超时', pending: '等待中' }
const detail = computed(() => ws.discoveryPapers.find(p => p.paper_id === focused.value) || ws.discoveryPapers[0])
const picked = computed(() => Object.values(selected.value))
const publicIndexes = (paper: Paper) => paper.locations.map((location, index) => ({ location, index })).filter(({ location }) => location.is_oa === true && !!safeLink(location.url))
const fulltextPapers = computed(() => picked.value.filter(p => publicIndexes(p).length > 0))
const abstractPapers = computed(() => picked.value.filter(p => !!p.abstract))
watch(detail, paper => { if (paper && chosenLocations.value[paper.paper_id] === undefined) { const first = publicIndexes(paper)[0]; if (first) chosenLocations.value[paper.paper_id] = first.index } }, { immediate: true })
function select(paper: Paper, value: unknown) { if (value) selected.value[paper.paper_id] = paper; else delete selected.value[paper.paper_id] }
async function search(cursor?: string) {
  if (!query.value.trim() || busy.value) return
  if (yearFrom.value && yearTo.value && yearFrom.value > yearTo.value) { error.value = '起始年份不能晚于截止年份'; return }
  requesting.value = true; error.value = ''; append.value = !!cursor
  try { const task = ws.track(await post<Task>('/academic-searches', { query: query.value.trim(), provider: provider.value, year_from: yearFrom.value ?? null, year_to: yearTo.value ?? null, open_access_only: oaOnly.value, cursor: cursor || null, limit: searchLimit.value })); ws.discoveryTaskId = task.id; ws.discoveryQuery = query.value; router.replace({ path: '/discover', query: { search: task.id } }); if (!cursor) { ws.discoveryPapers = []; selected.value = {}; focused.value = '' } }
  catch (e) { error.value = e instanceof Error ? e.message : '检索未完成' } finally { requesting.value = false }
}
function applyTask(task?: Task) {
  if (!task || (!terminal(task.status) && !task.result?.papers?.length)) return
  const revision = `${task.id}:${task.status}:${JSON.stringify(task.result)}`
  if (appliedTask.value === revision) return
  const firstApplication = !appliedTask.value.startsWith(`${task.id}:`)
  appliedTask.value = revision; const incoming = (task.result?.papers || []).map((p: Paper) => ({ ...p, _search_id: task.id }))
  if (append.value) { const merged = new Map(ws.discoveryPapers.map(p => [p.paper_id, p])); for (const paper of incoming) merged.set(paper.paper_id, paper); ws.discoveryPapers = [...merged.values()] } else ws.discoveryPapers = incoming
  if (firstApplication) {
    if (task.result?.query) { query.value = task.result.query; ws.discoveryQuery = task.result.query }
    if (task.result?.provider) provider.value = task.result.provider
    if (Number.isInteger(task.result?.limit) && task.result.limit >= 1 && task.result.limit <= 50) searchLimit.value = task.result.limit
    if (task.result?.filters) { yearFrom.value = task.result.filters.year_from ?? undefined; yearTo.value = task.result.filters.year_to ?? undefined; oaOnly.value = !!task.result.filters.open_access_only }
  }; nextCursor.value = task.result?.next_cursor || null; warnings.value = task.result?.warnings || []; error.value = task.error || ''; focused.value ||= ws.discoveryPapers[0]?.paper_id || ''
}
watch(active, task => applyTask(task), { deep: true, immediate: true })
watch(() => route.query.search, async id => { if (typeof id !== 'string' || id === ws.discoveryTaskId) return; try { const task = ws.track(await request<Task>(`/academic-searches/${id}`)); ws.discoveryTaskId = id; append.value = false; applyTask(task) } catch (e) { showError(e) } }, { immediate: true })
async function study(fulltext: boolean) {
  const papers = fulltext ? fulltextPapers.value : abstractPapers.value
  if (!papers.length || requesting.value) return
  if (fulltext && papers.length > Math.max(0, 10 - ws.activeTasks.length)) { ElMessage.warning(`当前还能创建 ${Math.max(0, 10 - ws.activeTasks.length)} 项任务，请减少所选文献或等待已有任务完成。`); return }
  requesting.value = true
  try {
    const session = await ws.createSession(papers.length === 1 ? papers[0]!.title.slice(0, 100) : `${papers.length} 篇文献研读`)
    const tasks: string[] = []
    if (fulltext) {
      for (const paper of papers) {
        const index = chosenLocations.value[paper.paper_id] ?? publicIndexes(paper)[0]!.index
        const task = ws.track(await post<Task>('/reading-tasks', { search_id: paper._search_id, paper_id: paper.paper_id, session_id: session.id, location_index: index })); tasks.push(task.id)
      }
    } else ws.sessionAbstracts[session.id] = papers.map(p => ({ search_id: p._search_id!, paper_id: p.paper_id, title: p.title }))
    router.push({ path: '/read', query: { session: session.id, mode: 'external', ...(tasks.length ? { tasks: tasks.join(',') } : {}) } })
  } catch (e) { showError(e) } finally { requesting.value = false }
}
</script>
<template><div class="discovery-page"><header class="page-heading"><div><div class="eyebrow">LITERATURE DISCOVERY</div><h1>发现下一篇值得读的论文</h1><p>从学术索引找到文献，选择后再获取公开全文。</p></div><span class="subtle-badge"><span class="tiny-dot"></span>选择由你，证据可溯</span></header><section class="search-panel"><form class="discovery-search" @submit.prevent="search()"><el-icon><Search /></el-icon><input v-model="query" aria-label="文献检索问题" placeholder="输入研究问题、论文题名或 DOI…" maxlength="500" /><el-button type="primary" native-type="submit" :loading="busy" :disabled="!query.trim() || !ws.config?.external_search_enabled">检索文献<el-icon><ArrowRight /></el-icon></el-button></form><div class="search-options"><el-select v-model="provider" aria-label="学术来源" class="provider-select"><el-option label="全部学术来源" value="auto"/><el-option label="Semantic Scholar" value="semantic_scholar"/><el-option label="OpenAlex" value="openalex"/><el-option label="Crossref" value="crossref"/></el-select><el-checkbox v-model="oaOnly">仅公开全文候选</el-checkbox><el-button text @click="advanced = !advanced"><el-icon><Filter /></el-icon>更多筛选</el-button><span class="search-boundary">检索不会自动下载或加入文献库</span></div><div v-if="advanced" class="advanced-filters"><label>发表年份</label><el-input-number v-model="yearFrom" :min="1600" :max="2100" :controls="false" placeholder="起始年份" aria-label="起始年份"/><span>—</span><el-input-number v-model="yearTo" :min="1600" :max="2100" :controls="false" placeholder="截止年份" aria-label="截止年份"/><span class="muted-text">来源不支持的条件会在结果中说明。</span></div></section><el-alert v-if="!ws.config?.external_search_enabled" title="外部检索暂未开启，请联系管理员。仍可研读已有文献。" type="info" :closable="false"/><el-alert v-if="error" :title="error" type="error" show-icon :closable="false"/><el-alert v-for="warning in warnings" :key="warning" :title="warning" type="warning" :closable="false" show-icon/>
<div v-if="active?.result?.provider_status" class="provider-progress" aria-live="polite"><span v-for="(state, source) in active.result.provider_status" :key="source" :class="['provider-state', String(state)]"><span class="tiny-dot"></span>{{ providerLabels[String(source)] || source }}<small>{{ providerStates[String(state)] || state }}</small></span></div><div v-if="busy && !ws.discoveryPapers.length" class="search-loading"><el-skeleton :rows="7" animated /><p>正在检索题录与摘要，全文将在你选择后获取。</p></div><div v-else-if="!ws.discoveryPapers.length" class="discovery-empty"><div class="empty-search-symbol"><el-icon><Search /></el-icon></div><h2>{{ active && terminal(active.status) ? '没有找到符合条件的文献' : '从一个研究问题开始' }}</h2><p>{{ active && terminal(active.status) ? '尝试更换关键词、放宽年份范围或切换学术来源。' : '试着描述你关心的对象、方法或实验条件。' }}</p><div class="discovery-principles"><span><el-icon><Select /></el-icon>多源检索</span><span><el-icon><Document /></el-icon>按篇选文</span><span><el-icon><Reading /></el-icon>原文研读</span></div></div>
<div v-else class="discovery-results"><section class="paper-list"><div class="section-toolbar"><span><strong>{{ ws.discoveryPapers.length }}</strong> 篇候选文献</span><span class="muted-text">按来源相关性排序</span></div><article v-for="paper in ws.discoveryPapers" :key="paper.paper_id" :class="['paper-result', { focused: detail?.paper_id === paper.paper_id, picked: selected[paper.paper_id] }]" @click="focused = paper.paper_id"><el-checkbox :model-value="!!selected[paper.paper_id]" :aria-label="`选择 ${paper.title}`" @click.stop @change="select(paper, $event)"/><div class="paper-result-content"><div class="paper-meta"><span>{{ paper.year || '年份未知' }}</span><span>{{ paper.venue || paper.providers.join(' · ') }}</span></div><h3>{{ paper.title }}</h3><p>{{ paper.authors.slice(0, 4).join(' · ') }}{{ paper.authors.length > 4 ? ' 等' : '' }}</p><div class="paper-tags"><span :class="['evidence-badge', paper.abstract ? 'abstract' : 'metadata']">{{ paper.abstract ? '有摘要' : '仅题录' }}</span><span v-if="publicIndexes(paper).length" class="availability-badge">公开 PDF 候选</span><span v-if="paper.is_retracted" class="retracted-badge">已撤稿</span></div></div></article><el-button v-if="nextCursor" class="load-more" :loading="busy" @click="search(nextCursor!)">加载更多结果</el-button></section><aside class="paper-detail" v-if="detail"><div class="detail-topline">文献详情<span v-for="source in detail.providers" :key="source" class="source-label">{{ source }}</span></div><h2>{{ detail.title }}</h2><p class="detail-authors">{{ detail.authors.join(' · ') || '作者信息暂缺' }}</p><div class="detail-meta">{{ detail.year || '年份未知' }}<span v-if="detail.venue"> · {{ detail.venue }}</span></div><a v-if="detail.doi" :href="safeLink(`https://doi.org/${detail.doi}`)" target="_blank" rel="noopener noreferrer" class="doi-link"><el-icon><Link /></el-icon>{{ detail.doi }}</a><div class="detail-divider"></div><h4>摘要<span class="muted-text">ABSTRACT</span></h4><p class="abstract-text">{{ detail.abstract || '该索引没有提供摘要。题录可用于发现文献，不能作为学术结论的回答证据。' }}</p><section class="fulltext-options"><h4>公开全文<span class="muted-text">SOURCE</span></h4><template v-if="publicIndexes(detail).length"><el-radio-group v-model="chosenLocations[detail.paper_id]" class="location-options"><el-radio v-for="entry in publicIndexes(detail)" :key="entry.index" :value="entry.index"><span>{{ entry.location.provider || '公开来源' }} · {{ entry.location.version || '版本未知' }}</span><small>{{ entry.location.license || '未提供许可信息' }}</small></el-radio></el-radio-group><p class="field-note">候选链接尚未核验。获取后会检查文件身份和解析范围。</p></template><p v-else class="field-note">暂无可直接获取的公开 PDF。可查看原站，或在文献库上传已有 PDF。</p><a v-for="(location, i) in detail.locations.filter(l => safeLink(l.landing_url)) .slice(0, 2)" :key="i" :href="safeLink(location.landing_url)" target="_blank" rel="noopener noreferrer" class="text-action">查看原始来源<el-icon><ArrowRight /></el-icon></a></section></aside></div>
<div v-if="picked.length" class="selection-bar"><div><span class="selected-count">{{ picked.length }}</span><strong>篇文献已选择</strong><small>{{ fulltextPapers.length }} 篇有公开 PDF 候选 · {{ abstractPapers.length }} 篇有摘要</small><el-button text @click="selected = {}">清空</el-button></div><div><el-button :disabled="!abstractPapers.length || requesting" @click="study(false)">仅按摘要研读</el-button><el-button type="primary" :loading="requesting" :disabled="!fulltextPapers.length" @click="study(true)">获取 {{ fulltextPapers.length }} 篇全文并研读<el-icon><ArrowRight /></el-icon></el-button></div></div></div></template>
