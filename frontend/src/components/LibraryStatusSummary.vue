<script setup lang="ts">
import { computed } from 'vue'
import { Collection, Reading, Loading, Warning } from '@element-plus/icons-vue'
import type { DocumentRow } from '../types'
import { documentStatusGroup, type LibraryStatusFilter } from '../libraryStatus'

const props = defineProps<{ documents: DocumentRow[]; selected: string; loading: boolean }>()
const emit = defineEmits<{ filter: [value: LibraryStatusFilter] }>()
const cards = computed(() => {
  const counts = { ready: 0, processing: 0, needs_attention: 0 }
  for (const document of props.documents) counts[documentStatusGroup(document.status)]++
  return [
    { value: 'all' as const, label: '全部文献', count: props.documents.length, detail: '查看本库所有论文', icon: Collection },
    { value: 'ready' as const, label: '可以研读', count: counts.ready, detail: '已完成解析与索引', icon: Reading },
    { value: 'processing' as const, label: '正在处理', count: counts.processing, detail: '进度可在任务中心查看', icon: Loading },
    { value: 'needs_attention' as const, label: '需要关注', count: counts.needs_attention, detail: '失败、中断或等待维护', icon: Warning },
  ]
})
</script>

<template>
  <section class="library-summary" aria-label="文献状态概览" :aria-busy="loading">
    <button v-for="card in cards" :key="card.value" type="button"
      :class="['summary-card', card.value, { selected: selected === card.value }]"
      :aria-pressed="selected === card.value" :disabled="loading" @click="emit('filter', card.value)">
      <span class="card-top"><el-icon><component :is="card.icon" /></el-icon><span>{{ card.label }}</span><span class="filter-hint">筛选</span></span>
      <strong>{{ loading ? '—' : card.count }}</strong>
      <small>{{ card.detail }}</small>
    </button>
  </section>
</template>

<style scoped>
.library-summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 0 0 24px; }
.summary-card { text-align: left; border: 1px solid var(--line); border-radius: 12px; background: #fff; color: var(--ink); padding: 17px 18px; transition: border-color .15s, background .15s; }
.summary-card:hover:not(:disabled) { border-color: var(--accent); }
.summary-card.selected { border-color: var(--accent); background: #f8f6ff; box-shadow: inset 0 0 0 1px var(--accent); }
.summary-card:disabled { cursor: wait; }
.card-top { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #656b7d; }
.card-top .el-icon { font-size: 17px; color: var(--accent); }
.ready .card-top .el-icon { color: #48765d; }
.needs_attention .card-top .el-icon { color: #946d32; }
.filter-hint { margin-left: auto; font-size: 10px; color: #79748c; }
.summary-card strong { display: block; font-size: 27px; line-height: 1.3; margin: 10px 0 4px; font-weight: 600; font-variant-numeric: tabular-nums; }
.summary-card small { font-size: 11px; color: #717689; }
@media(max-width:1000px) { .library-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; } }
@media(max-width:480px) { .summary-card { padding: 12px; } .card-top { font-size: 11px; gap: 5px; } .filter-hint { display: none; } .summary-card strong { font-size: 23px; } .summary-card small { font-size: 10px; } }
</style>
