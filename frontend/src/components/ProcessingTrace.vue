<script setup lang="ts">
import { computed } from 'vue'
import type { AnswerResult, Evidence } from '../types'
import { budgetRows, compressionStatus, documentEvidence, evidenceLabel, finite, metric, scoreGroups, stageLabels, timingLabels } from '../tracePresentation'
const props = defineProps<{ result?: AnswerResult }>()
const emit = defineEmits<{ evidence: [evidence: Evidence] }>()
const trace = computed(() => props.result?.trace)
const stages = computed(() => Object.entries(stageLabels).flatMap(([key, label]) => {
  const stage = trace.value?.stages?.[key]
  if (!stage || typeof stage !== 'object') return []
  const documents = Array.isArray(stage.documents) ? stage.documents.map(documentEvidence).filter((item): item is Evidence => !!item) : []
  return [{ key, label, stage, documents }]
}))
const timings = computed(() => Object.entries(timingLabels).flatMap(([key, label]) => finite(trace.value?.timings?.[key]) ? [{ label, value: `${metric(trace.value!.timings![key])} 秒` }] : []))
const budgets = computed(() => budgetRows(trace.value?.packing))
const queries = computed(() => Array.isArray(trace.value?.queries) ? trace.value!.queries!.filter(query => query && typeof query.text === 'string') : [])
const records = computed(() => Array.isArray(trace.value?.compression?.records) ? trace.value!.compression!.records!.filter(record => record && typeof record === 'object') : [])
const scopeCount = computed(() => {
  const ids = trace.value?.scope?.retrieval_doc_ids
  return Array.isArray(ids) ? ids.length : undefined
})
function excerpt(text: string) { const chars = Array.from(text); return chars.slice(0, 160).join('') + (chars.length > 160 ? '…' : '') }
</script>
<template>
  <details class="processing-trace">
    <summary>本轮检索与证据处理<span>查看处理记录</span></summary>
    <div class="trace-body">
      <p class="trace-note">仅展示本轮已保存的记录。名次、距离、融合分和重排分含义不同，不能互相替代。</p>
      <p v-if="!trace" class="trace-missing">本轮未保存处理记录，无法补推检索名次或处理分数。</p>
      <template v-else>
        <div class="trace-overview"><span v-if="trace.pipeline">处理路径：{{ trace.pipeline }}</span><span v-if="scopeCount !== undefined">检索范围：{{ scopeCount }} 篇文献</span></div>
        <section class="trace-section"><h4>实际查询</h4><ol v-if="queries.length" class="trace-queries"><li v-for="(query, index) in queries" :key="query.query_id || index"><span>{{ query.text }}</span><small v-if="query.status">{{ query.status }}</small></li></ol><p v-else class="trace-missing">未记录查询展开。</p></section>
        <section class="trace-section"><h4>候选与证据</h4><p v-if="!stages.length" class="trace-missing">未记录各阶段的证据快照。</p>
          <details v-for="item in stages" :key="item.key" class="trace-stage">
            <summary>{{ item.label }}<span>{{ finite(item.stage.count) ? `${item.stage.count} 条` : item.stage.documents ? `${item.documents.length} 条快照` : '数量未记录' }}</span></summary>
            <p v-if="item.stage.status || item.stage.reason || item.stage.invoked === false" class="trace-stage-note">{{ item.stage.invoked === false ? '本轮未调用。' : '' }}{{ item.stage.status }} {{ item.stage.reason }}</p>
            <p v-if="item.stage.score_type" class="trace-stage-note">分数类型：{{ item.stage.score_type }}</p>
            <p v-if="!item.documents.length" class="trace-missing">此阶段没有可打开的证据快照。</p>
            <article v-for="(document, index) in item.documents" :key="`${document.metadata.chunk_id || document.metadata.evidence_id || ''}-${index}`" class="trace-document">
              <button class="trace-source" type="button" @click="emit('evidence', document)">{{ evidenceLabel(document) }}<span>查看来源 ↗</span></button>
              <p class="trace-identity"><span v-if="document.metadata.page_start || document.metadata.page">PDF 第 {{ document.metadata.page_start || document.metadata.page }} 页</span><span v-if="document.metadata.chunk_id">片段 {{ document.metadata.chunk_id }}</span></p>
              <p class="trace-excerpt">{{ excerpt(document.text) }}</p>
              <div v-for="(group, groupIndex) in scoreGroups(document)" :key="groupIndex"><p class="trace-stage-note">{{ group.label }}</p><p v-if="group.note" class="trace-missing">{{ group.note }}</p><dl class="trace-scores"><div v-for="row in group.rows" :key="row.label"><dt>{{ row.label }}</dt><dd>{{ row.value }}</dd></div></dl></div>
              <p v-if="document.metadata.rerank_backend" class="trace-stage-note">重排方式：{{ document.metadata.rerank_backend }}<strong v-if="document.metadata.rerank_degraded === true"> · 降级路径</strong></p>
              <p v-if="document.metadata.rerank_reason" class="trace-stage-note">{{ document.metadata.rerank_reason }}</p>
            </article>
          </details>
        </section>
        <section class="trace-section"><h4>压缩与预算</h4><p v-if="!trace.compression" class="trace-missing">未记录压缩明细。</p><p v-else class="trace-stage-note">{{ compressionStatus(trace.compression) }}<span v-if="trace.compression.scorer"> · {{ trace.compression.scorer }}</span></p>
          <p v-if="trace.compression?.reason" class="trace-stage-note">原因：{{ trace.compression.reason }}</p>
          <ul v-if="records.length" class="trace-compression"><li v-for="(record, index) in records" :key="`${record.chunk_id || record.identity?.chunk_id || ''}-${index}`"><span>{{ record.chunk_id || record.identity?.chunk_id || '片段身份未记录' }}</span><strong>{{ record.status || '状态未记录' }}</strong><p>{{ record.reason || '原因未记录' }}</p></li></ul>
          <dl v-if="budgets.length" class="trace-scores"><div v-for="row in budgets" :key="row.label"><dt>{{ row.label }}</dt><dd>{{ row.value }}</dd></div></dl><p v-else class="trace-missing">未记录可展示的预算数值。</p>
        </section>
        <dl v-if="timings.length" class="trace-scores trace-timings"><div v-for="row in timings" :key="row.label"><dt>{{ row.label }}</dt><dd>{{ row.value }}</dd></div></dl>
      </template>
    </div>
  </details>
</template>
<style scoped>
.processing-trace{margin:14px 0;border:1px solid #e7e9f0;border-radius:8px;background:#fafbfe;font-size:12px}.processing-trace>summary{padding:10px 12px;font-weight:500;cursor:pointer;color:#4f566b}.processing-trace>summary>span{float:right;font-size:11px;color:#8a90a1}.trace-body{padding:0 12px 12px}.trace-note,.trace-missing{font-size:11px;color:#7c8496;margin:8px 0;line-height:1.7}.trace-overview{display:flex;flex-wrap:wrap;gap:8px 18px;color:#697287;font-size:11px}.trace-section{margin-top:15px}.trace-section h4{font-size:12px;margin:0 0 6px}.trace-queries{padding-left:20px;margin:0;word-break:break-word}.trace-queries li{margin:5px 0}.trace-queries small{margin-left:8px;color:#8a90a1}.trace-stage{border-top:1px solid #e7e9f0}.trace-stage>summary{padding:8px 0;cursor:pointer;color:#616b80}.trace-stage>summary>span{float:right;color:#9299a8;font-size:11px}.trace-stage-note{font-size:11px;line-height:1.65;margin:5px 0;overflow-wrap:anywhere}.trace-stage-note strong{color:#9b6f27;font-weight:500}.trace-document{padding:9px 10px;margin:5px 0 10px;background:white;border:1px solid #eaecf2;border-radius:6px}.trace-source{display:flex;width:100%;align-items:flex-start;gap:10px;justify-content:space-between;border:0;background:transparent;padding:0;text-align:left;color:#6057b9;font-size:12px;overflow-wrap:anywhere}.trace-source>span{white-space:nowrap;font-size:10px;color:#8b85ba}.trace-identity{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:10px;color:#939aaa;margin:5px 0;overflow-wrap:anywhere}.trace-excerpt{font-size:11px;line-height:1.65;margin:6px 0;color:#70788b;white-space:pre-wrap;overflow-wrap:anywhere}.trace-scores{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 14px;margin:8px 0 0;font-size:10px}.trace-scores>div{display:flex;gap:8px;justify-content:space-between}.trace-scores dt{color:#8d95a4}.trace-scores dd{margin:0;color:#5f687b;font-variant-numeric:tabular-nums;text-align:right;overflow-wrap:anywhere}.trace-compression{padding-left:16px;font-size:11px;overflow-wrap:anywhere}.trace-compression li{margin:7px 0}.trace-compression strong{margin-left:7px;font-weight:500;color:#677188}.trace-compression p{margin:3px 0}.trace-timings{border-top:1px solid #e7e9f0;padding-top:9px}@media(max-width:400px){.processing-trace>summary>span{float:none;display:block;margin-left:16px}.trace-scores{grid-template-columns:1fr}}
</style>
