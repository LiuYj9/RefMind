<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { CircleCheck, Warning, RefreshRight, Close, ArrowRight } from '@element-plus/icons-vue'
import { ElMessageBox, ElMessage } from 'element-plus'
import { useWorkspace } from '../store'
import { post, showError } from '../api'
import { terminal, taskLabels, statusLabels } from '../types'
import type { Task } from '../types'
const props = defineProps<{ task: Task; compact?: boolean }>()
const ws = useWorkspace(), router = useRouter(), busy = ref(false)
const readyForReview = computed(() => props.task.status === 'needs_review')
const asset = computed(() => props.task.result?.paper ? props.task.result : null)
const partial = computed(() => asset.value?.status === 'partial')
async function action(type: 'cancel' | 'retry') { busy.value = true; try { await post(`/tasks/${props.task.id}/${type}`); ElMessage.success(type === 'cancel' ? (props.task.kind === 'legacy_reindex' ? '维护申请已撤回' : '已请求取消，当前阶段结束后停止。') : props.task.kind === 'legacy_reindex' ? '已重新申请，等待管理员安排离线维护' : '任务已重新排队'); await ws.refreshTasks() } catch (e) { showError(e) } finally { busy.value = false } }
async function confirm() {
  try {
    await ElMessageBox.confirm(partial.value ? '将只在已成功解析的页面范围内进行研读，回答不会代表全文。是否继续？' : '请先打开原文，核对题名、作者与所选论文一致。确认后才会继续解析。', partial.value ? '仅研读可用页面' : '核对论文身份', { confirmButtonText: partial.value ? '研读已解析部分' : '原文身份一致，继续', cancelButtonText: '暂不确认', type: 'warning' })
    busy.value = true; ws.track(await post(`/reading-tasks/${props.task.id}/confirm`, { identity_confirmed: !partial.value, accept_partial: partial.value })); ElMessage.success('已记录你的选择')
  } catch (e) { if (e !== 'cancel' && e !== 'close') showError(e) } finally { busy.value = false }
}
function open() { if (props.task.kind === 'search') router.push({ path: '/discover', query: { search: props.task.id } }); else if (props.task.result?.session_id) router.push({ path: '/read', query: { session: props.task.result.session_id } }) }
</script>
<template><article :class="['task-card', { compact }]"><div class="task-card-heading"><span :class="['task-icon', { warning: readyForReview || task.status === 'failed', success: task.status === 'completed' }]"><el-icon><Warning v-if="readyForReview || task.status === 'failed'"/><CircleCheck v-else-if="task.status === 'completed'"/><RefreshRight v-else /></el-icon></span><div class="task-title"><strong>{{ task.result?.paper?.title || task.result?.title || taskLabels[task.kind] || task.kind }}</strong><span>{{ taskLabels[task.kind] }}<b>·</b>{{ task.stage || statusLabels[task.status] || task.status }}</span></div><el-tag :type="task.status === 'completed' ? 'success' : readyForReview || task.status === 'failed' ? 'warning' : 'info'" effect="plain" size="small">{{ statusLabels[task.status] || task.status }}</el-tag></div><el-progress v-if="!terminal(task.status)" :percentage="Math.min(100, Math.round((task.progress || 0) * 100))" :stroke-width="4" :show-text="false" /><p v-if="task.kind === 'legacy_reindex'" class="task-review-note">{{ task.status === 'completed' ? '重建已完成，原文与历史引用保留。' : '由管理员在停止写入服务后执行。申请不会自动启动解析或调用模型。' }}</p><p v-if="task.kind === 'legacy_reindex'" class="field-note">申请编号：{{ task.id }}</p><p v-if="task.error" class="inline-error">{{ task.error }}</p><p v-if="readyForReview" class="task-review-note">{{ partial ? '部分页面尚未提取到可用内容。可确认只研读已解析范围。' : '自动核验未能确认论文身份。请打开 PDF 核对后再继续。' }}</p><div v-if="asset?.coverage && readyForReview" class="coverage-note">页面覆盖：{{ asset.coverage.parsed_pages?.length || 0 }} / {{ asset.coverage.total_pages }} 页可用</div><div v-if="!compact" class="task-actions"><template v-if="readyForReview"><a :href="`/api/assets/${task.id}/pdf`" target="_blank" rel="noopener" class="text-action">打开原文核对</a><el-button size="small" type="primary" :loading="busy" @click="confirm">{{ partial ? '研读已解析部分' : '确认身份并继续' }}</el-button></template><el-button v-if="!terminal(task.status)" text :loading="busy" @click="action('cancel')"><el-icon><Close /></el-icon>{{ task.kind === 'legacy_reindex' ? '撤回申请' : '取消任务' }}</el-button><el-button v-if="['failed', 'interrupted', 'cancelled'].includes(task.status)" text :loading="busy" @click="action('retry')"><el-icon><RefreshRight /></el-icon>{{ task.kind === 'legacy_reindex' ? '重新申请维护' : '重试' }}</el-button><el-button v-if="task.result?.session_id || (task.kind === 'search' && task.result?.papers)" text @click="open">{{ task.kind === 'search' ? '查看检索结果' : '返回研读' }}<el-icon><ArrowRight /></el-icon></el-button></div></article></template>
