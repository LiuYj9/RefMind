<script setup lang="ts">
import { ref, watch } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { request, post, showError } from '../api'
import { useWorkspace } from '../store'
import type { Task } from '../types'
const props = defineProps<{ docId: number; status: string; writable: boolean }>()
const ws = useWorkspace()
type Inspection = { protected: boolean; can_request: boolean; reason: string; expected_sha256: string | null }
const info = ref<Inspection | null>(null), loading = ref(false), busy = ref(false), error = ref('')
let generation = 0
async function inspect() {
  const token = ++generation, id = props.docId
  info.value = null; error.value = ''; loading.value = true
  try { const value = await request<Inspection>(`/documents/${id}/index-maintenance`); if (token === generation) info.value = value }
  catch (e) { if (token === generation) error.value = e instanceof Error ? e.message : '暂时无法检查索引状态' }
  finally { if (token === generation) loading.value = false }
}
async function submit() {
  const id = props.docId, digest = info.value?.expected_sha256
  if (!digest || !props.writable) return
  try {
    await ElMessageBox.confirm('申请将交由管理员在停止写入服务后执行，可能产生解析和嵌入费用。重建期间该文献暂停新问答，原文和历史引用保留。', '申请索引维护', { confirmButtonText: '提交维护申请', cancelButtonText: '暂不申请' })
    busy.value = true
    ws.track(await post<Task>(`/documents/${id}/reindex-requests`, { expected_sha256: digest }))
    ElMessage.success('申请已保存，等待管理员安排离线维护')
  } catch (e) { if (e !== 'cancel' && e !== 'close') showError(e) }
  finally { busy.value = false }
}
watch(() => [props.docId, props.status], inspect, { immediate: true })
</script>
<template>
  <section class="index-maintenance" aria-label="索引维护" v-loading="loading">
    <h4>索引与证据状态</h4>
    <p v-if="['reindexing', 'reindex_failed'].includes(status)" class="maintenance-paused">这篇文献的索引重建尚未完成，暂不能用于新问答。原文与历史引用仍可查看。</p>
    <p v-if="error" role="alert">{{ error }}</p>
    <p v-else-if="info">{{ info.reason }}</p>
    <template v-if="info && !info.protected">
      <p class="field-note">申请只会保存到任务中心，由管理员安排离线维护。页面刷新后仍可查看进度。</p>
      <el-button v-if="writable" :disabled="!info.can_request" :loading="busy" @click="submit">{{ ['reindexing', 'reindex_failed'].includes(status) ? '申请重试重建' : '申请索引维护' }}</el-button>
      <router-link to="/tasks" class="maintenance-tasks">查看维护申请</router-link>
    </template>
    <el-button v-if="error" text @click="inspect">重新检查</el-button>
  </section>
</template>
<style scoped>
.index-maintenance { margin-top: 20px; padding: 16px; border: 1px solid var(--el-border-color); border-radius: 12px; background: var(--el-fill-color-extra-light); }
.index-maintenance h4 { margin: 0 0 10px; }
.index-maintenance p { font-size: 13px; line-height: 1.7; overflow-wrap: anywhere; }
.maintenance-paused { color: var(--el-color-warning-dark-2); }
.maintenance-tasks { display: block; margin-top: 12px; color: var(--el-color-primary); font-size: 13px; }
</style>
