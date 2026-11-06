import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import { request, post, setCsrf, ApiError } from './api'
import { terminal } from './types'
import type { User, Library, Session, Task, RuntimeConfig, Paper } from './types'

export const useWorkspace = defineStore('workspace', () => {
  const user = ref<User | null>(null), initialized = ref(false)
  const libraries = ref<Library[]>([]), sessions = ref<Session[]>([]), tasks = ref<Task[]>([])
  const config = ref<RuntimeConfig | null>(null), selectedLibrary = ref<number | null>(null)
  const connectionError = ref('')
  const discoveryPapers = ref<Paper[]>([]), discoveryTaskId = ref(''), discoveryQuery = ref('')
  const abstractChoices = ref<{ search_id: string; paper_id: string; title: string }[]>([])
  const sessionAbstracts = ref<Record<number, { search_id: string; paper_id: string; title: string }[]>>({})
  const sessionAssetScopes = ref<Record<number, { selected: string[]; seen: string[] }>>({})
  function restoreReadingScope() {
    if (!user.value) return
    try {
      const stored = JSON.parse(sessionStorage.getItem(`refmind-scopes-${user.value.id}`) || '{}')
      if (stored && typeof stored === 'object' && !Array.isArray(stored)) { sessionAbstracts.value = stored.abstracts || stored; sessionAssetScopes.value = stored.assets || {} }
    } catch { sessionAbstracts.value = {} }
  }
  watch([sessionAbstracts, sessionAssetScopes], () => { if (user.value) { try { sessionStorage.setItem(`refmind-scopes-${user.value.id}`, JSON.stringify({ abstracts: sessionAbstracts.value, assets: sessionAssetScopes.value })) } catch {} } }, { deep: true })
  const activeTasks = computed(() => tasks.value.filter(t => !terminal(t.status)))
  let initializing: Promise<void> | null = null, poll: ReturnType<typeof setInterval> | null = null, polling = false
  let seen = new Set<string>()
  async function loadWorkspace() {
    const [libs, chats, cfg, jobs] = await Promise.all([request<Library[]>('/libraries'), request<Session[]>('/sessions'), request<RuntimeConfig>('/config'), request<Task[]>('/tasks')])
    libraries.value = libs; sessions.value = chats; config.value = cfg; tasks.value = jobs
    if (!libs.some(l => l.id === selectedLibrary.value)) selectedLibrary.value = libs[0]?.id ?? null
    seen = new Set(jobs.filter(t => terminal(t.status)).map(t => `${t.id}:${t.status}`))
  }
  async function initialize() {
    if (initialized.value) return
    if (initializing) return initializing
    initializing = (async () => {
      try { user.value = await request<User>('/auth/me'); setCsrf(user.value.csrf); restoreReadingScope(); await loadWorkspace(); startPolling() }
      catch (e) { clear(); if (!(e instanceof ApiError && e.status === 401)) connectionError.value = e instanceof Error ? e.message : '暂时无法连接工作空间' }
      finally { initialized.value = true; initializing = null }
    })()
    return initializing
  }
  async function login(username: string, password: string) {
    user.value = await post<User>('/auth/login', { username, password }); setCsrf(user.value.csrf); restoreReadingScope()
    await loadWorkspace(); connectionError.value = ''; initialized.value = true; startPolling()
  }
  function clear() {
    if (user.value) { try { sessionStorage.removeItem(`refmind-scopes-${user.value.id}`) } catch {} }
    user.value = null; libraries.value = []; sessions.value = []; tasks.value = []; config.value = null
    discoveryPapers.value = []; abstractChoices.value = []; sessionAbstracts.value = {}; sessionAssetScopes.value = {}; discoveryTaskId.value = ''; discoveryQuery.value = ''; selectedLibrary.value = null
    setCsrf(''); seen.clear(); if (poll) clearInterval(poll); poll = null
  }
  async function logout() { try { await post('/auth/logout') } finally { clear() } }
  function track(task: Task) { tasks.value = [task, ...tasks.value.filter(t => t.id !== task.id)]; return task }
  async function refreshTasks() {
    if (!user.value || polling) return
    polling = true
    try {
      const jobs = await request<Task[]>('/tasks'); tasks.value = jobs
      for (const task of jobs) if (terminal(task.status) && !seen.has(`${task.id}:${task.status}`)) {
        seen.add(`${task.id}:${task.status}`); window.dispatchEvent(new CustomEvent('refmind:task-complete', { detail: task }))
      }
    } finally { polling = false }
  }
  function startPolling() { if (poll) return; poll = setInterval(() => { if (!document.hidden && user.value) refreshTasks().catch(() => {}) }, 1000) }
  async function createSession(name: string, groupId?: number | null) {
    const session = await post<Session>('/sessions', { name, group_id: groupId ?? null }); sessions.value.unshift(session); return session
  }
  async function refreshLibraries() { libraries.value = await request<Library[]>('/libraries') }
  return { user, initialized, libraries, sessions, tasks, config, selectedLibrary, activeTasks, discoveryPapers, discoveryTaskId, discoveryQuery, abstractChoices, sessionAbstracts, sessionAssetScopes, connectionError,
    initialize, login, logout, clear, loadWorkspace, track, refreshTasks, createSession, refreshLibraries }
})
