<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ChatDotRound, Search, Collection, List, Setting, Plus, ArrowRight, SwitchButton } from '@element-plus/icons-vue'
import { useWorkspace } from './store'
import { showError } from './api'
const route = useRoute(), router = useRouter(), ws = useWorkspace()
const nav = [{ path: '/read', name: '研读空间', icon: ChatDotRound }, { path: '/discover', name: '文献发现', icon: Search }, { path: '/library', name: '文献库', icon: Collection }, { path: '/tasks', name: '任务中心', icon: List }]
const library = computed(() => ws.libraries.find(l => l.id === ws.selectedLibrary))
async function newReading() { try { const session = await ws.createSession('新研读', ws.selectedLibrary); router.push({ path: '/read', query: { session: session.id } }) } catch (e) { showError(e) } }
async function logout() { try { await ws.logout(); router.replace('/login') } catch (e) { showError(e) } }
</script>
<template>
  <router-view v-if="route.path === '/login'" />
  <div v-else class="app-frame">
    <aside class="sidebar">
      <router-link class="brand" to="/read" aria-label="RefMind 研读空间"><span class="brand-mark"><i></i><i></i><i></i></span><span>RefMind<small>RESEARCH WORKSPACE</small></span></router-link>
      <el-button class="new-reading" @click="newReading"><el-icon><Plus /></el-icon>新建研读</el-button>
      <div class="nav-caption">工作空间</div>
      <nav aria-label="主导航"><router-link v-for="item in nav" :key="item.path" :to="item.path" :aria-label="item.name" :title="item.name" :class="['nav-item', { active: route.path === item.path }]"><el-icon><component :is="item.icon" /></el-icon><span>{{ item.name }}</span><span v-if="item.path === '/tasks' && ws.activeTasks.length" class="count">{{ ws.activeTasks.length }}</span></router-link></nav>
      <div class="sidebar-divider"></div><div class="nav-caption">最近研读 <span>{{ ws.sessions.length }}</span></div>
      <div class="recent-sessions"><router-link v-for="session in ws.sessions.slice(0, 12)" :key="session.id" :to="{ path: '/read', query: { session: session.id } }" :class="['session-link', { selected: route.path === '/read' && String(route.query.session) === String(session.id) }]"><span class="session-dot"></span><span>{{ session.name }}</span></router-link><p v-if="!ws.sessions.length" class="sidebar-empty">第一场研读，从一个问题开始。</p></div>
      <div class="sidebar-bottom"><router-link v-if="ws.user?.is_admin" to="/admin" aria-label="管理设置" title="管理设置" :class="['nav-item', { active: route.path === '/admin' }]"><el-icon><Setting /></el-icon>管理设置</router-link><div class="user-bar"><span class="avatar">{{ ws.user?.username.slice(0, 1).toUpperCase() }}</span><div><strong>{{ ws.user?.username }}</strong><small>{{ ws.user?.is_admin ? '工作空间管理员' : '课题组成员' }}</small></div><el-button text circle aria-label="退出登录" @click="logout"><el-icon><SwitchButton /></el-icon></el-button></div></div>
    </aside>
    <div class="workspace-main"><header class="topbar"><div class="breadcrumbs"><span>工作空间</span><el-icon><ArrowRight /></el-icon><strong>{{ route.meta.title }}</strong></div><div class="topbar-right"><el-button class="mobile-new-reading" text circle aria-label="新建研读" title="新建研读" @click="newReading"><el-icon><Plus /></el-icon></el-button><span v-if="library" class="context-chip"><span class="tiny-dot"></span>{{ library.title }}<small>{{ library.kind === 'shared' ? '共享' : '私有' }}</small></span><router-link to="/tasks" class="connection-indicator"><span :class="['tiny-dot', { working: ws.activeTasks.length }]"></span>{{ ws.activeTasks.length ? `${ws.activeTasks.length} 项任务处理中或等待` : '内网工作空间' }}</router-link></div></header><main class="page-area"><router-view /></main></div>
  </div>
</template>
