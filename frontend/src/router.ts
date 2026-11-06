import { createRouter, createWebHashHistory } from 'vue-router'
import { useWorkspace } from './store'

export const router = createRouter({ history: createWebHashHistory(), routes: [
  { path: '/', redirect: '/read' }, { path: '/login', component: () => import('./views/LoginPage.vue') },
  { path: '/read', component: () => import('./views/ReadingPage.vue'), meta: { title: '研读空间' } },
  { path: '/discover', component: () => import('./views/DiscoveryPage.vue'), meta: { title: '文献发现' } },
  { path: '/library', component: () => import('./views/LibraryPage.vue'), meta: { title: '文献库' } },
  { path: '/tasks', component: () => import('./views/TasksPage.vue'), meta: { title: '任务中心' } },
  { path: '/admin', component: () => import('./views/AdminPage.vue'), meta: { title: '管理设置', admin: true } },
  { path: '/:pathMatch(.*)*', redirect: '/read' },
] })
router.beforeEach(async to => {
  const workspace = useWorkspace()
  await workspace.initialize()
  if (!workspace.user && to.path !== '/login') return '/login'
  if (workspace.user && to.path === '/login') return '/read'
  if (to.meta.admin && !workspace.user?.is_admin) return '/read'
})
