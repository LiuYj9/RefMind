import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import 'element-plus/dist/index.css'
import 'katex/dist/katex.min.css'
import './styles.css'
import App from './App.vue'
import { router } from './router'
import { useWorkspace } from './store'

const app = createApp(App)
const pinia = createPinia()
app.use(pinia).use(ElementPlus, { locale: zhCn }).use(router)
window.addEventListener('refmind:unauthorized', () => { useWorkspace().clear(); router.replace('/login') })
router.onError(error => { console.error('Workspace navigation failed', error) })
app.mount('#app')
