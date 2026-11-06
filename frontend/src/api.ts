import { ElMessage } from 'element-plus'
let csrf = ''
export const setCsrf = (value: string) => { csrf = value }
export class ApiError extends Error { constructor(message: string, public status: number) { super(message) } }
export async function request<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers)
  const method = (options.method || 'GET').toUpperCase()
  if (!['GET', 'HEAD'].includes(method) && csrf) headers.set('X-CSRF-Token', csrf)
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  const response = await fetch(`/api${path}`, { ...options, headers, credentials: 'same-origin' })
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    if (response.status === 401 && path !== '/auth/login') window.dispatchEvent(new Event('refmind:unauthorized'))
    const message = typeof data?.detail === 'string' ? data.detail : Array.isArray(data?.detail) ? data.detail.map((e: any) => e.msg).join('；') : `请求未完成（${response.status}）`
    throw new ApiError(message, response.status)
  }
  return data as T
}
export const post = <T = any>(path: string, data?: unknown) => request<T>(path, { method: 'POST', body: data === undefined ? undefined : JSON.stringify(data) })
export function showError(error: unknown) { ElMessage.error(error instanceof Error ? error.message : '操作未完成，请重试') }
