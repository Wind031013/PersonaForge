import type { ChatRequest, ChatResponse, FineTuneRequest } from '../types'

const BASE_URL = '/api'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

export const api = {
  chat: (data: ChatRequest) =>
    request<ChatResponse>('/chat', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  chatSimple: (message: string) =>
    request<ChatResponse>('/chat/simple', {
      method: 'POST',
      body: JSON.stringify({ message }),
    }),

  getRoles: () =>
    request<{ roles: Array<{ id: string; name: string; description: string; avatar?: string }> }>('/roles'),

  startPipeline: (data: FineTuneRequest) =>
    request<{ task_id: string }>('/pipeline/start', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  getPipelineStatus: (taskId: string) =>
    request<{ steps: Array<{ id: string; name: string; status: string; message: string; duration?: number }> }>(
      `/pipeline/status/${taskId}`
    ),

  startCrawl: (url: string) =>
    request<{ message: string; output_dir: string }>('/pipeline/crawl', {
      method: 'POST',
      body: JSON.stringify({ url }),
    }),

  health: () =>
    request<{ status: string; model_loaded: boolean }>('/health'),
}
