import type {
  Conversation,
  ConversationDetail,
  CrisisResource,
  GraphLink,
  GraphNode,
  InsightsData,
  ModelReport,
  ModelStatus,
  User,
} from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const { json, headers, ...rest } = init
  const res = await fetch(path, {
    credentials: 'same-origin',
    ...rest,
    headers: { ...(json !== undefined ? { 'Content-Type': 'application/json' } : {}), ...headers },
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : (body.detail?.[0]?.msg ?? detail)
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail)
  }
  return (res.status === 204 ? undefined : await res.json()) as T
}

export const api = {
  me: () => request<User>('/api/auth/me'),
  login: (email: string, password: string) => request<User>('/api/auth/login', { method: 'POST', json: { email, password } }),
  register: (name: string, email: string, password: string) =>
    request<User>('/api/auth/register', { method: 'POST', json: { name, email, password } }),
  logout: () => request<void>('/api/auth/logout', { method: 'POST' }),
  deleteAccount: () => request<void>('/api/auth/me', { method: 'DELETE' }),

  conversations: () => request<Conversation[]>('/api/conversations'),
  createConversation: () => request<Conversation>('/api/conversations', { method: 'POST', json: {} }),
  conversation: (id: string) => request<ConversationDetail>(`/api/conversations/${id}`),
  renameConversation: (id: string, title: string) =>
    request<Conversation>(`/api/conversations/${id}`, { method: 'PATCH', json: { title } }),
  deleteConversation: (id: string) => request<void>(`/api/conversations/${id}`, { method: 'DELETE' }),

  insights: (days: number) => request<InsightsData>(`/api/insights?days=${days}`),
  notes: () => request<{ notes: string; updated_at: string | null }>('/api/memory/notes'),
  graph: () => request<{ nodes: GraphNode[]; links: GraphLink[] }>('/api/memory/graph'),
  forget: () => request<void>('/api/memory', { method: 'DELETE' }),

  models: () => request<ModelStatus>('/api/system/models'),
  resources: () => request<{ crisis_resources: CrisisResource[] }>('/api/system/resources'),
  metrics: () => request<{ models: ModelReport[] }>('/api/system/metrics'),
}
