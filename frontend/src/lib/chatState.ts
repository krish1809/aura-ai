import type { Capabilities, CrisisResource, FaceUpdate, Message, ServerEvent } from './types'

/** Everything the chat view shows, driven by user actions and WebSocket events. */
export interface ChatState {
  messages: Message[]
  loading: boolean
  pending: { content: string; modality: 'text' | 'voice' } | null
  streaming: string | null
  error: string | null
  crisis: CrisisResource[] | null
  capabilities: Capabilities | null
  face: FaceUpdate | null
}

export type ChatAction =
  | { type: 'loaded'; messages: Message[] }
  | { type: 'sending'; content: string; modality: 'text' | 'voice' }
  | { type: 'dismiss-crisis' }
  | { type: 'clear-error' }
  | ServerEvent

export const initialChatState: ChatState = { messages: [], loading: false, pending: null, streaming: null, error: null, crisis: null, capabilities: null, face: null }

export function chatReducer(s: ChatState, a: ChatAction): ChatState {
  switch (a.type) {
    case 'loaded':
      return { ...s, loading: false, messages: a.messages }
    case 'sending':
      return { ...s, error: null, pending: { content: a.content, modality: a.modality }, streaming: null }
    case 'ready':
      return { ...s, capabilities: a.capabilities ?? s.capabilities }
    case 'transcript':
      return s.pending ? { ...s, pending: { ...s.pending, content: a.text } } : s
    case 'analysis':
      return { ...s, messages: [...s.messages, a.message], pending: null, streaming: '' }
    case 'safety':
      return { ...s, crisis: a.resources }
    case 'token':
      return { ...s, streaming: (s.streaming ?? '') + a.text }
    case 'done':
      return { ...s, messages: [...s.messages, a.message], streaming: null }
    case 'error':
      return { ...s, error: a.detail, pending: null, streaming: null }
    case 'face':
      return { ...s, face: { frame: a.frame, smoothed: a.smoothed } }
    case 'dismiss-crisis':
      return { ...s, crisis: null }
    case 'clear-error':
      return { ...s, error: null }
    default:
      return s
  }
}
