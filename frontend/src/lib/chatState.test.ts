import { describe, expect, it } from 'vitest'

import { chatReducer, initialChatState, type ChatAction, type ChatState } from './chatState'
import type { Message } from './types'

const msg = (role: Message['role'], content: string): Message => ({
  id: `${role}-${content}`,
  role,
  content,
  modality: 'text',
  analysis: null,
  meta: null,
  created_at: '2026-10-07T00:00:00Z',
})

const run = (actions: ChatAction[], start: ChatState = initialChatState) => actions.reduce(chatReducer, start)

describe('chatReducer', () => {
  it('walks one turn: optimistic message -> analysis -> streamed tokens -> stored reply', () => {
    let s = run([{ type: 'sending', content: 'I feel low', modality: 'text' }])
    expect(s.pending).toEqual({ content: 'I feel low', modality: 'text' })

    s = run([{ type: 'analysis', message: msg('user', 'I feel low') }], s)
    expect(s.pending).toBeNull()
    expect(s.streaming).toBe('') // Aura is now "thinking"

    s = run([{ type: 'token', text: 'That ' }, { type: 'token', text: 'sounds hard.' }], s)
    expect(s.streaming).toBe('That sounds hard.')

    s = run([{ type: 'done', message: msg('assistant', 'That sounds hard.'), conversation: { id: 'c', title: 't' } }], s)
    expect(s.streaming).toBeNull()
    expect(s.messages.map((m) => m.role)).toEqual(['user', 'assistant'])
  })

  it('replaces the optimistic voice message text with the transcript', () => {
    const s = run([
      { type: 'sending', content: '', modality: 'voice' },
      { type: 'transcript', text: 'I cannot sleep' },
    ])
    expect(s.pending).toEqual({ content: 'I cannot sleep', modality: 'voice' })
  })

  it('an error clears in-flight state so the composer unlocks', () => {
    const s = run([
      { type: 'sending', content: 'hi', modality: 'text' },
      { type: 'error', detail: 'Something went wrong' },
    ])
    expect(s).toMatchObject({ pending: null, streaming: null, error: 'Something went wrong' })
  })

  it('keeps crisis resources until dismissed', () => {
    const resources = [{ region: 'India', name: 'Tele-MANAS', contact: '14416' }]
    let s = run([{ type: 'safety', level: 'crisis', resources }])
    expect(s.crisis).toEqual(resources)
    s = run([{ type: 'dismiss-crisis' }], s)
    expect(s.crisis).toBeNull()
  })

  it('a reconnect without capabilities keeps the previous ones', () => {
    const caps = { voice: true, voice_emotion: true, face: false, llm: 'mock' }
    const s = run([{ type: 'ready', capabilities: caps }, { type: 'ready', capabilities: null }])
    expect(s.capabilities).toEqual(caps)
  })
})
