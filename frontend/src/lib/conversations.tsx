import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

import { api } from './api'
import type { Conversation } from './types'

interface ConversationsState {
  conversations: Conversation[] | null
  refresh: () => Promise<void>
  upsertTitle: (id: string, title: string) => void
  remove: (id: string) => Promise<void>
  rename: (id: string, title: string) => Promise<void>
}

const Ctx = createContext<ConversationsState | null>(null)

export function ConversationsProvider({ children }: { children: ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[] | null>(null)

  const refresh = useCallback(async () => setConversations(await api.conversations()), [])
  useEffect(() => {
    api.conversations().then(setConversations, () => setConversations([]))
  }, [])

  const upsertTitle = useCallback(
    (id: string, title: string) =>
      setConversations((list) => {
        if (!list) return list
        const existing = list.find((c) => c.id === id)
        const now = new Date().toISOString()
        const updated = existing
          ? { ...existing, title, updated_at: now }
          : { id, title, created_at: now, updated_at: now, user_turns: 1 }
        return [updated, ...list.filter((c) => c.id !== id)]
      }),
    [],
  )

  const remove = useCallback(async (id: string) => {
    await api.deleteConversation(id)
    setConversations((list) => list?.filter((c) => c.id !== id) ?? null)
  }, [])

  const rename = useCallback(async (id: string, title: string) => {
    const conv = await api.renameConversation(id, title)
    setConversations((list) => list?.map((c) => (c.id === id ? conv : c)) ?? null)
  }, [])

  const value = useMemo(
    () => ({ conversations, refresh, upsertTitle, remove, rename }),
    [conversations, refresh, upsertTitle, remove, rename],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useConversations(): ConversationsState {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useConversations must be used inside <ConversationsProvider>')
  return ctx
}
