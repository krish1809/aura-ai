import clsx from 'clsx'
import { PanelRightClose, PanelRightOpen, WifiOff } from 'lucide-react'
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router'

import { InsightPanel } from '../components/InsightPanel'
import { CameraPreview } from '../components/chat/CameraPreview'
import { Composer } from '../components/chat/Composer'
import { CrisisCard } from '../components/chat/CrisisCard'
import { AssistantBubble, UserBubble } from '../components/chat/MessageBubble'
import { IconButton, Orb } from '../components/ui'
import { ApiError, api } from '../lib/api'
import { useAuth } from '../lib/auth'
import { useConversations } from '../lib/conversations'
import { initialChatState as initial, chatReducer as reducer } from '../lib/chatState'
import type { Message, ServerEvent } from '../lib/types'
import { useCamera } from '../lib/useCamera'
import { useChatSocket } from '../lib/useChatSocket'

const SUGGESTIONS = [
  'I’ve been really stressed about my exams',
  'I had a big fight with my best friend',
  'I can’t sleep and I don’t know why',
  'Something good happened today!',
]

function usePanelPreference(): [boolean, (v: boolean) => void] {
  const [open, setOpen] = useState(() => {
    try {
      return localStorage.getItem('aura-panel') !== 'closed' && window.matchMedia('(min-width: 1280px)').matches
    } catch {
      return false
    }
  })
  const set = (v: boolean) => {
    setOpen(v)
    try {
      localStorage.setItem('aura-panel', v ? 'open' : 'closed')
    } catch {
      /* ignore */
    }
  }
  return [open, set]
}

export function ChatPage() {
  const { id } = useParams()
  const location = useLocation()
  const navigate = useNavigate()
  const { user } = useAuth()
  const { upsertTitle } = useConversations()
  const [state, dispatch] = useReducer(reducer, { ...initial, loading: !!id })
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [panelOpen, setPanelOpen] = usePanelPreference()
  const [cameraOn, setCameraOn] = useState(false)
  const scroller = useRef<HTMLDivElement>(null)
  const firstSent = useRef(false)

  const onEvent = useCallback(
    (e: ServerEvent) => {
      dispatch(e)
      if (e.type === 'done') upsertTitle(e.conversation.id, e.conversation.title)
      if (e.type === 'analysis') setSelectedId(e.message.id)
    },
    [upsertTitle],
  )
  const { status, send } = useChatSocket(id, onEvent)

  // load history (this component is keyed by conversation id, so it starts fresh)
  useEffect(() => {
    if (!id) return
    let cancelled = false
    api.conversation(id).then(
      (c) => {
        if (cancelled) return
        dispatch({ type: 'loaded', messages: c.messages })
        const lastUser = [...c.messages].reverse().find((m) => m.role === 'user')
        setSelectedId(lastUser?.id ?? null)
      },
      (err) => {
        if (!cancelled && err instanceof ApiError && err.status === 404) navigate('/chat', { replace: true })
      },
    )
    return () => {
      cancelled = true
    }
  }, [id, navigate])

  // the first message of a brand-new conversation is sent once its socket opens
  useEffect(() => {
    const first = (location.state as { first?: { text?: string; audio?: string } } | null)?.first
    if (!first || status !== 'open' || firstSent.current) return
    firstSent.current = true
    send({ type: 'message', text: first.text ?? '', audio: first.audio ?? null })
    dispatch({ type: 'sending', content: first.text ?? '', modality: first.audio ? 'voice' : 'text' })
    navigate(location.pathname, { replace: true, state: null })
  }, [location.state, location.pathname, status, send, navigate])

  // keep the newest content in view
  useEffect(() => {
    const el = scroller.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [state.messages.length, state.pending, state.streaming, state.crisis])

  const busy = state.pending !== null || state.streaming !== null

  const submit = async (payload: { text?: string; audio?: string }) => {
    if (busy) return
    if (!id) {
      try {
        const conv = await api.createConversation()
        upsertTitle(conv.id, conv.title)
        navigate(`/chat/${conv.id}`, { state: { first: payload } })
      } catch {
        dispatch({ type: 'error', detail: 'Could not start a conversation. Is the server running?' })
      }
      return
    }
    if (!send({ type: 'message', text: payload.text ?? '', audio: payload.audio ?? null })) {
      dispatch({ type: 'error', detail: 'Not connected yet - reconnecting…' })
      return
    }
    dispatch({ type: 'sending', content: payload.text ?? '', modality: payload.audio ? 'voice' : 'text' })
  }

  const sendFrame = useCallback((image: string) => send({ type: 'frame', image }), [send])
  const { videoRef, error: cameraError } = useCamera(cameraOn && !!id, sendFrame)

  const selected = useMemo(() => state.messages.find((m) => m.id === selectedId) ?? null, [state.messages, selectedId])
  const empty = !state.loading && state.messages.length === 0 && !state.pending

  const inspect = (m: Message) => {
    setSelectedId(m.id)
    setPanelOpen(true)
  }

  return (
    <div className="flex min-h-0 flex-1">
      <div className="relative flex min-w-0 flex-1 flex-col">
        <div className="flex min-h-12 items-center justify-end gap-2 px-3 pt-2">
          {id && status !== 'open' && (
            <span className="mr-auto inline-flex items-center gap-1.5 rounded-full bg-warning-soft px-2.5 py-1 text-xs text-ink">
              <WifiOff size={12} aria-hidden /> {status === 'connecting' ? 'Connecting…' : 'Offline - retrying'}
            </span>
          )}
          {!empty && (
            <IconButton label={panelOpen ? 'Hide what Aura noticed' : 'Show what Aura noticed'} onClick={() => setPanelOpen(!panelOpen)}>
              {panelOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
            </IconButton>
          )}
        </div>

        <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto flex max-w-3xl flex-col gap-6 px-4 pt-4 pb-8">
            {empty && (
              <div className="flex flex-col items-center gap-6 pt-[8vh] text-center">
                <Orb className="size-16" thinking />
                <div className="space-y-2">
                  <h1 className="font-display text-3xl font-semibold tracking-tight">Hi {user?.name?.split(' ')[0]}, how are you feeling today?</h1>
                  <p className="text-ink-2">Talk, type or turn on your camera. Aura listens to how you feel, not just what you say.</p>
                </div>
                <div className="flex max-w-xl flex-wrap justify-center gap-2">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => void submit({ text: s })}
                      className="rounded-full border border-line bg-surface px-3.5 py-1.5 text-sm text-ink-2 transition hover:border-accent/40 hover:text-ink"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {state.loading && <p className="text-center text-sm text-ink-3">Loading conversation…</p>}

            {state.messages.map((m) =>
              m.role === 'user' ? (
                <UserBubble key={m.id} message={m} selected={panelOpen && m.id === selectedId} onInspect={() => inspect(m)} />
              ) : (
                <AssistantBubble key={m.id} content={m.content} strategy={m.meta?.strategy} />
              ),
            )}
            {state.pending && <UserBubble message={state.pending} />}
            {state.crisis && <CrisisCard resources={state.crisis} onClose={() => dispatch({ type: 'dismiss-crisis' })} />}
            {state.streaming !== null && <AssistantBubble content={state.streaming} streaming />}
            {state.error && (
              <div role="alert" className="flex items-center justify-between gap-3 rounded-xl bg-warning-soft px-4 py-2.5 text-sm">
                <span>{state.error}</span>
                <button className="text-ink-2 underline" onClick={() => dispatch({ type: 'clear-error' })}>
                  Dismiss
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="mx-auto w-full max-w-3xl px-4 pb-4">
          <Composer
            busy={busy || (!!id && status !== 'open')}
            capabilities={state.capabilities}
            cameraOn={cameraOn}
            onToggleCamera={() => setCameraOn((v) => !v)}
            onSendText={(text) => void submit({ text })}
            onSendVoice={(audio) => void submit({ audio })}
          />
        </div>

        {cameraOn && id && (
          <div className="absolute right-4 bottom-32 z-10">
            <CameraPreview videoRef={videoRef} face={state.face} error={cameraError} />
          </div>
        )}
      </div>

      {panelOpen && !empty && (
        <>
          <button className="fixed inset-0 z-30 bg-black/30 xl:hidden" aria-label="Close panel" onClick={() => setPanelOpen(false)} />
          <aside
            className={clsx(
              'fixed inset-y-0 right-0 z-40 w-[min(26rem,100%)] border-l border-line bg-surface shadow-xl',
              'xl:static xl:z-auto xl:w-96 xl:shadow-none',
            )}
            aria-label="What Aura noticed"
          >
            <InsightPanel message={selected} onClose={() => setPanelOpen(false)} />
          </aside>
        </>
      )}
    </div>
  )
}
