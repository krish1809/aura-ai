import clsx from 'clsx'
import { ArrowUp, Camera, Mic, Square, X } from 'lucide-react'
import { useEffect, useRef, useState, type KeyboardEvent } from 'react'

import { MAX_SECONDS } from '../../lib/audio'
import type { Capabilities } from '../../lib/types'
import type { VoiceControls } from '../../lib/useVoiceRecorder'
import { IconButton } from '../ui'

interface Props {
  busy: boolean
  capabilities: Capabilities | null
  cameraOn: boolean
  onToggleCamera: () => void
  onSendText: (text: string) => void
  voice: VoiceControls
}

/** Waveform-style level meter shown while recording. */
function LevelBars({ level, seconds }: { level: number; seconds: number }) {
  return (
    <span className="flex h-6 flex-1 items-center gap-0.5" aria-hidden>
      {Array.from({ length: 28 }, (_, i) => (
        <span
          key={i}
          className="w-1 rounded-full bg-accent transition-[height] duration-100"
          style={{ height: `${Math.max(12, Math.min(100, level * 900 * (0.6 + 0.4 * Math.sin(i * 1.7 + seconds * 6))))}%` }}
        />
      ))}
    </span>
  )
}

export function Composer({ busy, capabilities, cameraOn, onToggleCamera, onSendText, voice }: Props) {
  const [text, setText] = useState('')
  const textarea = useRef<HTMLTextAreaElement>(null)
  const canTalk = !!capabilities?.voice
  const canSee = !!capabilities?.face

  useEffect(() => {
    const el = textarea.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [text])

  const submit = () => {
    const value = text.trim()
    if (!value || busy) return
    onSendText(value)
    setText('')
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  const unavailable = capabilities === null ? 'Checking what this server supports…' : 'needs the full model setup (see How Aura works)'

  return (
    <div className="space-y-1.5">
      <div
        className={clsx(
          'flex items-end gap-1.5 rounded-2xl border bg-surface p-1.5 shadow-sm transition',
          voice.recording ? 'border-critical/60' : 'border-line focus-within:border-accent/50',
        )}
      >
        {voice.recording ? (
          <div className="flex min-h-11 flex-1 items-center gap-3 px-2" role="status" aria-live="polite">
            <span className="size-2.5 animate-pulse rounded-full bg-critical" aria-hidden />
            <span className="text-sm font-medium text-ink tabular-nums">
              Listening… {Math.floor(voice.seconds / 60)}:{String(Math.floor(voice.seconds % 60)).padStart(2, '0')}
              <span className="font-normal text-ink-3"> / {MAX_SECONDS}s</span>
            </span>
            <LevelBars level={voice.level} seconds={voice.seconds} />
            <IconButton label="Cancel recording" onClick={() => void voice.cancel()} className="size-9">
              <X size={18} />
            </IconButton>
          </div>
        ) : (
          <>
            <button
              type="button"
              onClick={onToggleCamera}
              disabled={!canSee}
              aria-pressed={cameraOn}
              title={canSee ? (cameraOn ? 'Turn the camera off' : 'Let Aura see your expression - frames are never stored') : `Camera ${unavailable}`}
              className={clsx(
                'inline-flex h-11 shrink-0 items-center gap-1.5 rounded-xl px-3 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40',
                cameraOn ? 'bg-critical-soft text-ink' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
              )}
            >
              <Camera size={18} aria-hidden />
              <span className="hidden sm:inline">{cameraOn ? 'Camera on' : 'Camera'}</span>
              {cameraOn && <span className="size-2 rounded-full bg-critical" aria-hidden />}
            </button>
            <label htmlFor="composer" className="sr-only">
              Message Aura
            </label>
            <textarea
              id="composer"
              ref={textarea}
              rows={1}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={onKeyDown}
              maxLength={4000}
              placeholder={canTalk ? 'Type, or tap Talk to speak…' : 'Share what’s on your mind…'}
              className="max-h-[200px] min-h-11 flex-1 resize-none bg-transparent px-2 py-3 text-[15px] text-ink outline-none placeholder:text-ink-3"
            />
          </>
        )}
        {voice.recording ? (
          <button
            type="button"
            onClick={() => void voice.stopAndSend()}
            className="inline-flex h-11 shrink-0 items-center gap-1.5 rounded-xl bg-critical px-4 text-sm font-medium text-white transition hover:brightness-110"
          >
            <Square size={14} fill="currentColor" aria-hidden /> Send
          </button>
        ) : text.trim() ? (
          <IconButton label="Send message" disabled={busy} onClick={submit} className="size-11 bg-accent text-accent-ink hover:bg-accent hover:text-accent-ink hover:brightness-110">
            <ArrowUp size={18} />
          </IconButton>
        ) : (
          <button
            type="button"
            onClick={() => void voice.start()}
            disabled={busy || !canTalk}
            title={canTalk ? 'Speak instead of typing - Aura also listens to your tone of voice' : `Voice ${unavailable}`}
            className="inline-flex h-11 shrink-0 items-center gap-1.5 rounded-xl bg-accent px-4 text-sm font-medium text-accent-ink transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Mic size={18} aria-hidden /> Talk
          </button>
        )}
      </div>
      <p className={clsx('px-2 text-xs', voice.error ? 'text-critical' : 'text-ink-3')} aria-live="polite">
        {voice.error ?? 'Aura is a supportive companion, not a therapist. If you are in danger, contact local emergency services.'}
      </p>
    </div>
  )
}
