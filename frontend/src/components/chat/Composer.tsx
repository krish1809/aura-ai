import clsx from 'clsx'
import { ArrowUp, Camera, CameraOff, Mic, Square, X } from 'lucide-react'
import { useEffect, useRef, useState, type KeyboardEvent } from 'react'

import { MAX_SECONDS, VoiceRecorder } from '../../lib/audio'
import type { Capabilities } from '../../lib/types'
import { IconButton } from '../ui'

interface Props {
  busy: boolean
  capabilities: Capabilities | null
  cameraOn: boolean
  onToggleCamera: () => void
  onSendText: (text: string) => void
  onSendVoice: (wavBase64: string) => void
  placeholder?: string
}

export function Composer({ busy, capabilities, cameraOn, onToggleCamera, onSendText, onSendVoice, placeholder }: Props) {
  const [text, setText] = useState('')
  const [recording, setRecording] = useState(false)
  const [level, setLevel] = useState(0)
  const [seconds, setSeconds] = useState(0)
  const [micError, setMicError] = useState<string | null>(null)
  const recorder = useRef<VoiceRecorder | null>(null)
  const textarea = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const el = textarea.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [text])

  useEffect(() => {
    if (!recording) return
    const t = window.setInterval(() => setSeconds(recorder.current?.seconds ?? 0), 200)
    return () => window.clearInterval(t)
  }, [recording])

  useEffect(() => () => void recorder.current?.cancel(), [])

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

  const stopAndSend = async () => {
    const rec = recorder.current
    if (!rec) return
    recorder.current = null
    setRecording(false)
    const { base64, seconds: secs } = await rec.stop()
    if (secs < 0.6) {
      setMicError('That was too short - hold on a little longer.')
      return
    }
    onSendVoice(base64)
  }

  const startRecording = async () => {
    setMicError(null)
    const rec = new VoiceRecorder()
    try {
      await rec.start(setLevel, () => void stopAndSend())
      recorder.current = rec
      setSeconds(0)
      setRecording(true)
    } catch {
      await rec.cancel()
      setMicError('Microphone unavailable or permission denied.')
    }
  }

  const cancelRecording = async () => {
    await recorder.current?.cancel()
    recorder.current = null
    setRecording(false)
  }

  const voiceHelp = capabilities?.voice ? 'Speak instead of typing' : 'Voice needs the full model setup (speech-to-text)'
  const cameraHelp = capabilities?.face
    ? cameraOn
      ? 'Turn camera off'
      : 'Let Aura see your expression (frames are never stored)'
    : 'Camera needs the full model setup (facial emotion model)'

  return (
    <div className="space-y-1.5">
      <div
        className={clsx(
          'flex items-end gap-1.5 rounded-2xl border border-line bg-surface p-1.5 shadow-sm transition focus-within:border-accent/50',
          recording && 'border-critical/50',
        )}
      >
        {recording ? (
          <div className="flex min-h-10 flex-1 items-center gap-3 px-2" role="status" aria-live="polite">
            <span className="size-2.5 animate-pulse rounded-full bg-critical" aria-hidden />
            <span className="text-sm text-ink tabular-nums">
              {Math.floor(seconds / 60)}:{String(Math.floor(seconds % 60)).padStart(2, '0')}
              <span className="text-ink-3"> / {MAX_SECONDS}s</span>
            </span>
            <span className="flex h-6 flex-1 items-center gap-0.5" aria-hidden>
              {Array.from({ length: 24 }, (_, i) => (
                <span
                  key={i}
                  className="w-1 rounded-full bg-accent transition-[height] duration-100"
                  style={{ height: `${Math.max(12, Math.min(100, level * 900 * (0.6 + 0.4 * Math.sin(i * 1.7 + seconds * 6))))}%` }}
                />
              ))}
            </span>
            <IconButton label="Cancel recording" onClick={() => void cancelRecording()} className="size-9">
              <X size={18} />
            </IconButton>
          </div>
        ) : (
          <>
            <IconButton label={cameraHelp} active={cameraOn} disabled={!capabilities?.face} onClick={onToggleCamera}>
              {cameraOn ? <Camera size={18} /> : <CameraOff size={18} />}
            </IconButton>
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
              placeholder={placeholder ?? 'Share what’s on your mind…'}
              className="max-h-[200px] min-h-10 flex-1 resize-none bg-transparent px-2 py-2.5 text-[15px] text-ink outline-none placeholder:text-ink-3"
            />
          </>
        )}
        {recording ? (
          <IconButton label="Stop and send" onClick={() => void stopAndSend()} className="bg-critical text-white hover:bg-critical hover:text-white">
            <Square size={16} fill="currentColor" />
          </IconButton>
        ) : text.trim() ? (
          <IconButton label="Send message" disabled={busy} onClick={submit} className="bg-accent text-accent-ink hover:bg-accent hover:text-accent-ink hover:brightness-110">
            <ArrowUp size={18} />
          </IconButton>
        ) : (
          <IconButton label={voiceHelp} disabled={busy || !capabilities?.voice} onClick={() => void startRecording()}>
            <Mic size={18} />
          </IconButton>
        )}
      </div>
      <p className={clsx('px-2 text-xs', micError ? 'text-critical' : 'text-ink-3')} aria-live="polite">
        {micError ?? 'Aura is a supportive companion, not a therapist. If you are in danger, contact local emergency services.'}
      </p>
    </div>
  )
}
