import { useCallback, useEffect, useRef, useState } from 'react'

import { VoiceRecorder } from './audio'

export interface VoiceControls {
  recording: boolean
  seconds: number
  level: number
  error: string | null
  start: () => Promise<void>
  stopAndSend: () => Promise<void>
  cancel: () => Promise<void>
  clearError: () => void
}

/** Microphone recording shared by the start screen and the message bar. */
export function useVoiceRecorder(onRecorded: (wavBase64: string) => void): VoiceControls {
  const recorder = useRef<VoiceRecorder | null>(null)
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [level, setLevel] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const onDone = useRef(onRecorded)
  useEffect(() => {
    onDone.current = onRecorded
  }, [onRecorded])

  useEffect(() => {
    if (!recording) return
    const t = window.setInterval(() => setSeconds(recorder.current?.seconds ?? 0), 200)
    return () => window.clearInterval(t)
  }, [recording])

  useEffect(() => () => void recorder.current?.cancel(), [])

  const stopAndSend = useCallback(async () => {
    const rec = recorder.current
    if (!rec) return
    recorder.current = null
    setRecording(false)
    const { base64, seconds: secs } = await rec.stop()
    if (secs < 0.6) {
      setError('That was too short - tap the mic, speak, then tap send.')
      return
    }
    onDone.current(base64)
  }, [])

  const start = useCallback(async () => {
    setError(null)
    window.speechSynthesis?.cancel() // never record Aura's own voice
    const rec = new VoiceRecorder()
    try {
      await rec.start(setLevel, () => void stopAndSend())
      recorder.current = rec
      setSeconds(0)
      setRecording(true)
    } catch {
      await rec.cancel()
      setError('Microphone unavailable or permission denied.')
    }
  }, [stopAndSend])

  const cancel = useCallback(async () => {
    await recorder.current?.cancel()
    recorder.current = null
    setRecording(false)
  }, [])

  return { recording, seconds, level, error, start, stopAndSend, cancel, clearError: () => setError(null) }
}
