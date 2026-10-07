import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'

import { api } from './api'
import type { Capabilities } from './types'

interface MediaState {
  /** What this server can do (voice, face...). Known from page load - before any conversation exists. */
  capabilities: Capabilities | null
  mode: 'lite' | 'full' | null
  cameraOn: boolean
  setCameraOn: (on: boolean) => void
  /** Read Aura's replies aloud with the browser's speech synthesis. */
  speakReplies: boolean
  setSpeakReplies: (on: boolean) => void
  canSpeak: boolean
}

const MediaContext = createContext<MediaState | null>(null)

function readFlag(key: string): boolean {
  try {
    return localStorage.getItem(key) === 'on'
  } catch {
    return false
  }
}

export function MediaProvider({ children }: { children: ReactNode }) {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null)
  const [mode, setMode] = useState<'lite' | 'full' | null>(null)
  // camera state lives here so it survives switching from the start screen into a new conversation
  const [cameraOn, setCameraOn] = useState(false)
  const [speakReplies, setSpeak] = useState(() => readFlag('aura-speak'))
  const canSpeak = typeof window !== 'undefined' && 'speechSynthesis' in window

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    const poll = () =>
      api.models().then(
        (m) => {
          if (cancelled) return
          setMode(m.mode)
          if (m.ready) setCapabilities(m.components.capabilities)
          else timer = window.setTimeout(poll, 3000) // big models are still loading
        },
        () => {
          if (!cancelled) timer = window.setTimeout(poll, 5000)
        },
      )
    poll()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [])

  const value = useMemo<MediaState>(
    () => ({
      capabilities,
      mode,
      cameraOn,
      setCameraOn,
      speakReplies,
      canSpeak,
      setSpeakReplies: (on: boolean) => {
        setSpeak(on)
        if (!on) window.speechSynthesis?.cancel()
        try {
          localStorage.setItem('aura-speak', on ? 'on' : 'off')
        } catch {
          /* storage unavailable */
        }
      },
    }),
    [capabilities, mode, cameraOn, speakReplies, canSpeak],
  )
  return <MediaContext.Provider value={value}>{children}</MediaContext.Provider>
}

export function useMedia(): MediaState {
  const ctx = useContext(MediaContext)
  if (!ctx) throw new Error('useMedia must be used inside <MediaProvider>')
  return ctx
}

/** Speak a reply with the browser's built-in voice (no audio leaves the device). */
export function speak(text: string): void {
  const synth = window.speechSynthesis
  if (!synth || !text.trim()) return
  synth.cancel()
  const utterance = new SpeechSynthesisUtterance(text)
  const english = synth.getVoices().filter((v) => v.lang.startsWith('en'))
  utterance.voice = english.find((v) => /natural|google|samantha|female/i.test(v.name)) ?? english[0] ?? null
  utterance.rate = 1
  synth.speak(utterance)
}
