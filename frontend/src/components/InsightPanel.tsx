import { AlertTriangle, Camera, FileText, Mic, ShieldCheck, X } from 'lucide-react'
import type { ReactNode } from 'react'

import { EMOTION_LABEL, STRATEGY_HELP, STRATEGY_LABEL, TOPIC_LABEL, emotionColor, pct } from '../lib/emotions'
import { EMOTIONS, type Analysis, type Distribution, type Message, type ModalitySignal } from '../lib/types'
import { EmotionChip, IconButton, Meter, Section } from './ui'

/** Fused distribution: one measure across categories -> one series colour, every bar labelled. */
export function EmotionBars({ probs, top }: { probs: Distribution; top: string }) {
  return (
    <ul className="space-y-1.5" aria-label="Emotion distribution">
      {EMOTIONS.map((e) => {
        const v = probs[e] ?? 0
        return (
          <li key={e} className="grid grid-cols-[7.5rem_1fr_2.5rem] items-center gap-2 text-xs">
            <span className="flex items-center gap-1.5 truncate">
              <span className="size-2 shrink-0 rounded-full" style={{ background: emotionColor(e) }} aria-hidden />
              <span className={e === top ? 'font-semibold text-ink' : 'text-ink-2'}>{EMOTION_LABEL[e]}</span>
            </span>
            <span className="h-2 w-full rounded-r-[4px]" aria-hidden>
              <span
                className="block h-2 rounded-r-[4px] bg-chart"
                style={{ width: `${v * 100}%`, opacity: e === top ? 1 : 0.55 }}
              />
            </span>
            <span className={`text-right tabular-nums ${e === top ? 'font-semibold text-ink' : 'text-ink-3'}`}>{pct(v)}</span>
          </li>
        )
      })}
    </ul>
  )
}

const MODALITIES: { key: 'text' | 'voice' | 'face'; label: string; icon: typeof FileText; off: string }[] = [
  { key: 'text', label: 'Words', icon: FileText, off: 'no text' },
  { key: 'voice', label: 'Voice', icon: Mic, off: 'not a voice message' },
  { key: 'face', label: 'Face', icon: Camera, off: 'camera off' },
]

function ModalityRow({ label, icon: Icon, signal, weight, off }: {
  label: string
  icon: typeof FileText
  signal: ModalitySignal | null
  weight?: number
  off: string
}) {
  const on = !!(signal?.available && signal.emotion)
  return (
    <li className="flex items-center gap-2 text-xs">
      <Icon size={14} className="shrink-0 text-ink-3" aria-hidden />
      <span className="w-12 shrink-0 text-ink-2">{label}</span>
      {on ? (
        <>
          <EmotionChip emotion={signal!.emotion!} confidence={signal!.confidence} />
          {weight !== undefined && <span className="ml-auto text-ink-3 tabular-nums">weight {pct(weight)}</span>}
        </>
      ) : (
        <span className="text-ink-3">{signal?.reason ?? off}</span>
      )}
    </li>
  )
}

function Highlighted({ text, spans }: { text: string; spans: { start: number; end: number }[] }) {
  const parts: ReactNode[] = []
  let at = 0
  for (const [i, s] of [...spans].sort((a, b) => a.start - b.start).entries()) {
    if (s.start > at) parts.push(text.slice(at, s.start))
    parts.push(
      <mark key={i} className="rounded bg-accent-soft px-0.5 text-ink">
        {text.slice(s.start, s.end)}
      </mark>,
    )
    at = s.end
  }
  parts.push(text.slice(at))
  return <p className="rounded-xl bg-surface-2 px-3 py-2 text-sm leading-relaxed text-ink-2">{parts}</p>
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-surface-2 px-3 py-2">
      <p className="text-[11px] text-ink-3">{label}</p>
      <p className="text-sm font-medium text-ink">{value}</p>
    </div>
  )
}

export function InsightPanel({ message, onClose }: { message: Message | null; onClose: () => void }) {
  const a: Analysis | null | undefined = message?.analysis
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <div>
          <h2 className="font-display text-lg font-semibold">What Aura noticed</h2>
          <p className="text-xs text-ink-3">How your message was understood before Aura replied</p>
        </div>
        <IconButton label="Close panel" onClick={onClose}>
          <X size={18} />
        </IconButton>
      </div>
      {!a ? (
        <p className="p-4 text-sm text-ink-3">Select one of your messages to see what Aura perceived.</p>
      ) : (
        <div className="flex-1 space-y-6 overflow-y-auto p-4">
          {a.safety.level !== 'none' ? (
            <div className="flex gap-2 rounded-xl bg-critical-soft px-3 py-2 text-sm">
              <AlertTriangle size={16} className="mt-0.5 shrink-0 text-critical" aria-hidden />
              <p>
                <span className="font-semibold">{a.safety.level === 'crisis' ? 'Crisis language detected.' : 'Signs of strong distress.'}</span>{' '}
                {a.safety.level === 'crisis'
                  ? 'Aura answered with a fixed safety message and helplines instead of generating a reply.'
                  : 'Aura was asked to be especially gentle and check in on your safety.'}
              </p>
            </div>
          ) : null}

          <Section title="Feeling" aside={a.emotion.modalities.length ? `from ${a.emotion.modalities.join(' + ')}` : undefined}>
            <div className="flex flex-wrap items-center gap-2">
              <EmotionChip emotion={a.emotion.emotion} confidence={a.emotion.confidence} className="text-sm" />
              {a.emotion.mixed && a.emotion.secondary && (
                <span className="text-xs text-ink-2">mixed with {EMOTION_LABEL[a.emotion.secondary].toLowerCase()}</span>
              )}
            </div>
            <EmotionBars probs={a.emotion.probs} top={a.emotion.emotion} />
            <ul className="space-y-1.5 pt-1" aria-label="Signals by modality">
              {MODALITIES.map((m) => (
                <ModalityRow key={m.key} label={m.label} icon={m.icon} off={m.off} signal={a.modalities[m.key]} weight={a.emotion.weights[m.key]} />
              ))}
            </ul>
          </Section>

          <Section title="Possible cause" aside={a.cause.extraction.source === 'model' ? 'cause-extraction model' : 'rule-based extraction'}>
            {a.cause.from === 'this message' && message ? (
              <Highlighted text={message.content} spans={a.cause.extraction.causes} />
            ) : a.cause.text ? (
              <p className="text-sm text-ink-2">
                Nothing new in this message, so Aura kept the cause from earlier:{' '}
                <span className="font-medium text-ink">“{a.cause.text}”</span>
              </p>
            ) : (
              <p className="text-sm text-ink-3">No explicit cause yet. Aura will keep listening for one.</p>
            )}
          </Section>

          {a.topic && (
            <Section title="Topic" aside={pct(a.topic.confidence)}>
              <p className="text-sm font-medium">{TOPIC_LABEL[a.topic.topic] ?? a.topic.topic}</p>
              <Meter value={a.topic.confidence} label="Topic confidence" />
            </Section>
          )}

          {a.safety.level === 'crisis' ? (
            <Section title="How Aura chose to respond">
              <p className="text-sm text-ink-2">
                With its safety response: a fixed, carefully worded message and real helplines. Aura never improvises when someone may be in danger.
              </p>
            </Section>
          ) : a.strategy && (
            <Section title="How Aura chose to respond">
              <div>
                <p className="text-sm font-medium">{STRATEGY_LABEL[a.strategy.strategy] ?? a.strategy.strategy}</p>
                <p className="text-xs text-ink-2">{STRATEGY_HELP[a.strategy.strategy]}</p>
              </div>
              <ul className="space-y-1 text-xs" aria-label="Strategy alternatives">
                {Object.entries(a.strategy.probs)
                  .slice(0, 3)
                  .map(([s, p]) => (
                    <li key={s} className="grid grid-cols-[1fr_2.5rem] items-center gap-2">
                      <span className="space-y-1">
                        <span className={s === a.strategy!.strategy ? 'text-ink' : 'text-ink-2'}>{STRATEGY_LABEL[s] ?? s}</span>
                        <Meter value={p} label={`${s} probability`} />
                      </span>
                      <span className="text-right text-ink-3 tabular-nums">{pct(p)}</span>
                    </li>
                  ))}
              </ul>
            </Section>
          )}

          {a.entities.length > 0 && (
            <Section title="People & things mentioned">
              <div className="flex flex-wrap gap-1.5">
                {a.entities.map((e) => (
                  <span key={`${e.label}-${e.text}`} className="rounded-full border border-line px-2.5 py-0.5 text-xs">
                    {e.text} <span className="text-ink-3">· {e.label === 'RELATION' ? 'person' : e.label.toLowerCase()}</span>
                  </span>
                ))}
              </div>
            </Section>
          )}

          {a.prosody && (
            <Section title="How it sounded">
              <div className="grid grid-cols-2 gap-2">
                <Stat label="Length" value={`${a.prosody.duration_s.toFixed(1)} s`} />
                <Stat label="Speaking rate" value={a.prosody.speech_rate_wps ? `${a.prosody.speech_rate_wps.toFixed(1)} words/s` : '–'} />
                <Stat label="Pitch" value={a.prosody.pitch_hz ? `${Math.round(a.prosody.pitch_hz)} Hz` : '–'} />
                <Stat label="Pitch variation" value={a.prosody.pitch_var_hz ? `±${Math.round(a.prosody.pitch_var_hz)} Hz` : '–'} />
              </div>
            </Section>
          )}

          <Section title="Processing">
            <p className="flex items-center gap-1.5 text-xs text-ink-3">
              <ShieldCheck size={13} aria-hidden /> Safety check: {a.safety.level}
            </p>
            <p className="text-xs text-ink-3">
              {Object.entries(a.timings_ms)
                .map(([k, v]) => `${k.replace('_', ' ')} ${v} ms`)
                .join(' · ')}
            </p>
          </Section>
        </div>
      )}
    </div>
  )
}
