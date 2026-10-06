import clsx from 'clsx'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

import { EMOTION_LABEL, emotionColor, pct } from '../lib/emotions'
import type { Emotion } from '../lib/types'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'

export function Button({
  variant = 'primary',
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      {...props}
      className={clsx(
        'inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50',
        variant === 'primary' && 'bg-accent text-accent-ink hover:brightness-110',
        variant === 'secondary' && 'border border-line bg-surface text-ink hover:bg-surface-2',
        variant === 'ghost' && 'text-ink-2 hover:bg-surface-2 hover:text-ink',
        variant === 'danger' && 'bg-critical text-white hover:brightness-110',
        className,
      )}
    />
  )
}

export function IconButton({
  label,
  active,
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; active?: boolean }) {
  return (
    <button
      {...props}
      aria-label={label}
      title={props.title ?? label}
      aria-pressed={active}
      className={clsx(
        'grid size-10 shrink-0 place-items-center rounded-xl transition',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-40',
        active ? 'bg-accent-soft text-accent' : 'text-ink-2 hover:bg-surface-2 hover:text-ink',
        className,
      )}
    >
      {children}
    </button>
  )
}

/** Emotion identity: a coloured dot *beside* text - the colour is never the only cue. */
export function EmotionChip({ emotion, confidence, className }: { emotion: Emotion; confidence?: number; className?: string }) {
  return (
    <span className={clsx('inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-2.5 py-0.5 text-xs text-ink-2', className)}>
      <span className="size-2 rounded-full" style={{ background: emotionColor(emotion) }} aria-hidden />
      <span className="font-medium text-ink">{EMOTION_LABEL[emotion]}</span>
      {confidence !== undefined && <span className="tabular-nums">{pct(confidence)}</span>}
    </span>
  )
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={clsx('rounded-2xl border border-line bg-surface', className)}>{children}</div>
}

export function Section({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="space-y-2.5">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-xs font-semibold tracking-wide text-ink-3 uppercase">{title}</h3>
        {aside && <span className="text-xs text-ink-3">{aside}</span>}
      </div>
      {children}
    </section>
  )
}

/** A single-value meter: the track is a lighter step of the fill's own hue. */
export function Meter({ value, label }: { value: number; label: string }) {
  return (
    <div
      className="h-1.5 w-full overflow-hidden rounded-full bg-chart-soft"
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(value * 100)}
    >
      <div className="h-full rounded-full bg-chart" style={{ width: `${Math.max(2, value * 100)}%` }} />
    </div>
  )
}

export function Orb({ className, thinking }: { className?: string; thinking?: boolean }) {
  return <span className={clsx('aura-orb inline-block rounded-full', thinking && 'animate-breathe', className)} aria-hidden />
}

export function Logo({ className }: { className?: string }) {
  return (
    <span className={clsx('inline-flex items-center gap-2', className)}>
      <Orb className="size-7" />
      <span className="font-display text-xl font-semibold tracking-tight">Aura</span>
    </span>
  )
}
