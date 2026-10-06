import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'

import { EMOTION_LABEL, emotionColor, pct } from '../lib/emotions'
import type { InsightsData } from '../lib/types'

/** Horizontal bars for one measure across categories: one series colour, every bar labelled. */
export function BarList({ items, format = (v) => String(v), label }: {
  items: { key: string; label: string; value: number; dot?: string }[]
  format?: (v: number) => string
  label: string
}) {
  const max = Math.max(1, ...items.map((i) => i.value))
  return (
    <ul className="space-y-2" aria-label={label}>
      {items.map((i) => (
        <li key={i.key} className="grid grid-cols-[minmax(7rem,11rem)_1fr_3rem] items-center gap-3 text-sm">
          <span className="flex min-w-0 items-center gap-1.5">
            {i.dot && <span className="size-2 shrink-0 rounded-full" style={{ background: i.dot }} aria-hidden />}
            <span className="truncate text-ink-2">{i.label}</span>
          </span>
          <span className="h-2.5" aria-hidden>
            {i.value > 0 && <span className="block h-2.5 rounded-r-[4px] bg-chart" style={{ width: `${(i.value / max) * 100}%` }} />}
          </span>
          <span className="text-right text-ink tabular-nums">{format(i.value)}</span>
        </li>
      ))}
    </ul>
  )
}

type Point = InsightsData['timeline'][number]

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(640)
  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(280, entry.contentRect.width)))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, width] as const
}

const fmtDate = (iso: string) => new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
const fmtTime = (iso: string) => new Date(iso).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
const fmtDateTime = (iso: string) =>
  new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })

/**
 * Mood over time - a single series (valence = pleasantness x confidence) on one
 * axis with a zero baseline. Crosshair + tooltip on hover, arrow keys on focus.
 */
const H = 240
const M = { t: 16, r: 28, b: 30, l: 76 }

export function MoodChart({ points }: { points: Point[] }) {
  const [wrap, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number | null>(null)

  const { xs, y } = useMemo(() => {
    const t = points.map((p) => new Date(p.at).getTime())
    let lo = Math.min(...t)
    let hi = Math.max(...t)
    if (lo === hi) {
      lo -= 3_600_000
      hi += 3_600_000
    }
    const x = (v: number) => M.l + ((v - lo) / (hi - lo)) * (width - M.l - M.r)
    return { xs: t.map(x), y: (v: number) => M.t + (1 - (v + 1) / 2) * (H - M.t - M.b) }
  }, [points, width])

  if (points.length === 0) return null
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${xs[i].toFixed(1)},${y(p.valence).toFixed(1)}`).join('')
  const showMarkers = points.length <= 40
  const last = points.length - 1

  const nearest = (clientX: number, rect: DOMRect) => {
    const px = clientX - rect.left
    let best = 0
    for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - px) < Math.abs(xs[best] - px)) best = i
    return best
  }
  const onMove = (e: PointerEvent<SVGSVGElement>) => setHover(nearest(e.clientX, e.currentTarget.getBoundingClientRect()))
  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    if (e.key === 'ArrowRight') setHover((h) => Math.min(last, (h ?? -1) + 1))
    else if (e.key === 'ArrowLeft') setHover((h) => Math.max(0, (h ?? points.length) - 1))
    else return
    e.preventDefault()
  }
  const h = hover !== null ? points[hover] : null
  const sameDay = fmtDate(points[0].at) === fmtDate(points[last].at)
  const axisLabel = (iso: string) => (sameDay ? fmtTime(iso) : fmtDate(iso))
  const ticks = [
    { v: 1, label: 'Pleasant' },
    { v: 0, label: 'Neutral' },
    { v: -1, label: 'Difficult' },
  ]

  return (
    <div ref={wrap} className="relative">
      <svg
        width={width}
        height={H}
        role="img"
        aria-label={`Mood over time, ${points.length} messages. Use the arrow keys to step through them.`}
        tabIndex={0}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
        onKeyDown={onKey}
        onBlur={() => setHover(null)}
        className="overflow-visible outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
      >
        {ticks.map((t) => (
          <g key={t.v}>
            <line x1={M.l} x2={width - M.r} y1={y(t.v)} y2={y(t.v)} stroke={t.v === 0 ? 'var(--ink-3)' : 'var(--grid)'} strokeWidth={1} />
            <text x={M.l - 10} y={y(t.v)} dy="0.32em" textAnchor="end" fontSize={12} fill="var(--ink-3)">
              {t.label}
            </text>
          </g>
        ))}
        <text x={M.l} y={H - 8} fontSize={12} fill="var(--ink-3)">
          {axisLabel(points[0].at)}
        </text>
        <text x={width - M.r} y={H - 8} fontSize={12} fill="var(--ink-3)" textAnchor="end">
          {sameDay ? `${fmtDate(points[last].at)}, ${fmtTime(points[last].at)}` : fmtDate(points[last].at)}
        </text>
        <path d={path} fill="none" stroke="var(--chart)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {(showMarkers ? points.map((_, i) => i) : [last]).map((i) => (
          <circle key={i} cx={xs[i]} cy={y(points[i].valence)} r={4} fill="var(--chart)" stroke="var(--surface)" strokeWidth={2} />
        ))}
        {h && hover !== null && (
          <g pointerEvents="none">
            <line x1={xs[hover]} x2={xs[hover]} y1={M.t} y2={H - M.b} stroke="var(--ink-3)" strokeWidth={1} />
            <circle cx={xs[hover]} cy={y(h.valence)} r={5} fill="var(--chart)" stroke="var(--surface)" strokeWidth={2} />
          </g>
        )}
      </svg>
      {h && hover !== null && (
        <div
          className="pointer-events-none absolute top-2 z-10 w-56 rounded-xl border border-line bg-surface px-3 py-2 text-xs shadow-lg"
          style={{ left: Math.min(Math.max(xs[hover] - 112, 0), width - 224) }}
          role="status"
        >
          <p className="flex items-center gap-1.5 text-sm font-semibold text-ink">
            <span className="h-0.5 w-3 rounded-full" style={{ background: emotionColor(h.emotion) }} aria-hidden />
            {EMOTION_LABEL[h.emotion]} · {pct(h.confidence)}
          </p>
          <p className="text-ink-2">{fmtDateTime(h.at)}</p>
          <p className="truncate text-ink-3">{h.conversation}</p>
        </div>
      )}
    </div>
  )
}

export function MoodTable({ points }: { points: Point[] }) {
  return (
    <div className="max-h-72 overflow-auto rounded-xl border border-line">
      <table className="w-full text-left text-sm">
        <thead className="sticky top-0 bg-surface-2 text-xs text-ink-3">
          <tr>
            <th className="px-3 py-2 font-medium">When</th>
            <th className="px-3 py-2 font-medium">Feeling</th>
            <th className="px-3 py-2 text-right font-medium">Confidence</th>
            <th className="px-3 py-2 font-medium">Conversation</th>
          </tr>
        </thead>
        <tbody>
          {[...points].reverse().map((p) => (
            <tr key={p.at + p.conversation_id} className="border-t border-line">
              <td className="px-3 py-1.5 whitespace-nowrap text-ink-2 tabular-nums">{fmtDateTime(p.at)}</td>
              <td className="px-3 py-1.5">{EMOTION_LABEL[p.emotion]}</td>
              <td className="px-3 py-1.5 text-right tabular-nums">{pct(p.confidence)}</td>
              <td className="max-w-48 truncate px-3 py-1.5 text-ink-2">{p.conversation}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
