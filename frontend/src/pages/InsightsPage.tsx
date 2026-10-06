import clsx from 'clsx'
import { useEffect, useState, type ReactNode } from 'react'
import { Link } from 'react-router'

import { BarList, MoodChart, MoodTable } from '../components/charts'
import { Card, EmotionChip } from '../components/ui'
import { api } from '../lib/api'
import { EMOTION_LABEL, STRATEGY_LABEL, TOPIC_LABEL, emotionColor } from '../lib/emotions'
import { EMOTIONS, type Emotion, type InsightsData } from '../lib/types'

const RANGES = [7, 30, 90] as const

function Tile({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Card className="space-y-1 p-4">
      <p className="text-sm text-ink-2">{label}</p>
      <div className="text-2xl font-semibold text-ink">{children}</div>
    </Card>
  )
}

export function InsightsPage() {
  const [days, setDays] = useState<number>(30)
  const [data, setData] = useState<InsightsData | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [asTable, setAsTable] = useState(false)

  useEffect(() => {
    let cancelled = false
    api.insights(days).then((d) => {
      if (!cancelled) {
        setData(d)
        setRefreshing(false)
      }
    })
    return () => {
      cancelled = true
    }
  }, [days])

  const topEmotion = data ? (Object.entries(data.emotions).sort((a, b) => b[1] - a[1])[0]?.[0] as Emotion | undefined) : undefined
  const topTopic = data ? Object.entries(data.topics)[0]?.[0] : undefined

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-6 px-4 py-6 sm:px-6">
        <header className="space-y-1">
          <h1 className="font-display text-3xl font-semibold tracking-tight">Insights</h1>
          <p className="text-ink-2">Patterns in how you’ve been feeling, based on what Aura noticed in your messages.</p>
        </header>

        <div className="flex flex-wrap items-center gap-2" role="radiogroup" aria-label="Time range">
          {RANGES.map((d) => (
            <button
              key={d}
              role="radio"
              aria-checked={days === d}
              onClick={() => {
                if (d === days) return
                setRefreshing(true)
                setDays(d)
              }}
              className={clsx(
                'rounded-full border px-3.5 py-1.5 text-sm transition',
                days === d ? 'border-accent bg-accent-soft font-medium text-ink' : 'border-line text-ink-2 hover:bg-surface-2',
              )}
            >
              Last {d} days
            </button>
          ))}
        </div>

        {data && data.messages === 0 ? (
          <Card className="space-y-2 p-8 text-center">
            <p className="font-medium">Nothing to show for this period yet.</p>
            <p className="text-sm text-ink-2">
              Insights appear after you’ve talked with Aura. <Link to="/chat" className="text-accent underline-offset-2 hover:underline">Start a conversation</Link>
            </p>
          </Card>
        ) : data ? (
          <div className={clsx('space-y-6 transition-opacity', refreshing && 'opacity-60')}>
            <div className="grid gap-3 sm:grid-cols-3">
              <Tile label="Messages Aura analysed">{data.messages.toLocaleString()}</Tile>
              <Tile label="Most frequent feeling">{topEmotion ? <EmotionChip emotion={topEmotion} className="text-sm" /> : '–'}</Tile>
              <Tile label="Most common topic">
                <span className="text-lg">{topTopic ? TOPIC_LABEL[topTopic] ?? topTopic : '–'}</span>
              </Tile>
            </div>

            <Card className="space-y-3 p-4 sm:p-5">
              <div className="flex items-baseline justify-between gap-2">
                <div>
                  <h2 className="font-semibold">Mood over time</h2>
                  <p className="text-xs text-ink-3">Each point is one of your messages: how pleasant the detected feeling was, weighted by confidence.</p>
                </div>
                <button className="shrink-0 text-xs text-ink-2 underline-offset-2 hover:underline" onClick={() => setAsTable((v) => !v)}>
                  {asTable ? 'Show chart' : 'Show as table'}
                </button>
              </div>
              {asTable ? <MoodTable points={data.timeline} /> : <MoodChart points={data.timeline} />}
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card className="space-y-4 p-4 sm:p-5">
                <h2 className="font-semibold">Feelings detected</h2>
                <BarList
                  label="Messages per feeling"
                  items={EMOTIONS.map((e) => ({ key: e, label: EMOTION_LABEL[e], value: data.emotions[e] ?? 0, dot: emotionColor(e) }))}
                />
              </Card>
              <Card className="space-y-3 p-4 sm:p-5">
                <h2 className="font-semibold">What’s been on your mind</h2>
                {data.top_causes.length ? (
                  <ol className="space-y-2">
                    {data.top_causes.map((c) => (
                      <li key={c.text} className="flex items-start justify-between gap-3 rounded-xl bg-surface-2 px-3 py-2 text-sm">
                        <span className="text-ink">“{c.text}”</span>
                        <span className="shrink-0 text-ink-3 tabular-nums">{c.count}×</span>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="text-sm text-ink-3">No explicit causes yet. They appear when you say why you feel a certain way.</p>
                )}
              </Card>
              <Card className="space-y-4 p-4 sm:p-5">
                <h2 className="font-semibold">Topics</h2>
                <BarList
                  label="Messages per topic"
                  items={Object.entries(data.topics).map(([k, v]) => ({ key: k, label: TOPIC_LABEL[k] ?? k, value: v }))}
                />
              </Card>
              <Card className="space-y-4 p-4 sm:p-5">
                <h2 className="font-semibold">How Aura supported you</h2>
                <BarList
                  label="Replies per support strategy"
                  items={Object.entries(data.strategies).map(([k, v]) => ({ key: k, label: STRATEGY_LABEL[k] ?? k, value: v }))}
                />
              </Card>
            </div>
          </div>
        ) : (
          <p className="text-sm text-ink-3">Loading…</p>
        )}
      </div>
    </div>
  )
}
