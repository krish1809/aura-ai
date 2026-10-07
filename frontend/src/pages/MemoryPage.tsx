import clsx from 'clsx'
import { Trash2 } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D, { type NodeObject } from 'react-force-graph-2d'

import { useDialog } from '../components/Dialog'
import { Button, Card } from '../components/ui'
import { api } from '../lib/api'
import { EMOTION_LABEL, TOPIC_LABEL } from '../lib/emotions'
import { cssVar } from '../lib/theme'
import type { GraphLink, GraphNode } from '../lib/types'

const KINDS: { kind: GraphNode['kind']; label: string }[] = [
  { kind: 'person', label: 'People' },
  { kind: 'cause', label: 'Causes' },
  { kind: 'topic', label: 'Topics' },
  { kind: 'emotion', label: 'Feelings' },
  { kind: 'entity', label: 'Places, dates & things' },
]
const RELATION: Record<string, string> = {
  causes: 'led to',
  part_of: 'is part of',
  involved_in: 'is involved in',
  associated_with: 'comes up with',
}

type Node = NodeObject<GraphNode>

/** Canvas can't read CSS variables, so resolve the palette and re-resolve on theme changes. */
function usePalette() {
  const read = () => ({
    surface: cssVar('--surface'),
    ink: cssVar('--ink-2'),
    line: cssVar('--line'),
    accent: cssVar('--accent'),
    kind: Object.fromEntries(KINDS.map(({ kind }) => [kind, cssVar(`--kind-${kind}`)])) as Record<GraphNode['kind'], string>,
  })
  const [palette, setPalette] = useState(read)
  useEffect(() => {
    const update = () => setPalette(read())
    const mo = new MutationObserver(update)
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    mq.addEventListener('change', update)
    return () => {
      mo.disconnect()
      mq.removeEventListener('change', update)
    }
  }, [])
  return palette
}

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(600)
  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, width] as const
}

const radius = (n: GraphNode) => 4 + Math.sqrt(n.mentions) * 2.5

/** Human-friendly names for topic and emotion nodes (stored as model labels). */
const display = (n: GraphNode) =>
  n.kind === 'topic' ? (TOPIC_LABEL[n.label] ?? n.label) : n.kind === 'emotion' ? (EMOTION_LABEL[n.label as keyof typeof EMOTION_LABEL] ?? n.label) : n.label

export function MemoryPage() {
  const [graph, setGraph] = useState<{ nodes: GraphNode[]; links: GraphLink[] } | null>(null)
  const [notes, setNotes] = useState<string>('')
  const [view, setView] = useState<'graph' | 'list'>('graph')
  const [focus, setFocus] = useState<GraphNode | null>(null)
  const [box, width] = useSize<HTMLDivElement>()
  const palette = usePalette()
  const dialog = useDialog()

  const load = useCallback(async () => {
    const [g, n] = await Promise.all([api.graph(), api.notes()])
    setGraph(g)
    setNotes(n.notes)
  }, [])
  useEffect(() => {
    Promise.all([api.graph(), api.notes()]).then(([g, n]) => {
      setGraph(g)
      setNotes(n.notes)
    })
  }, [])

  // force-graph mutates its input, so hand it a copy
  const data = useMemo(
    () => (graph ? { nodes: graph.nodes.map((n) => ({ ...n })), links: graph.links.map((l) => ({ ...l })) } : { nodes: [], links: [] }),
    [graph],
  )
  const byId = useMemo(() => new Map(graph?.nodes.map((n) => [n.id, n]) ?? []), [graph])
  const connections = focus
    ? (graph?.links ?? [])
        .filter((l) => l.source === focus.id || l.target === focus.id)
        .map((l) => ({ ...l, other: byId.get(l.source === focus.id ? l.target : l.source)!, outgoing: l.source === focus.id }))
    : []

  const forget = async () => {
    const ok = await dialog.confirm({
      title: 'Forget everything?',
      body: 'This erases Aura’s notes and your knowledge graph. Your chat history is kept.',
      confirmLabel: 'Forget everything',
      danger: true,
    })
    if (!ok) return
    await api.forget()
    setFocus(null)
    await load()
  }

  const empty = graph !== null && graph.nodes.length === 0

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-6xl space-y-6 px-4 py-6 sm:px-6">
        <header className="flex flex-wrap items-end justify-between gap-3">
          <div className="space-y-1">
            <h1 className="font-display text-3xl font-semibold tracking-tight">Memory</h1>
            <p className="text-ink-2">What Aura remembers so it doesn’t make you repeat yourself. You can erase it at any time.</p>
          </div>
          <Button variant="secondary" onClick={() => void forget()} disabled={empty && !notes}>
            <Trash2 size={16} /> Forget everything
          </Button>
        </header>

        <div className="grid gap-6 lg:grid-cols-[1fr_20rem]">
          <Card className="overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
              <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2" aria-label="Legend">
                {KINDS.map(({ kind, label }) => (
                  <li key={kind} className="flex items-center gap-1.5">
                    <span className="size-2.5 rounded-full" style={{ background: palette.kind[kind] }} aria-hidden />
                    {label}
                  </li>
                ))}
              </ul>
              <div className="flex rounded-lg bg-surface-2 p-0.5 text-xs" role="tablist">
                {(['graph', 'list'] as const).map((v) => (
                  <button
                    key={v}
                    role="tab"
                    aria-selected={view === v}
                    onClick={() => setView(v)}
                    className={clsx('rounded-md px-3 py-1 capitalize', view === v ? 'bg-surface font-medium text-ink shadow-xs' : 'text-ink-2')}
                  >
                    {v}
                  </button>
                ))}
              </div>
            </div>

            {empty ? (
              <p className="p-10 text-center text-sm text-ink-3">
                Nothing yet. As you talk about people, events and feelings, Aura connects them here.
              </p>
            ) : view === 'graph' ? (
              <div ref={box} className="relative h-[520px]">
                {graph && (
                  <ForceGraph2D<GraphNode, GraphLink>
                    graphData={data}
                    width={width}
                    height={520}
                    backgroundColor="rgba(0,0,0,0)"
                    nodeRelSize={4}
                    cooldownTicks={120}
                    linkColor={() => palette.line}
                    linkWidth={(l) => Math.min(4, 0.75 + l.weight * 0.5)}
                    linkDirectionalArrowLength={4}
                    linkDirectionalArrowRelPos={0.92}
                    onNodeClick={(n) => setFocus(byId.get(n.id as number) ?? null)}
                    onNodeHover={(n) => n && setFocus(byId.get(n.id as number) ?? null)}
                    nodePointerAreaPaint={(n: Node, color, ctx) => {
                      ctx.fillStyle = color
                      ctx.beginPath()
                      ctx.arc(n.x!, n.y!, Math.max(12, radius(n) + 4), 0, 2 * Math.PI)
                      ctx.fill()
                    }}
                    nodeCanvasObject={(n: Node, ctx, scale) => {
                      const r = radius(n)
                      ctx.beginPath()
                      ctx.arc(n.x!, n.y!, r + 2 / scale, 0, 2 * Math.PI) // 2px surface ring
                      ctx.fillStyle = focus?.id === n.id ? palette.accent : palette.surface
                      ctx.fill()
                      ctx.beginPath()
                      ctx.arc(n.x!, n.y!, r, 0, 2 * Math.PI)
                      ctx.fillStyle = palette.kind[n.kind]
                      ctx.fill()
                      const full = display(n)
                      const label = full.length > 28 ? `${full.slice(0, 27)}…` : full
                      ctx.font = `${11 / scale}px Inter Variable, system-ui, sans-serif`
                      ctx.textAlign = 'center'
                      ctx.textBaseline = 'top'
                      ctx.fillStyle = palette.ink
                      ctx.fillText(label, n.x!, n.y! + r + 3 / scale)
                    }}
                  />
                )}
              </div>
            ) : (
              <div className="grid gap-4 p-4 sm:grid-cols-2">
                {KINDS.map(({ kind, label }) => {
                  const items = (graph?.nodes ?? []).filter((n) => n.kind === kind).sort((a, b) => b.mentions - a.mentions)
                  if (!items.length) return null
                  return (
                    <div key={kind} className="space-y-1.5">
                      <h3 className="flex items-center gap-1.5 text-xs font-semibold tracking-wide text-ink-3 uppercase">
                        <span className="size-2 rounded-full" style={{ background: palette.kind[kind] }} aria-hidden /> {label}
                      </h3>
                      <ul className="space-y-1">
                        {items.map((n) => (
                          <li key={n.id}>
                            <button
                              onClick={() => setFocus(n)}
                              className="flex w-full justify-between gap-3 rounded-lg px-2 py-1 text-left text-sm hover:bg-surface-2"
                            >
                              <span className="truncate">{display(n)}</span>
                              <span className="shrink-0 text-ink-3 tabular-nums">{n.mentions}×</span>
                            </button>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )
                })}
              </div>
            )}
          </Card>

          <div className="space-y-6">
            <Card className="space-y-2 p-4">
              <h2 className="font-semibold">{focus ? display(focus) : 'Details'}</h2>
              {focus ? (
                <>
                  <p className="text-xs text-ink-3">
                    {KINDS.find((k) => k.kind === focus.kind)?.label} · mentioned {focus.mentions}× · last on{' '}
                    {new Date(focus.last_seen).toLocaleDateString()}
                  </p>
                  {connections.length ? (
                    <ul className="space-y-1 text-sm">
                      {connections.map((c) => (
                        <li key={`${c.source}-${c.target}-${c.relation}`} className="text-ink-2">
                          {c.outgoing ? (
                            <>
                              <span className="text-ink">{display(focus)}</span> {RELATION[c.relation] ?? c.relation}{' '}
                              <span className="text-ink">{c.other ? display(c.other) : ''}</span>
                            </>
                          ) : (
                            <>
                              <span className="text-ink">{c.other ? display(c.other) : ''}</span> {RELATION[c.relation] ?? c.relation}{' '}
                              <span className="text-ink">{display(focus)}</span>
                            </>
                          )}
                          {c.weight > 1 && <span className="text-ink-3"> ({c.weight}×)</span>}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-ink-3">No connections yet.</p>
                  )}
                </>
              ) : (
                <p className="text-sm text-ink-3">Hover or tap a node to see how it connects.</p>
              )}
            </Card>
            <Card className="space-y-2 p-4">
              <h2 className="font-semibold">Notes Aura keeps</h2>
              {notes ? (
                <p className="text-sm leading-relaxed whitespace-pre-line text-ink-2">{notes}</p>
              ) : (
                <p className="text-sm text-ink-3">Aura writes short notes after a few exchanges, and uses them to remember context next time.</p>
              )}
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
