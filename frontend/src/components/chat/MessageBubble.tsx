import clsx from 'clsx'
import { Mic, Sparkles } from 'lucide-react'

import { STRATEGY_LABEL } from '../../lib/emotions'
import type { Message } from '../../lib/types'
import { EmotionChip, Orb } from '../ui'

export function UserBubble({
  message,
  selected,
  onInspect,
}: {
  message: Pick<Message, 'content' | 'modality'> & Partial<Message>
  selected?: boolean
  onInspect?: () => void
}) {
  const emotion = message.analysis?.emotion
  return (
    <div className="flex flex-col items-end gap-1.5">
      <div
        className={clsx(
          'max-w-[85%] rounded-2xl rounded-br-md bg-user-bubble px-4 py-2.5 text-[15px] leading-relaxed whitespace-pre-wrap text-ink sm:max-w-[75%]',
          selected && 'ring-2 ring-accent/40',
        )}
      >
        {message.modality === 'voice' && <Mic size={14} className="mr-1.5 inline -translate-y-px text-ink-3" aria-label="Voice message" />}
        {message.content || <span className="text-ink-3 italic">Transcribing…</span>}
      </div>
      {message.analysis ? (
        <div className="flex items-center gap-1.5">
          {emotion && emotion.modalities.length > 0 && <EmotionChip emotion={emotion.emotion} />}
          {onInspect && (
            <button
              onClick={onInspect}
              className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs text-ink-3 transition hover:bg-surface-2 hover:text-ink"
            >
              <Sparkles size={12} aria-hidden /> Why this reply?
            </button>
          )}
        </div>
      ) : (
        <span className="text-xs text-ink-3">Understanding…</span>
      )}
    </div>
  )
}

export function AssistantBubble({ content, strategy, streaming }: { content: string; strategy?: string | null; streaming?: boolean }) {
  return (
    <div className="flex gap-3">
      <Orb className="mt-1 size-7 shrink-0" thinking={streaming && !content} />
      <div className="min-w-0 space-y-1">
        <div className="text-[15px] leading-relaxed whitespace-pre-wrap text-ink">
          {content || (streaming ? <span className="text-ink-3">Aura is thinking…</span> : null)}
          {streaming && content && <span className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 animate-pulse rounded-sm bg-ink-3" aria-hidden />}
        </div>
        {strategy && !streaming && <p className="text-xs text-ink-3">{STRATEGY_LABEL[strategy] ?? strategy}</p>}
      </div>
    </div>
  )
}
