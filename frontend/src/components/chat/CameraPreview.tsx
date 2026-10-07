import { CameraOff, ScanFace } from 'lucide-react'
import type { RefObject } from 'react'

import type { FaceUpdate } from '../../lib/types'
import { EmotionChip, IconButton } from '../ui'

export function CameraPreview({
  videoRef,
  face,
  error,
  connected,
  onClose,
}: {
  videoRef: RefObject<HTMLVideoElement | null>
  face: FaceUpdate | null
  error: string | null
  connected: boolean
  onClose: () => void
}) {
  const smoothed = face?.smoothed
  return (
    <div className="flex overflow-hidden rounded-2xl border border-line bg-surface shadow-sm md:block" aria-label="Your camera">
      <div className="relative w-36 shrink-0 md:w-full">
        <video ref={videoRef} muted playsInline className="aspect-[4/3] h-full w-full -scale-x-100 bg-surface-3 object-cover" />
        <span className="absolute top-2 left-2 inline-flex items-center gap-1.5 rounded-full bg-black/55 px-2 py-0.5 text-[11px] font-medium text-white">
          <span className="size-1.5 animate-pulse rounded-full bg-red-500" aria-hidden /> Live
        </span>
        <IconButton label="Turn the camera off" onClick={onClose} className="absolute top-1 right-1 size-8 bg-black/40 text-white hover:bg-black/60 hover:text-white">
          <CameraOff size={15} />
        </IconButton>
      </div>
      <div className="min-w-0 space-y-1.5 p-2.5">
        <p className="flex items-center gap-1.5 text-xs font-medium text-ink-2">
          <ScanFace size={14} aria-hidden /> Your expression
        </p>
        {error ? (
          <p className="text-xs text-critical">{error}</p>
        ) : smoothed?.emotion ? (
          <EmotionChip emotion={smoothed.emotion} confidence={smoothed.confidence} />
        ) : (
          <p className="text-xs text-ink-3">{!connected ? 'Connecting…' : (face?.frame.reason ?? 'Looking for your face…')}</p>
        )}
        <p className="text-[10px] leading-tight text-ink-3">Read live and blended into Aura’s understanding. Frames are never stored.</p>
      </div>
    </div>
  )
}
