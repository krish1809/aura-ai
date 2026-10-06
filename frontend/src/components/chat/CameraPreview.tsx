import type { RefObject } from 'react'

import type { FaceUpdate } from '../../lib/types'
import { EmotionChip } from '../ui'

export function CameraPreview({ videoRef, face, error }: { videoRef: RefObject<HTMLVideoElement | null>; face: FaceUpdate | null; error: string | null }) {
  const smoothed = face?.smoothed
  return (
    <div className="w-44 overflow-hidden rounded-2xl border border-line bg-surface shadow-lg">
      <video ref={videoRef} muted playsInline className="aspect-[4/3] w-full -scale-x-100 bg-surface-3 object-cover" aria-label="Your camera" />
      <div className="space-y-1 p-2">
        {error ? (
          <p className="text-xs text-critical">{error}</p>
        ) : smoothed?.emotion ? (
          <EmotionChip emotion={smoothed.emotion} confidence={smoothed.confidence} />
        ) : (
          <p className="text-xs text-ink-3">{face?.frame.reason ?? 'Looking for your face…'}</p>
        )}
        <p className="text-[10px] leading-tight text-ink-3">Analysed live, never stored.</p>
      </div>
    </div>
  )
}
