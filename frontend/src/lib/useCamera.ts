import { useEffect, useRef, useState } from 'react'

/**
 * While `enabled`, shows the webcam in `videoRef` and hands a small JPEG frame
 * to `onFrame` every `intervalMs`. Frames are analysed in memory on the server
 * and never stored.
 */
export function useCamera(enabled: boolean, onFrame: (base64Jpeg: string) => void, intervalMs = 1200) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [error, setError] = useState<string | null>(null)
  const frameHandler = useRef(onFrame)
  useEffect(() => {
    frameHandler.current = onFrame
  }, [onFrame])

  useEffect(() => {
    if (!enabled) return
    let stream: MediaStream | null = null
    let timer: number | undefined
    let cancelled = false
    const canvas = document.createElement('canvas')
    canvas.width = 320
    canvas.height = 240

    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop())
          return
        }
        stream = s
        setError(null)
        if (videoRef.current) {
          videoRef.current.srcObject = s
          void videoRef.current.play()
        }
        timer = window.setInterval(() => {
          const video = videoRef.current
          if (!video || video.readyState < 2) return
          canvas.getContext('2d')?.drawImage(video, 0, 0, canvas.width, canvas.height)
          frameHandler.current(canvas.toDataURL('image/jpeg', 0.7).split(',')[1])
        }, intervalMs)
      })
      .catch(() => setError('Camera unavailable or permission denied.'))

    return () => {
      cancelled = true
      window.clearInterval(timer)
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [enabled, intervalMs])

  return { videoRef, error }
}
