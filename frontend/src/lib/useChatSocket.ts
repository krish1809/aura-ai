import { useCallback, useEffect, useRef, useState } from 'react'

import type { ServerEvent } from './types'

export type SocketStatus = 'connecting' | 'open' | 'closed'

/**
 * One WebSocket per open conversation. Reconnects with backoff after network
 * drops; gives up on auth/ownership failures (close codes 4401/4404).
 */
export function useChatSocket(conversationId: string | undefined, onEvent: (e: ServerEvent) => void) {
  const [status, setStatus] = useState<SocketStatus>('closed')
  const wsRef = useRef<WebSocket | null>(null)
  const handler = useRef(onEvent)
  useEffect(() => {
    handler.current = onEvent
  }, [onEvent])

  useEffect(() => {
    if (!conversationId) return
    let stopped = false
    let attempt = 0
    let timer: number | undefined

    const connect = () => {
      setStatus('connecting')
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/api/ws/conversations/${conversationId}`)
      wsRef.current = ws
      ws.onopen = () => {
        attempt = 0
        setStatus('open')
      }
      ws.onmessage = (msg) => handler.current(JSON.parse(msg.data) as ServerEvent)
      ws.onclose = (ev) => {
        setStatus('closed')
        if (stopped || ev.code === 4401 || ev.code === 4404) return
        timer = window.setTimeout(connect, Math.min(1000 * 2 ** attempt++, 15000))
      }
    }
    connect()
    const ping = window.setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify({ type: 'ping' }))
    }, 25000)

    return () => {
      stopped = true
      window.clearTimeout(timer)
      window.clearInterval(ping)
      wsRef.current?.close()
      wsRef.current = null
    }
  }, [conversationId])

  const send = useCallback((payload: object) => {
    const ws = wsRef.current
    if (ws?.readyState !== WebSocket.OPEN) return false
    ws.send(JSON.stringify(payload))
    return true
  }, [])

  return { status, send }
}
