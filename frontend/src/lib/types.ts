export const EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral'] as const
export type Emotion = (typeof EMOTIONS)[number]
export type Distribution = Record<Emotion, number>

export interface ModalitySignal {
  available: boolean
  probs?: Distribution
  emotion?: Emotion
  confidence?: number
  raw?: Record<string, number>
  frames?: number
  box?: number[]
  reason?: string
}

export interface FusedEmotion {
  emotion: Emotion
  confidence: number
  probs: Distribution
  modalities: string[]
  weights: Record<string, number>
  per_modality?: Record<string, Emotion>
  agreement: number | null
  mixed: boolean
  secondary: Emotion | null
}

export interface Prosody {
  duration_s: number
  loudness_db: number
  loudness_var_db: number
  pitch_hz: number | null
  pitch_var_hz: number | null
  zero_crossing_rate: number
  voiced_ratio: number
  speech_rate_wps?: number
}

export interface Ranked {
  confidence: number
  probs: Record<string, number>
}

export interface Analysis {
  safety: { level: 'none' | 'elevated' | 'crisis'; matches: string[] }
  emotion: FusedEmotion
  modalities: { text: ModalitySignal | null; voice: ModalitySignal | null; face: ModalitySignal | null }
  prosody: Prosody | null
  cause: {
    text: string | null
    from: 'this message' | 'earlier in the conversation' | null
    extraction: {
      has_cause: boolean
      clause_confidence: number
      causes: { text: string; start: number; end: number; confidence: number | null }[]
      source: string
    }
  }
  entities: { text: string; label: string }[]
  topic: (Ranked & { topic: string }) | null
  strategy: (Ranked & { strategy: string }) | null
  timings_ms: Record<string, number>
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  modality: 'text' | 'voice'
  analysis: Analysis | null
  meta: { strategy: string | null; model: string; safety: string } | null
  created_at: string
}

export interface Conversation {
  id: string
  title: string
  created_at: string
  updated_at: string
  user_turns: number
}

export interface ConversationDetail extends Conversation {
  summary: string | null
  messages: Message[]
}

export interface User {
  id: string
  email: string
  name: string
  created_at: string
}

export interface Capabilities {
  voice: boolean
  voice_emotion: boolean
  face: boolean
  llm: string | null
}

export interface CrisisResource {
  region: string
  name: string
  contact: string
}

export interface FaceUpdate {
  frame: ModalitySignal
  smoothed: (ModalitySignal & { frames: number }) | null
}

export type ServerEvent =
  | { type: 'ready'; capabilities: Capabilities | null }
  | { type: 'transcript'; text: string }
  | { type: 'analysis'; message: Message }
  | { type: 'safety'; level: string; resources: CrisisResource[] }
  | { type: 'token'; text: string }
  | { type: 'done'; message: Message; conversation: { id: string; title: string } }
  | { type: 'error'; detail: string }
  | ({ type: 'face' } & FaceUpdate)
  | { type: 'pong' }

export interface GraphNode {
  id: number
  kind: 'person' | 'cause' | 'emotion' | 'topic' | 'entity'
  label: string
  mentions: number
  first_seen: string
  last_seen: string
}

export interface GraphLink {
  source: number
  target: number
  relation: string
  weight: number
}

export interface InsightsData {
  days: number
  messages: number
  timeline: { at: string; emotion: Emotion; confidence: number; valence: number; conversation_id: string; conversation: string }[]
  emotions: Partial<Record<Emotion, number>>
  top_causes: { text: string; count: number }[]
  topics: Record<string, number>
  strategies: Record<string, number>
  modalities: Record<string, number>
}

export interface ModelStatus {
  ready: boolean
  mode: 'lite' | 'full'
  components: {
    components: Record<string, { ready: boolean; source: string | null; load_s?: number; detail?: string }>
    capabilities: Capabilities
    error: string | null
  }
}

export interface ModelReport {
  id: string
  title: string
  protocol: string
  trained: boolean
  metrics: { label: string; value: number; kind: 'pct' | 'num' }[]
}
