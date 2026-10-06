/**
 * Microphone capture -> 16 kHz mono 16-bit PCM WAV (base64), the format the
 * backend decodes without extra dependencies. Uses an AudioWorklet so capture
 * stays glitch-free while the UI is busy rendering a streamed reply.
 */
export const TARGET_RATE = 16000
export const MAX_SECONDS = 60

export class VoiceRecorder {
  private ctx: AudioContext | null = null
  private stream: MediaStream | null = null
  private node: AudioWorkletNode | null = null
  private chunks: Float32Array[] = []
  private length = 0

  async start(onLevel: (rms: number) => void, onLimit: () => void): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    })
    this.ctx = new AudioContext()
    await this.ctx.audioWorklet.addModule('/recorder-worklet.js')
    const source = this.ctx.createMediaStreamSource(this.stream)
    this.node = new AudioWorkletNode(this.ctx, 'aura-recorder')
    const limit = MAX_SECONDS * this.ctx.sampleRate
    this.node.port.onmessage = (e: MessageEvent<Float32Array>) => {
      const chunk = e.data
      this.chunks.push(chunk)
      this.length += chunk.length
      let sum = 0
      for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i]
      onLevel(Math.sqrt(sum / chunk.length))
      if (this.length >= limit) onLimit()
    }
    source.connect(this.node)
  }

  get seconds(): number {
    return this.ctx ? this.length / this.ctx.sampleRate : 0
  }

  async stop(): Promise<{ base64: string; seconds: number }> {
    const rate = this.ctx?.sampleRate ?? TARGET_RATE
    const samples = new Float32Array(this.length)
    let offset = 0
    for (const c of this.chunks) {
      samples.set(c, offset)
      offset += c.length
    }
    const seconds = this.length / rate
    await this.cancel()
    return { base64: toBase64(encodeWav(downsample(samples, rate, TARGET_RATE), TARGET_RATE)), seconds }
  }

  async cancel(): Promise<void> {
    this.node?.disconnect()
    this.stream?.getTracks().forEach((t) => t.stop())
    if (this.ctx && this.ctx.state !== 'closed') await this.ctx.close()
    this.ctx = null
    this.node = null
    this.stream = null
    this.chunks = []
    this.length = 0
  }
}

/** Box-filter downsampling: averages each output window (cheap anti-aliasing). */
export function downsample(input: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return input
  const ratio = from / to
  const out = new Float32Array(Math.floor(input.length / ratio))
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio)
    const end = Math.min(input.length, Math.floor((i + 1) * ratio))
    let sum = 0
    for (let j = start; j < end; j++) sum += input[j]
    out[i] = end > start ? sum / (end - start) : 0
  }
  return out
}

export function encodeWav(samples: Float32Array, rate: number): Uint8Array {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const write = (offset: number, s: string) => [...s].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)))
  write(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  write(8, 'WAVE')
  write(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, rate, true)
  view.setUint32(28, rate * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  write(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Uint8Array(buffer)
}

export function toBase64(bytes: Uint8Array): string {
  let binary = ''
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  }
  return btoa(binary)
}
