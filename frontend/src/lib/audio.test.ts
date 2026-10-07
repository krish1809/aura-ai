import { describe, expect, it } from 'vitest'

import { downsample, encodeWav, toBase64 } from './audio'

describe('audio encoding', () => {
  it('downsamples 48 kHz to 16 kHz by averaging windows of 3', () => {
    const out = downsample(new Float32Array([0, 0.3, 0.6, 1, 1, 1]), 48000, 16000)
    expect(Array.from(out).map((v) => Number(v.toFixed(3)))).toEqual([0.3, 1])
  })

  it('writes a 16-bit mono PCM WAV header the backend can parse', () => {
    const wav = encodeWav(new Float32Array([0, 1, -1, 0.5]), 16000)
    const view = new DataView(wav.buffer)
    const text = (o: number) => String.fromCharCode(...wav.slice(o, o + 4))
    expect([text(0), text(8), text(12), text(36)]).toEqual(['RIFF', 'WAVE', 'fmt ', 'data'])
    expect(view.getUint16(20, true)).toBe(1) // PCM
    expect(view.getUint16(22, true)).toBe(1) // mono
    expect(view.getUint32(24, true)).toBe(16000)
    expect(view.getUint16(34, true)).toBe(16) // bits per sample
    expect(view.getUint32(40, true)).toBe(8) // 4 samples x 2 bytes
    expect([view.getInt16(44, true), view.getInt16(46, true), view.getInt16(48, true)]).toEqual([0, 32767, -32768])
  })

  it('base64-encodes large buffers without overflowing the call stack', () => {
    const bytes = new Uint8Array(200_000).fill(65)
    const b64 = toBase64(bytes)
    expect(b64.length).toBe(Math.ceil(200_000 / 3) * 4)
    expect(atob(b64).slice(0, 3)).toBe('AAA')
  })
})
