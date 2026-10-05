"""Audio loading and interpretable prosodic features (numpy/scipy only)."""

from __future__ import annotations

import io
import wave
from math import gcd
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def decode_pcm_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Parse 16-bit PCM WAV (what the Aura web client records) with the standard library."""
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        frames = (
            np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        )
        channels, sr = w.getnchannels(), w.getframerate()
    return frames.reshape(-1, channels).mean(axis=1), sr


def load_audio(source: str | Path | bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode audio bytes or a file path to mono float32 at `target_sr`.
    16-bit PCM WAV needs only numpy; other formats (FLAC/OGG...) use soundfile."""
    raw = source if isinstance(source, bytes) else None
    if raw is not None and raw[:4] == b"RIFF":
        try:
            wav, sr = decode_pcm_wav(raw)
            return resample(wav, sr, target_sr)
        except (ValueError, wave.Error):
            pass
    import soundfile as sf

    data, sr = sf.read(
        io.BytesIO(raw) if raw is not None else str(source), dtype="float32", always_2d=True
    )
    return resample(data.mean(axis=1), sr, target_sr)


def resample(wav: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target_sr:
        return wav.astype(np.float32)
    try:
        from scipy.signal import resample_poly
    except ImportError:  # lite install: linear interpolation is fine for speech
        n = int(round(len(wav) * target_sr / sr))
        return np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(
            np.float32
        )
    g = gcd(sr, target_sr)
    return resample_poly(wav, target_sr // g, sr // g).astype(np.float32)


def trim_silence(wav: np.ndarray, sr: int = SAMPLE_RATE, threshold_db: float = -40.0) -> np.ndarray:
    frame = int(0.02 * sr)
    if len(wav) < frame:
        return wav
    n = len(wav) // frame
    rms = np.sqrt(np.mean(wav[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    voiced = np.where(db > threshold_db)[0]
    if voiced.size == 0:
        return wav
    return wav[voiced[0] * frame : (voiced[-1] + 1) * frame]


def _frames(wav: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(wav) < size:
        wav = np.pad(wav, (0, size - len(wav)))
    n = 1 + (len(wav) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return wav[idx]


def _pitch(frame: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 400.0) -> float | None:
    frame = frame - frame.mean()
    if np.sqrt(np.mean(frame**2)) < 1e-3:
        return None
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
    lo, hi = int(sr / fmax), min(int(sr / fmin), len(ac) - 1)
    if hi <= lo or ac[0] <= 0:
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    return sr / lag if ac[lag] / ac[0] > 0.3 else None


def prosody(wav: np.ndarray, sr: int = SAMPLE_RATE, transcript: str | None = None) -> dict:
    """Pitch, loudness and tempo summaries that explain *how* something was said."""
    frames = _frames(wav, int(0.04 * sr), int(0.02 * sr))
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    pitches = [p for p in (_pitch(f, sr) for f in frames[::2]) if p]
    duration = len(wav) / sr
    voiced_ratio = float(np.mean(rms > 0.02)) if len(rms) else 0.0
    out = {
        "duration_s": round(duration, 2),
        "loudness_db": round(float(20 * np.log10(rms.mean() + 1e-9)), 1),
        "loudness_var_db": round(float(np.std(20 * np.log10(rms + 1e-9))), 1),
        "pitch_hz": round(float(np.median(pitches)), 1) if pitches else None,
        "pitch_var_hz": round(float(np.std(pitches)), 1) if len(pitches) > 2 else None,
        "zero_crossing_rate": round(float(zcr.mean()), 4),
        "voiced_ratio": round(voiced_ratio, 2),
    }
    if transcript and duration > 0:
        out["speech_rate_wps"] = round(len(transcript.split()) / duration, 2)
    return out
