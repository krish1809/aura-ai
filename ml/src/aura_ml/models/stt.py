"""Speech-to-text with Whisper (Hugging Face transformers)."""

from __future__ import annotations

import numpy as np

from aura_ml.audio import SAMPLE_RATE


class WhisperSTT:
    def __init__(self, model_name: str, device: str | None = None):
        import torch
        from transformers import pipeline

        cuda = torch.cuda.is_available() if device is None else device.startswith("cuda")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device="cuda" if cuda else "cpu",
            torch_dtype=torch.float16 if cuda else torch.float32,
        )
        self.model_name = model_name

    def transcribe(self, wav: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
        if len(wav) < sr // 4:
            return {"text": "", "duration_s": len(wav) / sr}
        out = self.pipe({"raw": wav, "sampling_rate": sr}, chunk_length_s=30, batch_size=4)
        return {
            "text": out["text"].strip(),
            "duration_s": round(len(wav) / sr, 2),
            "model": self.model_name,
        }
