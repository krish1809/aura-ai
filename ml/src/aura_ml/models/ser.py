"""Speech emotion recognition with a fine-tuned wav2vec2 classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from aura_ml.audio import SAMPLE_RATE, trim_silence
from aura_ml.labels import RAVDESS_TO_CANONICAL, project_distribution

MAX_SECONDS = 8.0


class SpeechEmotionRecognizer:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = AutoFeatureExtractor.from_pretrained(str(model_dir))
        self.model = (
            AutoModelForAudioClassification.from_pretrained(str(model_dir)).to(self.device).eval()
        )
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.inference_mode()
    def predict(self, wav: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
        """Returns the raw label distribution and its projection onto the canonical emotions."""
        wav = trim_silence(wav, sr)[: int(MAX_SECONDS * sr)]
        if len(wav) < sr // 4:
            return {"available": False, "reason": "audio too short"}
        inputs = self.extractor(wav, sampling_rate=sr, return_tensors="pt")
        logits = self.model(inputs["input_values"].to(self.device)).logits[0].float()
        probs = dict(zip(self.labels, logits.softmax(-1).tolist(), strict=True))
        canonical = project_distribution(probs, RAVDESS_TO_CANONICAL)
        top = max(canonical, key=canonical.get)
        return {
            "available": True,
            "raw": {k: round(v, 4) for k, v in probs.items()},
            "probs": {k: round(v, 4) for k, v in canonical.items()},
            "emotion": top,
            "confidence": round(canonical[top], 4),
        }
