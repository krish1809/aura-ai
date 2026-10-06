"""Emotion from text.

`TextEmotionClassifier` wraps a pretrained DistilRoBERTa model whose 7 labels are
exactly Aura's canonical emotions. `LexiconEmotionClassifier` is a dependency-free
fallback used when torch is not installed (lite mode, CI).
"""

from __future__ import annotations

import re

from aura_ml.labels import EMOTIONS


class TextEmotionClassifier:
    source = "model"

    def __init__(self, model_name: str, device: str | None = None):
        import torch
        from transformers import pipeline

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = pipeline(
            "text-classification", model=model_name, top_k=None, device=dev, truncation=True
        )

    def predict(self, text: str) -> dict:
        scores = {d["label"].lower(): float(d["score"]) for d in self.pipe(text)[0]}
        probs = {e: round(scores.get(e, 0.0), 4) for e in EMOTIONS}
        top = max(probs, key=probs.get)
        return {"available": True, "probs": probs, "emotion": top, "confidence": probs[top]}


class TfidfEmotionClassifier:
    """CPU model trained on GoEmotions (aura_ml.training.train_text_emotion)."""

    source = "tfidf-goemotions"

    def __init__(self, model_path):
        import joblib

        self.pipeline = joblib.load(model_path)
        self.classes = list(self.pipeline.classes_)

    def predict(self, text: str) -> dict:
        raw = dict(zip(self.classes, self.pipeline.predict_proba([text])[0], strict=True))
        probs = {e: round(float(raw.get(e, 0.0)), 4) for e in EMOTIONS}
        top = max(probs, key=probs.get)
        return {"available": True, "probs": probs, "emotion": top, "confidence": probs[top]}


_LEXICON = {
    "anger": "angry mad furious annoyed irritated frustrated pissed hate rage resent unfair yelled",
    "disgust": "disgusted gross sick revolting ashamed shame disgusting",
    "fear": "anxious anxiety afraid scared worried worry nervous panic terrified stress stressed fear dread overwhelmed",
    "joy": "happy glad excited great good relieved grateful thankful love better proud hopeful joy",
    "sadness": "sad depressed down lonely alone cry crying hopeless miss lost hurt heartbroken empty tired grief die dying death life worthless",
    "surprise": "surprised shocked unexpected suddenly wow unbelievable",
}
_WORD = re.compile(r"[a-z']+")


class LexiconEmotionClassifier:
    """Last-resort keyword fallback. Smoothed so it never claims certainty."""

    source = "lexicon"

    def __init__(self):
        self.index = {w: emo for emo, words in _LEXICON.items() for w in words.split()}

    def predict(self, text: str) -> dict:
        counts = dict.fromkeys(EMOTIONS, 0.3)
        counts["neutral"] = 0.8
        for word in _WORD.findall(text.lower()):
            for key, emo in self.index.items():
                if word.startswith(key):
                    counts[emo] += 1.0
                    break
        total = sum(counts.values())
        probs = {e: round(c / total, 4) for e, c in counts.items()}
        top = max(probs, key=probs.get)
        return {"available": True, "probs": probs, "emotion": top, "confidence": probs[top]}
