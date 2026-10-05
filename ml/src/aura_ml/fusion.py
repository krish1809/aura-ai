"""Late fusion of per-modality emotion distributions.

Each modality reports a distribution over the canonical emotions. Missing
modalities are dropped and the remaining weights renormalised, so a text-only
message is judged on text alone.

Default weights favour text: it is the modality whose model sees in-domain
data. The voice model is trained on acted speech (RAVDESS) and the face model on
FER2013 photos, both of which differ from a real webcam/microphone, so they act
as supporting evidence rather than overriding what the person actually wrote.
"""

from __future__ import annotations

from aura_ml.labels import EMOTIONS

DEFAULT_WEIGHTS = {"text": 0.5, "voice": 0.25, "face": 0.25}


def fuse(
    signals: dict[str, dict | None],
    weights: dict[str, float] | None = None,
    mixed_margin: float = 0.1,
) -> dict:
    weights = weights or DEFAULT_WEIGHTS
    available = {
        m: s
        for m, s in signals.items()
        if s and s.get("available", True) and s.get("probs") and weights.get(m, 0) > 0
    }
    if not available:
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "probs": dict.fromkeys(EMOTIONS, 0.0),
            "modalities": [],
            "weights": {},
            "agreement": None,
            "mixed": False,
            "secondary": None,
        }
    total = sum(weights[m] for m in available)
    used = {m: weights[m] / total for m in available}
    fused = {
        e: sum(used[m] * float(available[m]["probs"].get(e, 0.0)) for m in available)
        for e in EMOTIONS
    }
    ranked = sorted(fused, key=fused.get, reverse=True)
    top, second = ranked[0], ranked[1]
    tops = {m: max(s["probs"], key=s["probs"].get) for m, s in available.items()}
    agreement = sum(t == top for t in tops.values()) / len(tops)
    return {
        "emotion": top,
        "confidence": round(fused[top], 4),
        "probs": {e: round(p, 4) for e, p in fused.items()},
        "modalities": sorted(available),
        "weights": {m: round(w, 3) for m, w in used.items()},
        "per_modality": tops,
        "agreement": round(agreement, 3),
        "mixed": fused[top] - fused[second] < mixed_margin,
        "secondary": second,
    }


def average_distributions(dists: list[dict[str, float]]) -> dict[str, float] | None:
    """Mean of several distributions, e.g. face predictions from the last few webcam frames."""
    if not dists:
        return None
    return {e: sum(d.get(e, 0.0) for d in dists) / len(dists) for e in EMOTIONS}
