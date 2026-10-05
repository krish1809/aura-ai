"""Smooths per-frame facial-emotion predictions over a short time window."""

from __future__ import annotations

import time
from collections import deque

from aura_ml.fusion import average_distributions


class FaceTracker:
    def __init__(self, window_s: float = 8.0, max_frames: int = 12):
        self.window_s = window_s
        self.frames: deque[tuple[float, dict]] = deque(maxlen=max_frames)

    def add(self, result: dict) -> None:
        if result.get("available"):
            self.frames.append((time.monotonic(), result["probs"]))

    def current(self) -> dict | None:
        now = time.monotonic()
        recent = [p for t, p in self.frames if now - t <= self.window_s]
        avg = average_distributions(recent)
        if avg is None:
            return None
        top = max(avg, key=avg.get)
        return {
            "available": True,
            "probs": {k: round(v, 4) for k, v in avg.items()},
            "emotion": top,
            "confidence": round(avg[top], 4),
            "frames": len(recent),
        }
