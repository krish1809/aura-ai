"""In-memory sliding-window rate limiting for the auth endpoints.

Enough for a single-process deployment; a multi-instance deployment would move
these counters to Redis.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status


class SlidingWindow:
    def __init__(self, max_events: int, window_s: float):
        self.max_events, self.window_s = max_events, window_s
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, key: str, now: float) -> deque[float]:
        events = self._events[key]
        while events and now - events[0] > self.window_s:
            events.popleft()
        return events

    def check(self, key: str) -> None:
        """Raise 429 if `key` is already at its limit (does not record an event)."""
        now = time.monotonic()
        events = self._prune(key, now)
        if len(events) >= self.max_events:
            retry = int(self.window_s - (now - events[0])) + 1
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Too many attempts - please wait a few minutes and try again.",
                headers={"Retry-After": str(retry)},
            )

    def record(self, key: str) -> None:
        self._prune(key, time.monotonic()).append(time.monotonic())

    def reset(self, key: str) -> None:
        self._events.pop(key, None)


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"
