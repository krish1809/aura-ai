"""Async facade over the synchronous aura_ml model hub.

Two single-thread executors keep GPU work serialised without blocking the event
loop: one for the LLM (long, streaming) and one for perception models (STT,
emotion, cause, face). Separate lanes mean live webcam emotion keeps updating
while a reply is being generated.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from aura_api.config import Settings
from aura_ml.pipeline import HubConfig, ModelHub, TurnState, analyze_turn

log = logging.getLogger("aura_api.ml")
_END = object()


class ModelsNotReady(RuntimeError):
    pass


class MLService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.hub: ModelHub | None = None
        self.load_error: str | None = None
        self._llm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aura-llm")
        self._perception_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="aura-perception"
        )

    @property
    def ready(self) -> bool:
        return self.hub is not None

    def capabilities(self) -> dict:
        h = self.hub
        return {
            "voice": bool(h and h.stt),
            "voice_emotion": bool(h and h.ser),
            "face": bool(h and h.fer),
            "llm": h.status.get("llm", {}).get("source") if h else None,
        }

    def status(self) -> dict:
        return {
            "components": self.hub.status if self.hub else {},
            "capabilities": self.capabilities(),
            "error": self.load_error,
        }

    async def load(self) -> None:
        s = self.settings
        cfg = HubConfig(
            artifacts_dir=s.artifacts_dir,
            use_torch=s.ml_mode == "full",
            device=s.device,
            llm_provider=s.llm_provider,
            llm_base=s.llm_base,
            llm_adapter=s.llm_adapter,
            openai_base_url=s.openai_base_url,
            openai_model=s.openai_model,
            openai_api_key=s.openai_api_key,
        )
        try:
            self.hub = await asyncio.get_running_loop().run_in_executor(
                self._perception_pool, lambda: ModelHub(cfg).load()
            )
            log.info("models ready: %s", {k: v.get("source") for k, v in self.hub.status.items()})
        except Exception as exc:  # pragma: no cover - surfaced through /api/system/models
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.exception("model hub failed to load")

    def _require(self) -> ModelHub:
        if self.hub is None:
            raise ModelsNotReady("Aura's models are still loading - try again in a moment.")
        return self.hub

    async def _perceive(self, fn, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(
            self._perception_pool, partial(fn, *args, **kwargs)
        )

    async def transcribe(self, wav: np.ndarray) -> dict | None:
        hub = self._require()
        return await self._perceive(hub.stt.transcribe, wav) if hub.stt else None

    async def face(self, jpeg: bytes) -> dict | None:
        hub = self._require()
        return await self._perceive(hub.fer.predict_jpeg, jpeg) if hub.fer else None

    async def analyze(
        self, text: str, state: TurnState, audio: np.ndarray | None, face: dict | None
    ) -> dict:
        hub = self._require()
        return await self._perceive(analyze_turn, hub, text, state, audio, face)

    @property
    def llm_name(self) -> str:
        return self.hub.llm.name if self.hub and self.hub.llm else "unavailable"

    async def stream(self, messages: list[dict], *, use_adapter: bool = True) -> AsyncIterator[str]:
        hub = self._require()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker() -> None:
            try:
                for chunk in hub.llm.stream(
                    messages,
                    use_adapter=use_adapter,
                    max_new_tokens=self.settings.max_new_tokens,
                    temperature=self.settings.temperature,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _END)

        future = loop.run_in_executor(self._llm_pool, worker)
        while (item := await queue.get()) is not _END:
            if isinstance(item, Exception):
                raise item
            yield item
        await future

    async def complete(self, messages: list[dict], *, use_adapter: bool = True) -> str:
        return "".join([c async for c in self.stream(messages, use_adapter=use_adapter)]).strip()

    def shutdown(self) -> None:
        self._llm_pool.shutdown(wait=False, cancel_futures=True)
        self._perception_pool.shutdown(wait=False, cancel_futures=True)
