"""The per-turn orchestrator: Sense -> Understand -> Respond -> Remember.

Events sent to the client (all JSON objects with a "type"):
    transcript  {text}                      voice input was transcribed
    analysis    {message}                   the stored user message incl. what Aura perceived
    safety      {level, resources}          crisis language detected
    token       {text}                      streamed reply chunk
    done        {message, conversation}     the stored assistant message
    error       {detail}
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy import select

from aura_api.config import Settings
from aura_api.db import Database
from aura_api.models import Conversation, Message
from aura_api.schemas import MessageOut
from aura_api.services import memory
from aura_api.services.ml import MLService
from aura_ml.audio import SAMPLE_RATE, load_audio
from aura_ml.pipeline import TurnState, prompt_context
from aura_ml.prompts import build_messages
from aura_ml.safety import CRISIS_RESPONSE, DEFAULT_RESOURCES

log = logging.getLogger("aura_api.chat")
Send = Callable[[dict], Awaitable[None]]
FALLBACK_REPLY = "I'm here with you. Could you tell me a little more about what's on your mind?"


class TurnError(Exception):
    """A problem with the user's input that should be reported back, not logged as a crash."""


@dataclass
class TurnInput:
    text: str = ""
    audio_b64: str | None = None


def message_json(msg: Message) -> dict:
    return MessageOut.model_validate(msg).model_dump(mode="json")


def make_title(text: str, max_words: int = 7) -> str:
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    title = " ".join(words[:max_words])
    return (title + ("…" if len(words) > max_words else ""))[:120] or "New conversation"


def clean_reply(text: str) -> str:
    text = re.sub(r"^\s*(?:aura|assistant)\s*:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


class ChatOrchestrator:
    def __init__(self, ml: MLService, db: Database, settings: Settings):
        self.ml, self.db, self.settings = ml, db, settings
        self._background: set[asyncio.Task] = set()

    def _decode_audio(self, audio_b64: str):
        try:
            raw = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise TurnError("Audio could not be decoded.") from exc
        try:
            wav = load_audio(raw)
        except Exception as exc:
            raise TurnError("Unsupported audio format - send 16-bit PCM WAV.") from exc
        if len(wav) > self.settings.max_audio_seconds * SAMPLE_RATE:
            raise TurnError(
                f"Voice messages are limited to {self.settings.max_audio_seconds} seconds."
            )
        return wav

    async def handle(
        self, user_id: str, conversation_id: str, inp: TurnInput, face: dict | None, send: Send
    ) -> None:
        text = (inp.text or "").strip()
        wav = self._decode_audio(inp.audio_b64) if inp.audio_b64 else None
        if wav is not None and not text:
            transcript = await self.ml.transcribe(wav)
            if transcript is None:
                raise TurnError(
                    "Voice input needs the speech-to-text model (run the backend with AURA_ML_MODE=full)."
                )
            text = transcript["text"].strip()
            await send({"type": "transcript", "text": text})
        if not text:
            raise TurnError(
                "I couldn't hear any words in that recording."
                if wav is not None
                else "Message is empty."
            )
        if len(text) > self.settings.max_message_chars:
            raise TurnError(
                f"Messages are limited to {self.settings.max_message_chars} characters."
            )

        async with self.db.sessionmaker() as session:
            conv = await session.get(Conversation, conversation_id)
            if conv is None or conv.user_id != user_id:
                raise TurnError("Conversation not found.")
            history = list(
                await session.scalars(
                    select(Message)
                    .where(Message.conversation_id == conv.id)
                    .order_by(Message.created_at.desc())
                    .limit(self.settings.history_messages)
                )
            )[::-1]
            last_assistant = next((m for m in reversed(history) if m.role == "assistant"), None)
            user_so_far = " ".join(m.content for m in history if m.role == "user")
            state = TurnState(
                user_text_so_far=user_so_far[-2000:],
                prev_reply=last_assistant.content if last_assistant else "",
                prev_strategy=((last_assistant.meta or {}).get("strategy") or "none")
                if last_assistant
                else "none",
                turn_index=conv.user_turns + 1,
                last_cause=conv.last_cause,
            )

            # Sense + Understand
            analysis = await self.ml.analyze(text, state, wav, face)
            user_msg = Message(
                conversation_id=conv.id,
                role="user",
                content=text,
                modality="voice" if wav is not None else "text",
                analysis=analysis,
            )
            session.add(user_msg)
            conv.user_turns += 1
            if analysis["cause"]["from"] == "this message":
                conv.last_cause = analysis["cause"]["text"]
            if conv.title == "New conversation":
                conv.title = make_title(text)
            await session.commit()
            await send({"type": "analysis", "message": message_json(user_msg)})

            # Respond
            level = analysis["safety"]["level"]
            if level == "crisis":
                await send({"type": "safety", "level": level, "resources": DEFAULT_RESOURCES})
                reply, model_name = CRISIS_RESPONSE, "safety-template"
                for sentence in re.split(r"(?<=[.!?]) ", reply):
                    await send({"type": "token", "text": sentence + " "})
            else:
                notes = await memory.load_notes(session, user_id)
                ctx = prompt_context(analysis, notes)
                messages = build_messages(
                    text,
                    [{"role": m.role, "content": m.content} for m in history],
                    ctx,
                    max_history=self.settings.history_messages,
                )
                parts: list[str] = []
                async for chunk in self.ml.stream(messages):
                    parts.append(chunk)
                    await send({"type": "token", "text": chunk})
                reply, model_name = clean_reply("".join(parts)) or FALLBACK_REPLY, self.ml.llm_name

            strategy = (
                (analysis.get("strategy") or {}).get("strategy") if level != "crisis" else None
            )
            bot_msg = Message(
                conversation_id=conv.id,
                role="assistant",
                content=reply,
                meta={"strategy": strategy, "model": model_name, "safety": level},
            )
            session.add(bot_msg)
            await session.commit()

            # Remember - owned by the orchestrator so it survives the client disconnecting
            summarize = conv.user_turns - conv.summarized_turns >= self.settings.summarize_every
            self._spawn(self._remember(user_id, conv.id, analysis, summarize))
            await send(
                {
                    "type": "done",
                    "message": message_json(bot_msg),
                    "conversation": {"id": conv.id, "title": conv.title},
                }
            )

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _remember(
        self, user_id: str, conversation_id: str, analysis: dict, summarize: bool
    ) -> None:
        async with self.db.sessionmaker() as session:
            try:
                await memory.update_graph(session, user_id, analysis)
                await session.commit()
            except Exception:
                await session.rollback()
                log.exception("knowledge-graph update failed")
        if summarize:
            await memory.refresh_summary(self.db.sessionmaker, self.ml, user_id, conversation_id)

    async def drain(self) -> None:
        """Wait for background memory work (used by tests and on shutdown)."""
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)
