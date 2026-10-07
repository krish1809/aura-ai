"""Real-time chat over WebSocket.

Client -> server messages:
    {"type": "message", "text": "...", "audio": "<base64 16 kHz PCM WAV>" | null}
    {"type": "frame", "image": "<base64 JPEG>"}       webcam frame (~1 per second while the camera is on)
    {"type": "ping"}
Server -> client: see aura_api.services.chat, plus {"type": "face", ...} after each frame,
{"type": "ready", "capabilities": {...}} on connect and {"type": "pong"}.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from aura_api.models import Conversation
from aura_api.security import decode_token
from aura_api.services.chat import TurnError, TurnInput
from aura_api.services.face import FaceTracker
from aura_api.services.ml import ModelsNotReady
from aura_ml.safety import assess

router = APIRouter()
MAX_QUEUED_TURNS = 3
log = logging.getLogger("aura_api.ws")


@router.websocket("/api/ws/conversations/{conversation_id}")
async def chat_socket(ws: WebSocket, conversation_id: str) -> None:
    app = ws.app
    settings = app.state.settings
    token = ws.cookies.get(settings.cookie_name) or ws.query_params.get("token")
    user_id = decode_token(token, settings) if token else None
    if not user_id:
        await ws.close(code=4401, reason="not signed in")
        return
    async with app.state.db.sessionmaker() as session:
        conv = await session.get(Conversation, conversation_id)
        if conv is None or conv.user_id != user_id:
            await ws.close(code=4404, reason="conversation not found")
            return

    await ws.accept()
    send_lock = asyncio.Lock()
    tracker = FaceTracker()
    turn_task: asyncio.Task | None = None
    frame_task: asyncio.Task | None = None

    async def send(event: dict) -> None:
        async with send_lock:
            await ws.send_json(event)

    queued = 0

    async def run_turn(inp: TurnInput, previous: asyncio.Task | None) -> None:
        # Turns run strictly in order: a message sent while Aura is still replying waits
        # for the previous turn instead of being dropped.
        nonlocal queued
        try:
            if previous is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await previous
            await app.state.chat.handle(user_id, conversation_id, inp, tracker.current(), send)
        except (TurnError, ModelsNotReady) as exc:
            await send({"type": "error", "detail": str(exc)})
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("turn failed")
            await send(
                {
                    "type": "error",
                    "detail": "Something went wrong while replying. Please try again.",
                }
            )
        finally:
            queued -= 1

    async def run_frame(data: bytes) -> None:
        try:
            result = await app.state.ml.face(data)
            if result is not None:
                tracker.add(result)
                await send({"type": "face", "frame": result, "smoothed": tracker.current()})
        except Exception:
            log.exception("frame analysis failed")

    await send(
        {
            "type": "ready",
            "capabilities": app.state.ml.capabilities() if app.state.ml.ready else None,
        }
    )
    try:
        while True:
            event = await ws.receive_json()
            kind = event.get("type")
            if kind == "message":
                text = event.get("text") or ""
                # a crisis message is never turned away, however busy the connection is
                if queued >= MAX_QUEUED_TURNS and assess(text)["level"] != "crisis":
                    await send({"type": "error", "detail": "Aura is still replying - one moment."})
                    continue
                queued += 1
                turn_task = asyncio.create_task(
                    run_turn(TurnInput(text=text, audio_b64=event.get("audio")), turn_task)
                )
            elif kind == "frame":
                if frame_task and not frame_task.done():
                    continue  # drop frames while the previous one is still being analysed
                try:
                    data = base64.b64decode(event.get("image", ""), validate=True)
                except (binascii.Error, ValueError):
                    continue
                frame_task = asyncio.create_task(run_frame(data))
            elif kind == "ping":
                await send({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        for task in (turn_task, frame_task):
            if task and not task.done():
                task.cancel()
