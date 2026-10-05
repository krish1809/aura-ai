"""Aggregated views over what Aura perceived: mood over time, common causes, topics."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Query
from sqlalchemy import select

from aura_api.deps import CurrentUser, Session
from aura_api.models import Conversation, Message

router = APIRouter(prefix="/api/insights", tags=["insights"])

# Rough valence used to draw a single "mood" line (positive = pleasant).
VALENCE = {
    "joy": 1.0,
    "surprise": 0.3,
    "neutral": 0.0,
    "fear": -0.7,
    "sadness": -0.8,
    "anger": -0.7,
    "disgust": -0.6,
}


@router.get("")
async def insights(
    user: CurrentUser, session: Session, days: int = Query(30, ge=1, le=365)
) -> dict:
    since = datetime.now(UTC) - timedelta(days=days)
    rows = (
        await session.execute(
            select(Message, Conversation.title)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Conversation.user_id == user.id, Message.role == "user", Message.created_at >= since
            )
            .order_by(Message.created_at)
        )
    ).all()

    timeline, emotions, causes, topics, strategies, modalities = (
        [],
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    for msg, title in rows:
        a = msg.analysis or {}
        emo = (a.get("emotion") or {}).get("emotion")
        if not emo:
            continue
        conf = (a.get("emotion") or {}).get("confidence", 0.0)
        timeline.append(
            {
                "at": msg.created_at.isoformat(),
                "emotion": emo,
                "confidence": conf,
                "valence": round(VALENCE.get(emo, 0.0) * conf, 3),
                "conversation_id": msg.conversation_id,
                "conversation": title,
            }
        )
        emotions[emo] += 1
        for m in (a.get("emotion") or {}).get("modalities", []):
            modalities[m] += 1
        if (a.get("cause") or {}).get("from") == "this message":
            causes[a["cause"]["text"]] += 1
        if (t := (a.get("topic") or {}).get("topic")) and t != "other":
            topics[t] += 1
        if s := (a.get("strategy") or {}).get("strategy"):
            strategies[s] += 1

    return {
        "days": days,
        "messages": len(timeline),
        "timeline": timeline,
        "emotions": dict(emotions.most_common()),
        "top_causes": [{"text": t, "count": c} for t, c in causes.most_common(8)],
        "topics": dict(topics.most_common()),
        "strategies": dict(strategies.most_common()),
        "modalities": dict(modalities),
    }
