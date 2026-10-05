"""Long-term memory: the personal knowledge graph and LLM-written notes.

Graph (per user):  person/entity --involved_in--> cause --causes--> emotion
                   cause --part_of--> topic        person --associated_with--> emotion
Notes: every few user turns the base LLM (adapter off) rewrites a short summary of
the conversation; the user's notes are the latest summaries of recent conversations
and are fed back into the system prompt as "What you remember about them".
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aura_api.models import Conversation, GraphEdge, GraphNode, Message, UserMemory
from aura_api.services.ml import MLService
from aura_ml.prompts import summary_messages

log = logging.getLogger("aura_api.memory")
MAX_NOTES_CHARS = 1500


def _key(label: str) -> str:
    return " ".join(label.lower().split())[:160]


async def _node(
    session: AsyncSession, user_id: str, kind: str, label: str, now: datetime
) -> GraphNode:
    label = " ".join(label.split())[:160]
    node = await session.scalar(
        select(GraphNode).where(
            GraphNode.user_id == user_id, GraphNode.kind == kind, GraphNode.key == _key(label)
        )
    )
    if node is None:
        node = GraphNode(
            user_id=user_id, kind=kind, key=_key(label), label=label, mentions=0, first_seen=now
        )
        session.add(node)
        await session.flush()
    node.mentions += 1
    node.last_seen = now
    return node


async def _edge(
    session: AsyncSession,
    user_id: str,
    src: GraphNode,
    dst: GraphNode,
    relation: str,
    now: datetime,
) -> None:
    edge = await session.scalar(
        select(GraphEdge).where(
            GraphEdge.user_id == user_id,
            GraphEdge.source_id == src.id,
            GraphEdge.target_id == dst.id,
            GraphEdge.relation == relation,
        )
    )
    if edge is None:
        session.add(
            GraphEdge(
                user_id=user_id,
                source_id=src.id,
                target_id=dst.id,
                relation=relation,
                weight=1,
                last_seen=now,
            )
        )
    else:
        edge.weight += 1
        edge.last_seen = now


async def update_graph(session: AsyncSession, user_id: str, analysis: dict) -> None:
    now = datetime.now(UTC)
    emotion = analysis["emotion"]["emotion"] if analysis["emotion"].get("modalities") else None
    cause_text = (
        analysis["cause"]["text"] if analysis["cause"].get("from") == "this message" else None
    )
    topic = analysis.get("topic") or {}
    topic_label = (
        topic.get("topic")
        if topic.get("topic") not in (None, "other") and topic.get("confidence", 0) >= 0.4
        else None
    )

    emo_node = (
        await _node(session, user_id, "emotion", emotion, now)
        if emotion and emotion != "neutral"
        else None
    )
    cause_node = (
        await _node(session, user_id, "cause", cause_text[:120], now) if cause_text else None
    )
    topic_node = await _node(session, user_id, "topic", topic_label, now) if topic_label else None

    if cause_node and emo_node:
        await _edge(session, user_id, cause_node, emo_node, "causes", now)
    if cause_node and topic_node:
        await _edge(session, user_id, cause_node, topic_node, "part_of", now)
    for ent in analysis.get("entities", []):
        kind = "person" if ent["label"] in ("RELATION", "PERSON") else "entity"
        ent_node = await _node(session, user_id, kind, ent["text"], now)
        if cause_node:
            await _edge(session, user_id, ent_node, cause_node, "involved_in", now)
        elif emo_node:
            await _edge(session, user_id, ent_node, emo_node, "associated_with", now)


async def load_notes(session: AsyncSession, user_id: str) -> str | None:
    mem = await session.get(UserMemory, user_id)
    return mem.notes if mem and mem.notes.strip() else None


async def refresh_summary(sessionmaker, ml: MLService, user_id: str, conversation_id: str) -> None:
    """Re-summarise one conversation and rebuild the user's notes. Runs in the background."""
    try:
        async with sessionmaker() as session:
            conv = await session.get(Conversation, conversation_id)
            if conv is None:
                return
            msgs = list(
                await session.scalars(
                    select(Message)
                    .where(Message.conversation_id == conv.id)
                    .order_by(Message.created_at.desc())
                    .limit(12)
                )
            )[::-1]
            transcript = [{"role": m.role, "content": m.content} for m in msgs]
            summary = await ml.complete(
                summary_messages(transcript, conv.summary), use_adapter=False
            )
            if not summary:
                return
            conv.summary = summary.strip()[:800]
            conv.summarized_turns = conv.user_turns

            recent = list(
                await session.scalars(
                    select(Conversation)
                    .where(Conversation.user_id == user_id, Conversation.summary.is_not(None))
                    .order_by(Conversation.updated_at.desc())
                    .limit(3)
                )
            )
            notes = "\n".join(c.summary for c in recent[::-1] if c.summary)[-MAX_NOTES_CHARS:]
            mem = await session.get(UserMemory, user_id)
            if mem is None:
                session.add(UserMemory(user_id=user_id, notes=notes))
            else:
                mem.notes = notes
            await session.commit()
    except Exception:  # memory is best-effort; never break the chat
        log.exception("summary refresh failed for conversation %s", conversation_id)
