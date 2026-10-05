"""Database tables.

users ─┬─< conversations ─< messages        (chat history + per-message analysis)
       ├── user_memories                    (long-term notes written by the LLM)
       └─< graph_nodes ─< graph_edges       (personal knowledge graph: people, topics,
                                             causes, emotions and how they connect)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from aura_api.db import Base


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every backend (SQLite drops tzinfo on the way back)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(80))
    password_hash: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_now)

    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(120), default="New conversation")
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    last_cause: Mapped[str | None] = mapped_column(Text, default=None)
    user_turns: Mapped[int] = mapped_column(Integer, default=0)
    summarized_turns: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_now, onupdate=_now, index=True
    )

    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    modality: Mapped[str] = mapped_column(String(16), default="text")  # text | voice
    analysis: Mapped[dict | None] = mapped_column(
        JSON, default=None
    )  # user turns: what Aura perceived
    meta: Mapped[dict | None] = mapped_column(
        JSON, default=None
    )  # assistant turns: strategy, model, safety
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_now, index=True)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class UserMemory(Base):
    __tablename__ = "user_memories"

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    notes: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_now, onupdate=_now)


class GraphNode(Base):
    __tablename__ = "graph_nodes"
    __table_args__ = (UniqueConstraint("user_id", "kind", "key", name="uq_node"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # person | topic | cause | emotion | entity
    key: Mapped[str] = mapped_column(String(160))  # normalised label used for de-duplication
    label: Mapped[str] = mapped_column(String(160))
    mentions: Mapped[int] = mapped_column(Integer, default=1)
    first_seen: Mapped[datetime] = mapped_column(UTCDateTime(), default=_now)
    last_seen: Mapped[datetime] = mapped_column(UTCDateTime(), default=_now)


class GraphEdge(Base):
    __tablename__ = "graph_edges"
    __table_args__ = (
        UniqueConstraint("user_id", "source_id", "target_id", "relation", name="uq_edge"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("graph_nodes.id", ondelete="CASCADE"))
    target_id: Mapped[int] = mapped_column(ForeignKey("graph_nodes.id", ondelete="CASCADE"))
    relation: Mapped[str] = mapped_column(String(32))  # causes | felt_during | involves | about
    weight: Mapped[int] = mapped_column(Integer, default=1)
    last_seen: Mapped[datetime] = mapped_column(UTCDateTime(), default=_now)
