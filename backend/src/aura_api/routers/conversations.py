from __future__ import annotations

from fastapi import APIRouter, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from aura_api.deps import CurrentUser, OwnedConversation, Session
from aura_api.models import Conversation
from aura_api.schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    ConversationUpdate,
)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationOut])
async def list_conversations(user: CurrentUser, session: Session) -> list[Conversation]:
    rows = await session.scalars(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(rows)


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: ConversationCreate, user: CurrentUser, session: Session
) -> Conversation:
    conv = Conversation(user_id=user.id, title=body.title or "New conversation")
    session.add(conv)
    await session.commit()
    return conv


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conv: OwnedConversation, session: Session) -> Conversation:
    return await session.scalar(
        select(Conversation)
        .where(Conversation.id == conv.id)
        .options(selectinload(Conversation.messages))
    )


@router.patch("/{conversation_id}", response_model=ConversationOut)
async def rename_conversation(
    body: ConversationUpdate, conv: OwnedConversation, session: Session
) -> Conversation:
    conv.title = body.title.strip()
    await session.commit()
    return conv


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conv: OwnedConversation, session: Session) -> None:
    await session.delete(conv)
    await session.commit()
