"""FastAPI dependencies shared by the routers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from aura_api.config import Settings
from aura_api.models import Conversation, User
from aura_api.security import decode_token


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.db.sessionmaker() as session:
        yield session


Session = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def token_from_request(request: Request, settings: Settings) -> str | None:
    token = request.cookies.get(settings.cookie_name)
    auth = request.headers.get("authorization", "")
    if not token and auth.lower().startswith("bearer "):
        token = auth[7:]
    return token


async def current_user(request: Request, session: Session, settings: SettingsDep) -> User:
    token = token_from_request(request, settings)
    user_id = decode_token(token, settings) if token else None
    user = await session.get(User, user_id) if user_id else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not signed in")
    return user


CurrentUser = Annotated[User, Depends(current_user)]


async def owned_conversation(
    conversation_id: str, user: CurrentUser, session: Session
) -> Conversation:
    conv = await session.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return conv


OwnedConversation = Annotated[Conversation, Depends(owned_conversation)]
