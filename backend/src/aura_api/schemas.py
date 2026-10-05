from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterIn(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=8, max_length=128)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    name: str
    created_at: datetime


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    user_turns: int


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    modality: str
    analysis: dict | None = None
    meta: dict | None = None
    created_at: datetime


class ConversationDetail(ConversationOut):
    summary: str | None
    messages: list[MessageOut]


class MemoryNotes(BaseModel):
    notes: str
    updated_at: datetime | None


class GraphOut(BaseModel):
    nodes: list[dict]
    links: list[dict]
