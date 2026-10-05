"""What Aura remembers: long-term notes and the personal knowledge graph. Users can see and erase both."""

from __future__ import annotations

from fastapi import APIRouter, status
from sqlalchemy import delete, select

from aura_api.deps import CurrentUser, Session
from aura_api.models import Conversation, GraphEdge, GraphNode, UserMemory
from aura_api.schemas import GraphOut, MemoryNotes

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("/notes", response_model=MemoryNotes)
async def get_notes(user: CurrentUser, session: Session) -> MemoryNotes:
    mem = await session.get(UserMemory, user.id)
    return MemoryNotes(notes=mem.notes if mem else "", updated_at=mem.updated_at if mem else None)


@router.get("/graph", response_model=GraphOut)
async def get_graph(user: CurrentUser, session: Session) -> GraphOut:
    nodes = list(await session.scalars(select(GraphNode).where(GraphNode.user_id == user.id)))
    edges = list(await session.scalars(select(GraphEdge).where(GraphEdge.user_id == user.id)))
    return GraphOut(
        nodes=[
            {
                "id": n.id,
                "kind": n.kind,
                "label": n.label,
                "mentions": n.mentions,
                "first_seen": n.first_seen.isoformat(),
                "last_seen": n.last_seen.isoformat(),
            }
            for n in nodes
        ],
        links=[
            {
                "source": e.source_id,
                "target": e.target_id,
                "relation": e.relation,
                "weight": e.weight,
            }
            for e in edges
        ],
    )


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def forget_everything(user: CurrentUser, session: Session) -> None:
    """Erase long-term notes, the knowledge graph and conversation summaries (keeps chat history)."""
    await session.execute(delete(GraphEdge).where(GraphEdge.user_id == user.id))
    await session.execute(delete(GraphNode).where(GraphNode.user_id == user.id))
    await session.execute(delete(UserMemory).where(UserMemory.user_id == user.id))
    for conv in await session.scalars(select(Conversation).where(Conversation.user_id == user.id)):
        conv.summary, conv.summarized_turns = None, conv.user_turns
    await session.commit()
