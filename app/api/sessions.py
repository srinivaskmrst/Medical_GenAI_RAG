"""Chat session history endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import SessionMessagesResponse, SessionSummary
from app.rag.chat_history import get_session_messages, list_sessions

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionSummary])
def get_sessions() -> list[SessionSummary]:
    """List every known conversation, most recently active first."""
    return [SessionSummary(**entry) for entry in list_sessions()]


@router.get("/{session_id}/messages", response_model=SessionMessagesResponse)
def get_messages(session_id: str) -> SessionMessagesResponse:
    """Return the full chronological transcript for one conversation."""
    return SessionMessagesResponse(session_id=session_id, messages=get_session_messages(session_id))
