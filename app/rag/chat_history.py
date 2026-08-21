"""Cross-session chat history listing, backed by the long-term-memory Qdrant
collection that RAGService already writes every user/assistant turn into.

Distinct from LongTermMemory.retrieve (top-k semantic recall used to ground
new answers): this reconstructs the full, chronologically ordered transcript
per session_id so the UI can list and replay past conversations verbatim.
"""

from __future__ import annotations

from typing import Any

from app.vectorstore.factory import get_long_term_memory_vector_store

TITLE_MAX_LENGTH = 80


def list_sessions() -> list[dict[str, Any]]:
    """Group every stored turn by session_id into a recency-ordered summary list."""
    points = get_long_term_memory_vector_store().scroll_all()

    sessions: dict[str, dict[str, Any]] = {}
    for point in points:
        payload = point.get("payload") or {}
        session_id = payload.get("session_id")
        if not session_id:
            continue
        created_at = payload.get("created_at") or 0
        entry = sessions.setdefault(
            session_id,
            {"session_id": session_id, "title": "", "last_updated": 0.0, "message_count": 0, "_first_user_at": None},
        )
        entry["message_count"] += 1
        entry["last_updated"] = max(entry["last_updated"], created_at)
        if payload.get("role") == "user" and (entry["_first_user_at"] is None or created_at < entry["_first_user_at"]):
            entry["_first_user_at"] = created_at
            text = payload.get("text") or ""
            entry["title"] = text[:TITLE_MAX_LENGTH]

    summaries = list(sessions.values())
    for entry in summaries:
        entry.pop("_first_user_at", None)
        if not entry["title"]:
            entry["title"] = "Untitled conversation"
    summaries.sort(key=lambda entry: entry["last_updated"], reverse=True)
    return summaries


def get_session_messages(session_id: str) -> list[dict[str, Any]]:
    """Return the full chronological transcript for one session."""
    points = get_long_term_memory_vector_store().scroll_all(filters={"session_id": session_id})
    messages = [
        {
            "role": (point.get("payload") or {}).get("role"),
            "text": (point.get("payload") or {}).get("text"),
            "created_at": (point.get("payload") or {}).get("created_at"),
        }
        for point in points
    ]
    messages.sort(key=lambda item: item.get("created_at") or 0)
    return messages
