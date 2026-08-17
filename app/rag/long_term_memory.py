"""Cross-session long-term memory backed by a dedicated Qdrant collection.

Distinct from the semantic answer cache: this stores durable facts/turns
scoped to a session_id (e.g. a returning user or an ongoing case) so later
questions in the same session can be answered with awareness of earlier
context, rather than caching exact-answer lookups for near-duplicate
questions.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from app.vectorstore.factory import get_long_term_memory_vector_store


class LongTermMemory:
    """Stores and retrieves durable per-session conversational context."""

    def __init__(self, vector_store: Any | None = None, top_k: int = 3) -> None:
        self.vector_store = vector_store or get_long_term_memory_vector_store()
        self.top_k = top_k

    def retrieve(self, session_id: str | None, query_vector: list[float]) -> list[dict[str, Any]]:
        if not session_id:
            return []
        results = self.vector_store.search(
            vector=query_vector,
            top_k=self.top_k,
            filters={"session_id": session_id},
        )
        entries = []
        for item in results:
            payload = item.get("payload") or {}
            entries.append(
                {
                    "role": payload.get("role"),
                    "text": payload.get("text"),
                    "created_at": payload.get("created_at"),
                    "score": item.get("score"),
                }
            )
        return entries

    def store(self, session_id: str | None, role: str, text: str, vector: list[float]) -> None:
        if not session_id or not text:
            return
        entry_id = f"{session_id}:{uuid.uuid4()}"
        self.vector_store.upsert(
            [
                {
                    "id": entry_id,
                    "embedding": vector,
                    "metadata": {
                        "session_id": session_id,
                        "role": role,
                        "text": text,
                        "created_at": time.time(),
                    },
                }
            ]
        )
