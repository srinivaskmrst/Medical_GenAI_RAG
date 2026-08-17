"""Semantic answer cache backed by a dedicated Qdrant collection."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from app.config.settings import get_settings
from app.vectorstore.factory import get_cache_vector_store


class SemanticCache:
    """Cache question/answer pairs and serve near-duplicate questions instantly."""

    def __init__(self, vector_store: Any | None = None, similarity_threshold: float | None = None, enabled: bool | None = None) -> None:
        settings = get_settings()
        self.vector_store = vector_store or get_cache_vector_store()
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else settings.cache_similarity_threshold
        )
        self.enabled = settings.cache_enabled if enabled is None else enabled

    def lookup(self, query_vector: list[float]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        results = self.vector_store.search(vector=query_vector, top_k=1)
        if not results:
            return None
        top = results[0]
        if float(top.get("score", 0.0)) < self.similarity_threshold:
            return None
        payload = top.get("payload") or {}
        retrieval_metadata = dict(payload.get("retrieval_metadata") or {})
        retrieval_metadata["cache_hit"] = True
        retrieval_metadata["cache_similarity"] = float(top.get("score", 0.0))
        retrieval_metadata["cached_query"] = payload.get("query")
        return {
            "answer": payload.get("answer", ""),
            "sources": payload.get("sources", []),
            "retrieval_metadata": retrieval_metadata,
            "usage": payload.get("usage", {}),
            "confidence": payload.get("confidence", 0.0),
        }

    def store(
        self,
        query: str,
        query_vector: list[float],
        answer: str,
        sources: list[dict[str, Any]],
        retrieval_metadata: dict[str, Any],
        usage: dict[str, Any],
        confidence: float,
    ) -> None:
        if not self.enabled:
            return
        cache_id = hashlib.sha256(query.encode("utf-8")).hexdigest()
        self.vector_store.upsert(
            [
                {
                    "id": cache_id,
                    "embedding": query_vector,
                    "metadata": {
                        "query": query,
                        "answer": answer,
                        "sources": sources,
                        "retrieval_metadata": retrieval_metadata,
                        "usage": usage,
                        "confidence": confidence,
                        "created_at": time.time(),
                    },
                }
            ]
        )
