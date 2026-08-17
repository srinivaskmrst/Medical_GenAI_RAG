"""Semantic retrieval over vector embeddings."""

from __future__ import annotations

from typing import Any

from app.embeddings import get_embedding_provider
from app.vectorstore import get_vector_store


class SemanticRetriever:
    """Retrieve semantically relevant chunks using embeddings and vector similarity."""

    def __init__(self, top_k: int = 5, embedding_provider: Any | None = None, vector_store: Any | None = None) -> None:
        self.top_k = top_k
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.vector_store = vector_store or get_vector_store()

    def _normalize_result(self, item: dict[str, Any]) -> dict[str, Any]:
        payload = item.get("payload") or {}
        metadata = dict(payload.get("metadata") or {})
        normalized = {**payload, **item}
        normalized["metadata"] = metadata
        normalized.setdefault("document_id", payload.get("document_id") or metadata.get("document_id"))
        normalized.setdefault("chunk_id", payload.get("chunk_id") or metadata.get("chunk_id") or item.get("id"))
        normalized.setdefault("text", payload.get("text") or metadata.get("text"))
        normalized.setdefault("source", payload.get("source") or metadata.get("source"))
        normalized.setdefault("page_number", payload.get("page_number") or metadata.get("page_number"))
        normalized.setdefault("document_type", payload.get("document_type") or metadata.get("document_type"))
        normalized.setdefault("access_control", payload.get("access_control") or metadata.get("access_control"))
        normalized.setdefault("score", float(item.get("score", 0.0)))
        return normalized

    def retrieve(self, query: str, filters: dict[str, Any] | None = None, top_k: int | None = None) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query must not be empty")
        vector = self.embedding_provider.embed_query(query)
        limit = top_k or self.top_k
        raw_results = self.vector_store.search(vector=vector, top_k=limit, filters=filters)
        return [self._normalize_result(item) for item in raw_results]
