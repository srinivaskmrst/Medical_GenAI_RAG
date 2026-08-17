"""Dedicated query encoding service for retrieval."""

from __future__ import annotations

from app.embeddings import get_embedding_provider


class QueryEncoder:
    """Validate, normalize, and embed user queries before retrieval."""

    def __init__(self, embedding_provider=None) -> None:
        self.embedding_provider = embedding_provider or get_embedding_provider()

    def normalize(self, query: str) -> str:
        if query is None:
            raise ValueError("Query cannot be None")
        normalized = " ".join(str(query).strip().split())
        if not normalized:
            raise ValueError("Query cannot be empty")
        return normalized

    def encode(self, query: str) -> list[float]:
        cleaned = self.normalize(query)
        return self.embedding_provider.embed_query(cleaned)
