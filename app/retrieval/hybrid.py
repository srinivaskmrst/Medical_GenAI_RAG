"""Hybrid retrieval combining semantic and keyword scores."""

from __future__ import annotations

from typing import Any

from app.config.settings import get_settings


class HybridRetriever:
    """Fuse semantic and keyword retrieval results using configurable weighting."""

    def __init__(
        self,
        semantic_retriever: Any | None = None,
        keyword_retriever: Any | None = None,
        semantic_weight: float | None = None,
        keyword_weight: float | None = None,
        top_k: int | None = None,
    ) -> None:
        settings = get_settings()
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.semantic_weight = semantic_weight if semantic_weight is not None else settings.semantic_weight
        self.keyword_weight = keyword_weight if keyword_weight is not None else settings.keyword_weight
        self.top_k = top_k if top_k is not None else settings.top_k

    def retrieve(self, query: str, filters: dict[str, Any] | None = None, top_k: int | None = None) -> list[dict[str, Any]]:
        semantic_results = []
        keyword_results = []

        if self.semantic_retriever is not None:
            semantic_results = self.semantic_retriever.retrieve(query=query, filters=filters, top_k=top_k or self.top_k)
        if self.keyword_retriever is not None:
            keyword_results = self.keyword_retriever.retrieve(query=query, filters=filters, top_k=top_k or self.top_k)

        merged: dict[str, dict[str, Any]] = {}
        for item in semantic_results:
            key = str(item.get("id") or item.get("chunk_id") or item.get("document_id"))
            merged[key] = {**item, "score": float(item.get("score", 0.0)) * self.semantic_weight}
        for item in keyword_results:
            key = str(item.get("id") or item.get("chunk_id") or item.get("document_id"))
            existing = merged.get(key, {**item, "score": 0.0})
            existing["score"] = existing.get("score", 0.0) + float(item.get("score", 0.0)) * self.keyword_weight
            merged[key] = existing

        ranked = sorted(merged.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked[: top_k or self.top_k]
