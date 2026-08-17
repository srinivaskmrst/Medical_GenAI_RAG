"""Cross-encoder reranking utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

from app.config.settings import get_settings


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


class Reranker:
    """Reorders retrieved results by relevance using a cross-encoder model."""

    def __init__(self, model_name: str | None = None, enabled: bool | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.enabled = settings.reranker_enabled if enabled is None else enabled

    def rerank(self, query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not documents:
            return []
        if not self.enabled:
            return documents

        model = _load_cross_encoder(self.model_name)
        pairs = [(query, str(doc.get("text", ""))) for doc in documents]
        scores = model.predict(pairs)

        reranked = [{**doc, "score": float(score)} for doc, score in zip(documents, scores)]
        reranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return reranked
