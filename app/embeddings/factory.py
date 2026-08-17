"""Embedding provider factory."""

from __future__ import annotations

from functools import lru_cache

from app.config.settings import get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.providers.local import LocalEmbeddingProvider
from app.embeddings.providers.google import GoogleEmbedding
from app.embeddings.providers.openai import OpenAIEmbedding


@lru_cache(maxsize=8)
def get_embedding_provider(provider: str | None = None) -> EmbeddingProvider:
    """Return a configured embedding provider instance."""
    settings = get_settings()
    selected = (provider or settings.embedding_provider).lower()

    if selected in {"local", "sentence_transformers", "sentence-transformers", "sentencetransformer"}:
        return LocalEmbeddingProvider(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
    if selected == "openai":
        return OpenAIEmbedding(model_name=settings.embedding_model)
    if selected in {"google", "gemini"}:
        return GoogleEmbedding(model_name=settings.embedding_model)
    raise ValueError(f"Unsupported embedding provider: {provider or settings.embedding_provider}")


__all__ = ["EmbeddingProvider", "get_embedding_provider", "LocalEmbeddingProvider", "OpenAIEmbedding", "GoogleEmbedding"]
