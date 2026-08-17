"""OpenAI-compatible embedding provider placeholder."""

from __future__ import annotations

from app.embeddings.base import EmbeddingProvider


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider stub kept provider-agnostic by interface."""

    def __init__(self, model_name: str = "text-embedding-3-small", **_: object) -> None:
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("OpenAI embedding provider is not implemented yet")

    def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError("OpenAI embedding provider is not implemented yet")

    def get_dimension(self) -> int:
        raise NotImplementedError("OpenAI embedding provider is not implemented yet")
