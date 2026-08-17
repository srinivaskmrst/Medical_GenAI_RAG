"""Local embedding provider backed by sentence-transformers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.embeddings.base import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    """Lightweight local embedding provider using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        batch_size: int = 32,
        normalize_embeddings: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension: int | None = None

    def _validate_dimension(self, vector: Sequence[float]) -> list[float]:
        values = [float(item) for item in vector]
        if self.dimension is None:
            self.dimension = len(values)
        if len(values) != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch: expected "
                f"{self.dimension}, received {len(values)}"
            )
        return values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = np.asarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
        )
        if len(embeddings) == 0:
            return []
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        result: list[list[float]] = []
        for row in embeddings:
            result.append(self._validate_dimension(row))
        return result

    def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise ValueError("Query must not be empty")
        embedding = np.asarray(
            self.model.encode(
                [query],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
        )
        values = embedding[0] if embedding.ndim > 1 else embedding
        return self._validate_dimension(values)

    def get_dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())
