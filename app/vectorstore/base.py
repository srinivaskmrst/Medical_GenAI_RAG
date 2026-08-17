"""Vector store abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class VectorStore(ABC):
    """Abstract interface for vector database backends."""

    @abstractmethod
    def upsert(self, records: Sequence[dict[str, Any]]) -> list[str]:
        """Insert or update records."""

    @abstractmethod
    def search(self, vector: Sequence[float], top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Search the vector store for the nearest neighbors."""

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """Delete records by id."""

    @abstractmethod
    def delete_by_document(self, document_id: str) -> None:
        """Delete all chunks for a document."""

    @abstractmethod
    def health_check(self) -> bool:
        """Verify backend availability."""
