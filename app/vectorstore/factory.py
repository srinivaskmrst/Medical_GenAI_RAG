"""Factory helpers for vector store backends."""

from __future__ import annotations

from functools import lru_cache

from app.config.settings import get_settings
from app.embeddings.factory import get_embedding_provider
from app.vectorstore.base import VectorStore
from app.vectorstore.qdrant_store import QdrantVectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return the configured main-collection vector store (singleton)."""
    settings = get_settings()
    dimension = get_embedding_provider().get_dimension()
    return QdrantVectorStore(
        collection_name=settings.qdrant_collection,
        vector_size=dimension,
        vector_name=settings.qdrant_collection,
        distance="euclid",
        keyword_index_fields=[
            "document_id",
            "document_version",
            "document_type",
            "source",
            "department",
            "tenant_id",
            "access_control",
        ],
    )


@lru_cache(maxsize=1)
def get_cache_vector_store() -> VectorStore:
    """Return the semantic-cache vector store (singleton)."""
    settings = get_settings()
    dimension = get_embedding_provider().get_dimension()
    return QdrantVectorStore(
        collection_name=settings.qdrant_cache_collection,
        vector_size=dimension,
        vector_name=settings.qdrant_cache_collection,
        distance="cosine",
    )


@lru_cache(maxsize=1)
def get_long_term_memory_vector_store() -> VectorStore:
    """Return the cross-session long-term memory vector store (singleton)."""
    settings = get_settings()
    dimension = get_embedding_provider().get_dimension()
    return QdrantVectorStore(
        collection_name=settings.qdrant_long_collection,
        vector_size=dimension,
        vector_name=settings.qdrant_long_collection,
        distance="cosine",
        keyword_index_fields=["session_id"],
    )


__all__ = [
    "VectorStore",
    "get_vector_store",
    "get_cache_vector_store",
    "get_long_term_memory_vector_store",
    "QdrantVectorStore",
]
