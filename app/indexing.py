"""Document indexing service for embedding and storing chunks."""

from __future__ import annotations

from typing import Any

from app.embeddings import get_embedding_provider
from app.ingestion.chunker import Chunk
from app.vectorstore import get_vector_store


def chunk_to_record(chunk: Chunk) -> dict[str, Any]:
    """Adapt a DocumentChunker Chunk (metadata as a pydantic model) into the
    flat record shape DocumentIndexingService.index_chunks() expects."""
    metadata = chunk["metadata"]
    return {
        "document_id": metadata.document_id,
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "source": metadata.source,
        "page_number": metadata.page_number,
        "document_type": metadata.document_type,
        "metadata": metadata.model_dump(),
    }


class DocumentIndexingService:
    """Embed chunks and upsert them into the configured vector store."""

    def __init__(
        self,
        embedding_provider: Any | None = None,
        vector_store: Any | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.vector_store = vector_store or get_vector_store()

    def _build_record(self, chunk: dict[str, Any], embedding: list[float]) -> dict[str, Any]:
        metadata = dict(chunk.get("metadata") or {})
        metadata.setdefault("document_id", chunk.get("document_id"))
        metadata.setdefault("document_version", metadata.get("document_version") or chunk.get("document_version"))
        metadata.setdefault("document_type", chunk.get("document_type"))
        metadata.setdefault("source", chunk.get("source"))
        metadata.setdefault("page_number", chunk.get("page_number"))
        metadata.setdefault("department", metadata.get("department"))
        metadata.setdefault("tenant_id", metadata.get("tenant_id"))
        metadata.setdefault("access_control", metadata.get("access_control"))
        metadata.setdefault("created_at", metadata.get("created_at"))
        metadata.setdefault("chunk_id", chunk.get("chunk_id"))

        return {
            "id": str(chunk.get("chunk_id") or chunk.get("id") or chunk.get("document_id")),
            "document_id": chunk.get("document_id"),
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text", ""),
            "embedding": embedding,
            "source": chunk.get("source"),
            "page_number": chunk.get("page_number"),
            "document_type": chunk.get("document_type"),
            "metadata": metadata,
            "created_at": metadata.get("created_at"),
        }

    def index_chunks(self, chunks: list[dict[str, Any]]) -> list[str]:
        if not chunks:
            return []
        texts = [str(chunk.get("text", "")) for chunk in chunks]
        embeddings = self.embedding_provider.embed_documents(texts)
        records = [self._build_record(chunk, embedding) for chunk, embedding in zip(chunks, embeddings)]
        return self.vector_store.upsert(records)

    def delete_document(self, document_id: str) -> None:
        self.vector_store.delete_by_document(document_id)

    def reindex_document(self, document_id: str, chunks: list[dict[str, Any]]) -> list[str]:
        self.delete_document(document_id)
        return self.index_chunks(chunks)
