"""Qdrant vector store implementation."""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config.settings import get_settings
from app.vectorstore.base import VectorStore


def _to_point_id(raw_id: Any) -> str:
    """Qdrant point IDs must be an unsigned integer or a UUID.

    Our chunk/cache IDs are arbitrary strings (e.g. "README_0"), so
    non-conforming IDs are deterministically mapped to a UUID5 — re-indexing
    the same source ID always yields the same point ID (idempotent upserts).
    The original ID is preserved separately in the payload (chunk_id).
    """
    text = str(raw_id)
    if text.isdigit():
        return text
    try:
        uuid.UUID(text)
        return text
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, text))


@lru_cache(maxsize=1)
def _default_client() -> QdrantClient:
    """Shared Qdrant client for the process.

    Qdrant Cloud (QDRANT_ENDPOINT) is preferred when configured; otherwise
    falls back to embedded local-disk mode or a plain server URL. Cached as a
    singleton because embedded/local-disk mode locks its storage path per
    client instance, and multiple collections (main + cache) must share one
    open client from the same process.
    """
    settings = get_settings()
    if settings.qdrant_endpoint:
        return QdrantClient(url=settings.qdrant_endpoint, api_key=settings.qdrant_api_key)
    if settings.qdrant_mode == "embedded":
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


class QdrantVectorStore(VectorStore):
    """Production-oriented Qdrant-backed vector store."""

    def __init__(
        self,
        collection_name: str | None = None,
        vector_size: int = 768,
        url: str | None = None,
        path: str | None = None,
        client: QdrantClient | None = None,
        vector_name: str | None = None,
        distance: str = "cosine",
        keyword_index_fields: Sequence[str] | None = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = collection_name or settings.qdrant_collection
        self.vector_size = vector_size
        self.url = url
        self.vector_name = vector_name
        self.distance = distance
        self.keyword_index_fields = list(keyword_index_fields or [])

        if client is not None:
            self.client = client
        elif path is not None:
            self.client = QdrantClient(path=path)
        elif url is not None:
            self.client = QdrantClient(url=url)
        else:
            self.client = _default_client()

        self._ensure_collection()
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Qdrant (notably Cloud) rejects filtering on a field with no
        payload index — create one (idempotent) for every field we filter
        results by."""
        for field in self.keyword_index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def _ensure_collection(self) -> None:
        try:
            collections = self.client.get_collections().collections
        except Exception:
            collections = []
        existing = [c.name for c in collections]
        if self.collection_name not in existing:
            distance = qmodels.Distance.EUCLID if self.distance.lower() == "euclid" else qmodels.Distance.COSINE
            params = qmodels.VectorParams(size=self.vector_size, distance=distance)
            vectors_config = {self.vector_name: params} if self.vector_name else params
            self.client.create_collection(collection_name=self.collection_name, vectors_config=vectors_config)

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = dict(record.get("metadata") or {})
        payload.setdefault("document_id", record.get("document_id"))
        payload.setdefault("document_version", record.get("document_version"))
        payload.setdefault("document_type", record.get("document_type"))
        payload.setdefault("source", record.get("source"))
        payload.setdefault("page_number", record.get("page_number"))
        payload.setdefault("department", record.get("department"))
        payload.setdefault("tenant_id", record.get("tenant_id"))
        payload.setdefault("access_control", record.get("access_control"))
        payload.setdefault("created_at", record.get("created_at"))
        payload.setdefault("chunk_id", record.get("chunk_id"))
        payload.setdefault("text", record.get("text"))
        vector = list(record["embedding"])
        if self.vector_name:
            vector = {self.vector_name: vector}
        return {
            "id": _to_point_id(record["id"]),
            "vector": vector,
            "payload": payload,
        }

    def upsert(self, records: Sequence[dict[str, Any]]) -> list[str]:
        points = [self._normalize_record(record) for record in records]
        if not points:
            return []
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return [str(point["id"]) for point in points]

    def _build_filter(self, filters: dict[str, Any] | None) -> qmodels.Filter | None:
        if not filters:
            return None
        must = []
        for key, value in filters.items():
            if value is None:
                continue
            if key == "document_version" and value == "latest":
                continue
            must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value)))
        return qmodels.Filter(must=must) if must else None

    def search(self, vector: Sequence[float], top_k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        q_filters = self._build_filter(filters)
        query_vector = (self.vector_name, list(vector)) if self.vector_name else list(vector)
        result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=q_filters,
            with_payload=True,
        )
        return [
            {
                "id": str(item.id),
                "score": float(item.score),
                "payload": item.payload,
            }
            for item in result
        ]

    def scroll_all(self, batch_size: int = 256, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return every point's payload in the collection (paginated), optionally filtered."""
        q_filters = self._build_filter(filters)
        records: list[dict[str, Any]] = []
        offset = None
        while True:
            try:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    scroll_filter=q_filters,
                )
            except Exception:
                break
            for point in points:
                records.append({"id": str(point.id), "payload": point.payload})
            if offset is None:
                break
        return records

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points=[_to_point_id(item) for item in ids],
        )

    def delete_by_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
                )
            ),
        )

    def health_check(self) -> bool:
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False
