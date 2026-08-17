"""Tests for vector-store behavior."""

from __future__ import annotations

from types import SimpleNamespace

from app.vectorstore.qdrant_store import QdrantVectorStore, _to_point_id


class FakeQdrantClient:
    def __init__(self, *args, **kwargs):
        self.collections = []
        self.upserted = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=self.collection_name)]) if self.collection_name in [c.name for c in self.collections] else SimpleNamespace(collections=[])

    def create_collection(self, **kwargs):
        self.collections.append(SimpleNamespace(name=kwargs["collection_name"]))

    def upsert(self, collection_name, points):
        self.upserted.append((collection_name, points))

    def search(self, collection_name, query_vector, limit, query_filter=None, with_payload=True):
        return [
            SimpleNamespace(id="p1", score=0.92, payload={"document_id": "doc-1", "chunk_id": "c1"}),
            SimpleNamespace(id="p2", score=0.81, payload={"document_id": "doc-2", "chunk_id": "c2"}),
        ]

    def delete(self, collection_name, points=None, points_selector=None):
        return None

    def get_collection(self, collection_name):
        return SimpleNamespace(name=collection_name)


def test_qdrant_store_upserts_records(monkeypatch):
    fake_client = FakeQdrantClient()
    monkeypatch.setattr("app.vectorstore.qdrant_store.QdrantClient", lambda url=None: fake_client)
    store = QdrantVectorStore(collection_name="medical_test", vector_size=3, url="http://localhost:6333")
    ids = store.upsert([
        {"id": "p1", "embedding": [0.1, 0.2, 0.3], "document_id": "doc-1", "chunk_id": "c1", "metadata": {"document_type": "clinical_guideline"}},
    ])
    assert ids == [_to_point_id("p1")]
    assert fake_client.upserted


def test_qdrant_store_search_returns_results(monkeypatch):
    fake_client = FakeQdrantClient()
    monkeypatch.setattr("app.vectorstore.qdrant_store.QdrantClient", lambda url=None: fake_client)
    store = QdrantVectorStore(collection_name="medical_test", vector_size=3, url="http://localhost:6333")
    result = store.search([0.1, 0.2, 0.3], top_k=2)
    assert len(result) == 2
    assert result[0]["payload"]["document_id"] == "doc-1"


def test_to_point_id_normalizes_arbitrary_strings():
    assert _to_point_id("42") == "42"
    uid = "123e4567-e89b-12d3-a456-426614174000"
    assert _to_point_id(uid) == uid
    assert _to_point_id("README_0") == _to_point_id("README_0")
    assert _to_point_id("README_0") != _to_point_id("README_1")


def test_qdrant_store_health_check(monkeypatch):
    fake_client = FakeQdrantClient()
    monkeypatch.setattr("app.vectorstore.qdrant_store.QdrantClient", lambda url=None: fake_client)
    store = QdrantVectorStore(collection_name="medical_test", vector_size=3, url="http://localhost:6333")
    assert store.health_check() is True
