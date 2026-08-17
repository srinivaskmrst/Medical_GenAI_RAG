"""Tests for document indexing into the vector store."""

from __future__ import annotations

from app.indexing import DocumentIndexingService


class FakeEmbeddingProvider:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    def __init__(self):
        self.records = []

    def upsert(self, records):
        self.records.extend(records)
        return [record["id"] for record in records]

    def delete_by_document(self, document_id):
        self.records = [record for record in self.records if record["document_id"] != document_id]


def test_indexing_service_builds_vector_records():
    vector_store = FakeVectorStore()
    service = DocumentIndexingService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    chunks = [
        {
            "chunk_id": "doc_001_0",
            "document_id": "doc_001",
            "text": "Clinical guidance recommends treatment review.",
            "source": "clinical_guideline",
            "page_number": 3,
            "document_type": "clinical_guideline",
            "metadata": {
                "document_id": "doc_001",
                "document_version": "v1",
                "document_type": "clinical_guideline",
                "source": "clinical_guideline",
                "page_number": 3,
                "department": "cardiology",
                "tenant_id": "tenant-1",
                "access_control": "internal",
                "created_at": "2024-01-01T00:00:00",
            },
        }
    ]

    ids = service.index_chunks(chunks)

    assert ids == ["doc_001_0"]
    assert len(vector_store.records) == 1
    assert vector_store.records[0]["document_id"] == "doc_001"
    assert vector_store.records[0]["chunk_id"] == "doc_001_0"
    assert len(vector_store.records[0]["embedding"]) == 3


def test_indexing_service_delete_by_document_removes_all_chunks():
    vector_store = FakeVectorStore()
    service = DocumentIndexingService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
    )

    chunks = [
        {"chunk_id": "doc_002_0", "document_id": "doc_002", "text": "First chunk", "source": "guide", "page_number": 2, "document_type": "policy", "metadata": {"document_id": "doc_002"}},
        {"chunk_id": "doc_002_1", "document_id": "doc_002", "text": "Second chunk", "source": "guide", "page_number": 2, "document_type": "policy", "metadata": {"document_id": "doc_002"}},
    ]

    service.index_chunks(chunks)
    service.delete_document("doc_002")

    assert len(vector_store.records) == 0
