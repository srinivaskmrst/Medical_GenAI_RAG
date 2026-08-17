"""Tests for retrieval primitives."""

from app.retrieval.semantic import SemanticRetriever
from app.retrieval.keyword import BM25Retriever


class FakeEmbeddings:
    def embed_query(self, query):
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self):
        self.calls = []

    def search(self, vector, top_k=5, filters=None):
        self.calls.append({"vector": vector, "top_k": top_k, "filters": filters})
        return [{"id": "chunk-1", "score": 0.95, "payload": {"document_id": "doc-1", "chunk_id": "chunk-1", "text": "clinical guidance for treatment", "source": "guide", "page_number": 12, "metadata": {"access_control": "internal"}}}]


def test_semantic_retriever_calls_vector_store():
    fake_vector_store = FakeVectorStore()
    retriever = SemanticRetriever(top_k=3, embedding_provider=FakeEmbeddings(), vector_store=fake_vector_store)
    results = retriever.retrieve("what is the treatment guidance", top_k=3)
    assert results[0]["document_id"] == "doc-1"
    assert fake_vector_store.calls[0]["top_k"] == 3


def test_semantic_retriever_flattens_payload_results():
    fake_vector_store = FakeVectorStore()
    retriever = SemanticRetriever(top_k=3, embedding_provider=FakeEmbeddings(), vector_store=fake_vector_store)
    results = retriever.retrieve("what is the treatment guidance", top_k=3)
    assert results[0]["document_id"] == "doc-1"
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["text"] == "clinical guidance for treatment"


def test_keyword_retriever_ranks_matches():
    retriever = BM25Retriever([
        {"text": "clinical treatment guidance", "metadata": {"document_type": "clinical_guideline"}},
        {"text": "other unrelated note", "metadata": {"document_type": "note"}},
    ])
    results = retriever.retrieve("treatment guidance", top_k=2)
    assert results[0]["text"] == "clinical treatment guidance"
