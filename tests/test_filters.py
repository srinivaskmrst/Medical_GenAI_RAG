"""Tests for metadata and authorization filtering."""

from app.retrieval.filters import AuthorizationFilter, MetadataFilter


def test_metadata_filter_applies_document_filters():
    results = [
        {"document_id": "doc-1", "metadata": {"document_type": "clinical_guideline", "department": "cardiology"}},
        {"document_id": "doc-2", "metadata": {"document_type": "policy", "department": "admin"}},
    ]

    filtered = MetadataFilter().apply(results, {"document_type": "clinical_guideline", "department": "cardiology"})

    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "doc-1"


def test_authorization_filter_blocks_restricted_documents():
    results = [
        {"document_id": "doc-safe", "metadata": {"access_control": "internal"}},
        {"document_id": "doc-blocked", "metadata": {"access_control": "restricted"}},
    ]

    filtered = AuthorizationFilter().apply(results)

    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "doc-safe"
