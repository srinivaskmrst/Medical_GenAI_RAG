"""Tests for metadata filtering and authorization checks."""

from __future__ import annotations

from app.retrieval.filters import AuthorizationFilter


def test_retrieval_guardrail_keeps_authorized_results():
    guardrail = AuthorizationFilter()
    results = [
        {"metadata": {"access_control": "internal"}, "document_id": "doc-1", "text": "authorized"},
        {"metadata": {"access_control": "restricted"}, "document_id": "doc-2", "text": "blocked"},
    ]

    filtered = guardrail.apply(results)

    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "doc-1"


def test_retrieval_guardrail_handles_missing_metadata():
    guardrail = AuthorizationFilter()
    results = [{"document_id": "doc-3", "text": "ok"}]

    filtered = guardrail.apply(results)

    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "doc-3"
