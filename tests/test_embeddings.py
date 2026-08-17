"""Tests for embeddings generation."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from app.embeddings import LocalEmbeddingProvider, get_embedding_provider


@patch("app.embeddings.providers.local.SentenceTransformer")
def test_local_embedding_provider_embeds_documents(mock_model):
    mock_model.return_value.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]

    provider = LocalEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")
    result = provider.embed_documents(["hello", "world"])

    assert len(result) == 2
    assert len(result[0]) == 2
    assert result[0] == [0.1, 0.2]


@patch("app.embeddings.providers.local.SentenceTransformer")
def test_local_embedding_provider_validates_dimension(mock_model):
    mock_model.return_value.encode.side_effect = [
        [[0.1, 0.2, 0.3]],
        [[0.1, 0.2]],
    ]

    provider = LocalEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")
    provider.embed_query("hello")

    with pytest.raises(ValueError):
        provider.embed_query("world")


@patch("app.embeddings.providers.local.SentenceTransformer")
def test_factory_returns_provider_for_local_model(mock_model):
    get_embedding_provider.cache_clear()
    provider = get_embedding_provider("local")
    assert isinstance(provider, LocalEmbeddingProvider)


def test_factory_raises_for_unknown_provider():
    with pytest.raises(ValueError):
        get_embedding_provider("unknown-provider")
