"""Tests for configuration settings."""

from app.config.settings import get_settings


def test_settings_loads_default_values():
    settings = get_settings()
    assert settings.project_name == "Medical GenAI RAG"
    assert settings.qdrant_collection == "Medical_Main_memory"
    assert settings.embedding_provider == "google"
    assert settings.embedding_model == "gemini-embedding-001"
