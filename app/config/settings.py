"""Application settings and environment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Central configuration for the platform."""

    project_name: str = "Medical GenAI RAG"
    environment: str = os.getenv("ENVIRONMENT", "development")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.1")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))


settings = Settings()
