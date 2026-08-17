"""Application settings and environment configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the platform."""

    project_name: str = "Medical GenAI RAG"
    app_env: str = Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    qdrant_host: str = Field(default="localhost", validation_alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, validation_alias="QDRANT_PORT")
    qdrant_grpc_port: int = Field(default=6334, validation_alias="QDRANT_GRPC_PORT")
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_endpoint: str | None = Field(default=None, validation_alias="QDRANT_ENDPOINT")
    qdrant_collection: str = Field(default="Medical_Main_memory", validation_alias="QDRANT_COLLECTION")
    qdrant_cache_collection: str = Field(default="Medical_Cache_memory", validation_alias="QDRANT_CACHE_COLLECTION")
    qdrant_long_collection: str = Field(default="Medical_Long_memory", validation_alias="QDRANT_LONG_COLLECTION")
    qdrant_api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_mode: str = Field(default="embedded", validation_alias="QDRANT_MODE")
    qdrant_path: str = Field(default="./data/qdrant_storage", validation_alias="QDRANT_PATH")

    cache_enabled: bool = Field(default=True, validation_alias="CACHE_ENABLED")
    cache_similarity_threshold: float = Field(default=0.92, validation_alias="CACHE_SIMILARITY_THRESHOLD")

    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_request_timeout: int = Field(default=120, validation_alias="OLLAMA_REQUEST_TIMEOUT")

    embedding_provider: str = Field(default="local", validation_alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", validation_alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", validation_alias="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=32, validation_alias="EMBEDDING_BATCH_SIZE")
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")

    llm_provider: str = Field(default="ollama", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="llama3.1", validation_alias="OLLAMA_MODEL")
    temperature: float = Field(default=0.1, validation_alias="OLLAMA_TEMPERATURE")
    max_tokens: int = Field(default=512, validation_alias="MAX_TOKENS")

    top_k: int = Field(default=10, validation_alias="TOP_K")
    rerank_top_k: int = Field(default=5, validation_alias="RERANK_TOP_K")
    semantic_weight: float = Field(default=0.7, validation_alias="SEMANTIC_WEIGHT")
    keyword_weight: float = Field(default=0.3, validation_alias="KEYWORD_WEIGHT")
    score_threshold: float = Field(default=0.3, validation_alias="SCORE_THRESHOLD")

    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", validation_alias="RERANKER_MODEL")
    reranker_enabled: bool = Field(default=True, validation_alias="RERANKER_ENABLED")

    grounding_min_confidence: float = Field(default=0.5, validation_alias="GROUNDING_MIN_CONFIDENCE")

    chunk_size: int = Field(default=800, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, validation_alias="CHUNK_OVERLAP")

    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")

    inputfile_path: str = Field(default="./data/Medical_Input_Data", validation_alias="INPUTFILE_PATH")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()


settings = get_settings()
