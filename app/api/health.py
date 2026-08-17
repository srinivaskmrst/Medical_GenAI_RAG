"""Health check endpoints for the API service."""

from __future__ import annotations

from fastapi import APIRouter

from app.llm.factory import get_llm_provider
from app.vectorstore.factory import get_cache_vector_store, get_long_term_memory_vector_store, get_vector_store

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    qdrant_ok = get_vector_store().health_check()
    cache_ok = get_cache_vector_store().health_check()
    long_memory_ok = get_long_term_memory_vector_store().health_check()
    ollama_ok = get_llm_provider().health_check()
    return {
        "status": "ok" if (qdrant_ok and cache_ok and long_memory_ok and ollama_ok) else "degraded",
        "qdrant": qdrant_ok,
        "cache": cache_ok,
        "long_term_memory": long_memory_ok,
        "ollama": ollama_ok,
    }


@router.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@router.get("/version")
def version() -> dict[str, str]:
    return {"version": "1.0.0"}
