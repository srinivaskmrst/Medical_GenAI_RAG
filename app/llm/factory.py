"""LLM provider factory."""

from __future__ import annotations

from functools import lru_cache

from app.config.settings import get_settings
from app.llm.base import LLMProvider
from app.llm.ollama_service import OllamaLLMProvider


@lru_cache(maxsize=4)
def get_llm_provider(provider: str | None = None) -> LLMProvider:
    """Return a configured LLM provider instance."""
    settings = get_settings()
    selected = (provider or settings.llm_provider).lower()
    if selected in {"ollama", "local", "local_model"}:
        return OllamaLLMProvider(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            request_timeout=settings.ollama_request_timeout,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    raise ValueError(f"Unsupported LLM provider: {provider or settings.llm_provider}")


__all__ = ["LLMProvider", "get_llm_provider", "OllamaLLMProvider"]
