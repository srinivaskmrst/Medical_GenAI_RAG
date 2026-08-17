"""Ollama-backed LLM provider."""

from __future__ import annotations

import ollama

from app.config.settings import get_settings
from app.llm.base import LLMProvider


class OllamaLLMProvider(LLMProvider):
    """A local LLM provider for Ollama-based generation."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        request_timeout: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.ollama_base_url
        self.request_timeout = request_timeout or settings.ollama_request_timeout
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.max_tokens

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty")
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            stream=False,
        )
        message = response.get("message", {})
        return str(message.get("content", "")).strip()

    async def agenerate(self, prompt: str) -> str:
        return self.generate(prompt)

    def health_check(self) -> bool:
        try:
            ollama.list()
            return True
        except Exception:
            return False
