"""LLM provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common interface for language model providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt."""

    async def agenerate(self, prompt: str) -> str:
        """Async generation hook for providers that support it."""
        return self.generate(prompt)
