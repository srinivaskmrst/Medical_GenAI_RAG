"""Context building from retrieved chunks."""

from __future__ import annotations

from typing import Any


class ContextBuilder:
    """Deduplicate and trim retrieved chunks before creating LLM context."""

    def __init__(self, max_context_tokens: int = 2000) -> None:
        self.max_context_tokens = max_context_tokens

    def build(self, documents: list[dict[str, Any]], max_tokens: int | None = None) -> list[dict[str, Any]]:
        if not documents:
            return []

        deduped: dict[str, dict[str, Any]] = {}
        for item in documents:
            key = str(item.get("chunk_id") or item.get("id") or item.get("document_id"))
            deduped[key] = item

        ranked = sorted(deduped.values(), key=lambda doc: float(doc.get("score", 0.0)), reverse=True)
        limit = max_tokens if max_tokens is not None else self.max_context_tokens
        token_budget = 0
        final: list[dict[str, Any]] = []
        for item in ranked:
            text = str(item.get("text", ""))
            estimated_tokens = max(1, len(text.split()))
            if token_budget + estimated_tokens > limit:
                continue
            final.append(item)
            token_budget += estimated_tokens
        return final
