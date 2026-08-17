"""Keyword retrieval via BM25-inspired ranking."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any


class BM25Retriever:
    """Simple BM25-like retriever for exact and lexical matches."""

    def __init__(self, documents: list[dict[str, Any]] | None = None):
        self.documents = documents or []

    def index(self, documents: list[dict[str, Any]]) -> None:
        self.documents = list(documents)

    def add(self, documents: list[dict[str, Any]]) -> None:
        self.documents.extend(documents)

    def remove_document(self, document_id: str) -> None:
        self.documents = [
            doc for doc in self.documents
            if doc.get("document_id") != document_id and doc.get("metadata", {}).get("document_id") != document_id
        ]

    def retrieve(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        terms = query.lower().split()
        if not terms:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self.documents:
            text = str(doc.get("text", "")).lower()
            if not text:
                continue
            if filters:
                matched = True
                for key, value in filters.items():
                    if doc.get("metadata", {}).get(key) != value and doc.get(key) != value:
                        matched = False
                        break
                if not matched:
                    continue

            score = 0.0
            for term in terms:
                hits = text.count(term)
                if hits:
                    score += hits * (1 + math.log(1 + len(self.documents)))
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [{**doc, "score": score} for score, doc in scored[:top_k]]


@lru_cache(maxsize=1)
def get_keyword_retriever() -> BM25Retriever:
    """Return the process-wide shared BM25 retriever singleton."""
    return BM25Retriever()
