"""Metadata filtering and authorization checks for retrieval."""

from __future__ import annotations

from typing import Any


class MetadataFilter:
    """Apply metadata-based filters to retrieval results before context building."""

    def apply(self, results: list[dict[str, Any]], filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not results:
            return []
        filter_map = filters or {}
        filtered: list[dict[str, Any]] = []
        for item in results:
            metadata = item.get("metadata") or {}
            is_allowed = True
            for key, expected in filter_map.items():
                if expected is None:
                    continue
                if key == "document_id":
                    actual = item.get("document_id")
                elif key == "document_version":
                    actual = metadata.get("document_version") or item.get("document_version")
                elif key == "source":
                    actual = metadata.get("source") or item.get("source")
                elif key == "document_type":
                    actual = metadata.get("document_type") or item.get("document_type")
                elif key == "department":
                    actual = metadata.get("department") or item.get("department")
                elif key == "tenant_id":
                    actual = metadata.get("tenant_id") or item.get("tenant_id")
                elif key == "access_control":
                    actual = metadata.get("access_control") or item.get("access_control")
                elif key == "page_number":
                    actual = metadata.get("page_number") or item.get("page_number")
                elif key == "date":
                    actual = metadata.get("effective_date") or metadata.get("created_at") or item.get("created_at")
                else:
                    actual = metadata.get(key) or item.get(key)
                if actual != expected:
                    is_allowed = False
                    break
            if is_allowed:
                filtered.append(item)
        return filtered


class AuthorizationFilter:
    """Reject restricted documents before they are given to the LLM."""

    def apply(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        allowed: list[dict[str, Any]] = []
        for item in results:
            metadata = item.get("metadata") or {}
            access_level = metadata.get("access_control") or item.get("access_control") or "internal"
            if access_level in {"restricted", "denied", "blocked"}:
                continue
            allowed.append(item)
        return allowed
