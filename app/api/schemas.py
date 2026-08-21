"""FastAPI request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: str | None = Field(default=None, description="Scopes cross-session long-term memory")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0


class IngestRequest(BaseModel):
    file_path: str


class IngestResponse(BaseModel):
    status: str
    file_path: str
    num_chunks: int = 0
    ids: list[str] = Field(default_factory=list)
    skipped_files: list[dict[str, str]] = Field(default_factory=list)


class SessionSummary(BaseModel):
    session_id: str
    title: str
    last_updated: float = 0.0
    message_count: int = 0


class SessionMessage(BaseModel):
    role: str | None = None
    text: str | None = None
    created_at: float | None = None


class SessionMessagesResponse(BaseModel):
    session_id: str
    messages: list[SessionMessage] = Field(default_factory=list)
