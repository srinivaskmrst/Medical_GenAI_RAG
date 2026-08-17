"""RAG query endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import QueryRequest, QueryResponse
from app.rag.rag_service import RAGService

router = APIRouter(tags=["rag"])
service = RAGService()


@router.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest) -> QueryResponse:
    result = service.answer(
        query=payload.query,
        filters=payload.filters,
        top_k=payload.top_k,
        session_id=payload.session_id,
    )
    return QueryResponse(**result)
