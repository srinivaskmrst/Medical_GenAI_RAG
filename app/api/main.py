"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.rag import router as rag_router
from app.retrieval.keyword import get_keyword_retriever
from app.vectorstore.factory import get_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prime the in-memory BM25 keyword index from whatever is already in Qdrant."""
    try:
        records = get_vector_store().scroll_all()
        documents = [record["payload"] for record in records if (record.get("payload") or {}).get("text")]
        get_keyword_retriever().index(documents)
    except Exception:
        pass
    yield


app = FastAPI(title="Medical GenAI RAG", version="1.0.0", lifespan=lifespan)

app.include_router(health_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Guardrail rejections and validation errors surface as 400s, not 500s."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Medical GenAI RAG", "status": "ok"}
