"""Document management endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.schemas import IngestRequest, IngestResponse
from app.indexing import DocumentIndexingService, chunk_to_record
from app.ingestion.chunker import Chunk, DocumentChunker
from app.retrieval.keyword import get_keyword_retriever

router = APIRouter(prefix="/documents", tags=["documents"])
indexing_service = DocumentIndexingService()


def _chunk_document(file_path: str) -> tuple[list[Chunk], list[dict[str, str]]]:
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {file_path}")
    chunker = DocumentChunker(str(path))
    if path.is_dir():
        chunks = chunker.chunk_directory(str(path))
        return chunks, chunker.skipped_files
    return chunker.chunk_document(), []


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(payload: IngestRequest) -> IngestResponse:
    """Chunk a file or directory without indexing it (dry run)."""
    chunks, skipped_files = _chunk_document(payload.file_path)
    return IngestResponse(
        status="chunked", file_path=payload.file_path, num_chunks=len(chunks), skipped_files=skipped_files
    )


@router.post("/index", response_model=IngestResponse)
def index_document(payload: IngestRequest) -> IngestResponse:
    """Chunk a file or directory and index it into the vector store."""
    chunks, skipped_files = _chunk_document(payload.file_path)
    records = [chunk_to_record(chunk) for chunk in chunks]
    ids = indexing_service.index_chunks(records)
    get_keyword_retriever().add(records)
    return IngestResponse(
        status="indexed",
        file_path=payload.file_path,
        num_chunks=len(chunks),
        ids=ids,
        skipped_files=skipped_files,
    )


@router.delete("/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    indexing_service.delete_document(document_id)
    get_keyword_retriever().remove_document(document_id)
    return {"status": "deleted", "document_id": document_id}
