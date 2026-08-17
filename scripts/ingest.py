"""Script to ingest source documents into the RAG pipeline (dry run: chunk only, no indexing)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import get_settings
from app.ingestion.chunker import DocumentChunker


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Chunk documents without indexing them (dry run).")
    parser.add_argument("path", nargs="?", default=settings.inputfile_path, help="File or directory to chunk")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        raise SystemExit(f"Path not found: {target}")

    chunker = DocumentChunker(str(target))
    chunks = chunker.chunk_directory(str(target)) if target.is_dir() else chunker.chunk_document()

    per_document: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk["metadata"].document_id
        per_document[doc_id] = per_document.get(doc_id, 0) + 1

    print(f"Chunked {len(chunks)} chunks from {len(per_document)} document(s) under {target}:")
    for doc_id, count in sorted(per_document.items()):
        print(f"  {doc_id}: {count} chunks")


if __name__ == "__main__":
    main()
