"""Script to chunk and index documents into the configured vector store.

Offline/batch use only: don't run this while the API server is also up
against an embedded local-disk Qdrant path (see QDRANT_MODE in .env) — the
embedded store only allows one process on its storage path at a time. Prefer
the UI's Ingest tab while the server is running.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import get_settings
from app.indexing import DocumentIndexingService, chunk_to_record
from app.ingestion.chunker import DocumentChunker


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Chunk and index documents into the vector store.")
    parser.add_argument("path", nargs="?", default=settings.inputfile_path, help="File or directory to index")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        raise SystemExit(f"Path not found: {target}")

    chunker = DocumentChunker(str(target))
    chunks = chunker.chunk_directory(str(target)) if target.is_dir() else chunker.chunk_document()
    records = [chunk_to_record(chunk) for chunk in chunks]

    service = DocumentIndexingService()
    ids = service.index_chunks(records)

    print(f"Indexed {len(ids)} chunks from {target} into '{settings.qdrant_collection}'.")


if __name__ == "__main__":
    main()
