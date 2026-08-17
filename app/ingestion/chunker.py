"""Chunking logic for ingested documents."""

from __future__ import annotations

import logging
import os
from typing import List, TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.metadata import DocumentMetadata, MetadataExtractor
from app.ingestion.parser import DocumentParser

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


class Chunk(TypedDict):
    """A single chunk of text plus the document metadata it was drawn from."""

    chunk_id: str
    chunk_index: int
    text: str
    metadata: DocumentMetadata


class DocumentChunker:
    """Parses a document, chunks its text, and attaches its metadata to each chunk."""

    def __init__(
        self,
        file_path: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.file_path = file_path
        self.parser = DocumentParser(file_path)
        self.metadata_extractor = MetadataExtractor(file_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_document(self) -> List[Chunk]:
        """Parse the document and split it into metadata-tagged chunks."""
        text = self.parser.parse()
        if not text:
            return []

        metadata = self.metadata_extractor.extract()
        texts = self.text_splitter.split_text(text)

        return [
            Chunk(
                chunk_id=f"{metadata.document_id}_{index}",
                chunk_index=index,
                text=chunk_text,
                metadata=metadata,
            )
            for index, chunk_text in enumerate(texts)
        ]

    def chunk_directory(self, directory_path: str) -> List[Chunk]:
        """Chunk every supported file under a directory, recursing into subfolders."""
        all_chunks: List[Chunk] = []
        for root, dirs, files in os.walk(directory_path):
            dirs.sort()
            for entry in sorted(files):
                entry_path = os.path.join(root, entry)
                try:
                    all_chunks.extend(DocumentChunker(entry_path).chunk_document())
                except Exception:
                    logger.warning("Skipping %s: failed to chunk", entry_path, exc_info=True)
                    continue
        return all_chunks
