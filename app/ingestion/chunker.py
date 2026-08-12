"""Chunking logic for ingested documents."""

import os

from ingestion.loaders import DocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentChunker:
    """Chunker for documents loaded by DocumentLoader."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = DocumentLoader(file_path)

    def ReadEachFileFromDirectory(self, directory_path: str) -> list:
        """Read each file from the directory and return a list of text content."""
        text_content = []
        for entry in os.listdir(directory_path):
            entry_path = os.path.join(directory_path, entry)
            if not os.path.isfile(entry_path):
                continue
            try:
                text_content.extend(DocumentLoader(entry_path).CallingLoader())
            except ValueError:
                continue
        return text_content

    def chunk_document(self) -> list:
        """Chunk the document into smaller parts."""
        text_content = self.loader.CallingLoader()
        # Implement your chunking logic here. For example, you can split by paragraphs or sentences.
        chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Adjust the chunk size as needed
            chunk_overlap=200,  # Adjust the overlap as needed
        )

        for text in text_content:
            if text:  # Ensure the text is not None or empty
                chunks.extend(text_splitter.split_text(text))   
        
        return chunks
