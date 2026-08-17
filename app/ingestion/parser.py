"""Parsing utilities for source documents: cleaning, normalization, and
JSON-to-text conversion so every file type reaches the chunker as a single,
readable block of text.
"""

from __future__ import annotations

import json
import re
from typing import List

from app.ingestion.loaders import DocumentLoader

# Every synthetic PDF/DOCX in the corpus prints a disclaimer banner near the
# top (directly followed by a metadata banner, stripped separately below),
# and some also repeat a second, shorter disclaimer near an embedded image.
_DISCLAIMER_PATTERNS = (
    re.compile(
        r"SYNTHETIC TEST DATA.*?do not use for actual patient care\.",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"This is a synthetic example document created for RAG system testing\."
        r".*?institutional policy or clinical guidance\.",
        re.IGNORECASE | re.DOTALL,
    ),
)
# Metadata banner, e.g.:
# "Document ID: DOC_025 | Type: screen_instructions | Domain: radiology | Workflow: WF_001 | Screen: SCREEN_01"
_METADATA_BANNER_PATTERN = re.compile(r"^Document ID:\s*\S+.*$", re.MULTILINE)

_INLINE_WHITESPACE_PATTERN = re.compile(r"[ \t]+")
_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")

# Keys already captured structurally by MetadataExtractor and not useful as
# free-text chunk content.
_JSON_METADATA_KEYS = {
    "document_id",
    "document_name",
    "document_type",
    "version",
    "workflow_id",
    "screen_id",
    "medical_domain",
    "effective_date",
    "source",
    "access_level",
    "disclaimer",
}


class DocumentParser:
    """Parses a source file into a single, clean, chunk-ready text block."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = DocumentLoader(file_path)

    def clean_text(self, text: str) -> str:
        """Strip boilerplate/disclaimers and normalize whitespace."""
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        for pattern in _DISCLAIMER_PATTERNS:
            text = pattern.sub("", text)
        text = _METADATA_BANNER_PATTERN.sub("", text)
        text = _INLINE_WHITESPACE_PATTERN.sub(" ", text)

        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)
        text = _BLANK_LINES_PATTERN.sub("\n\n", text)
        return text.strip()

    def json_to_text(self, data: dict) -> str:
        """Render a structured JSON screen-definition as readable text."""
        lines: List[str] = []

        title = data.get("document_name") or data.get("screen_name")
        if title:
            lines.append(str(title))

        for key, value in data.items():
            if key in _JSON_METADATA_KEYS:
                continue
            if not value:
                continue
            label = key.replace("_", " ").title()
            if isinstance(value, list):
                lines.append(f"{label}:")
                lines.extend(f"- {item}" for item in value)
            else:
                lines.append(f"{label}: {value}")

        return "\n".join(lines)

    def parse(self) -> str:
        """Load and normalize the document into a single clean text block."""
        file_type = self.loader.IdentifyFileType()

        if file_type == ".json":
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return self.clean_text(self.json_to_text(data))
            return self.clean_text(str(data))

        text_content = self.loader.CallingLoader()
        joined = "\n".join(segment for segment in text_content if segment)
        return self.clean_text(joined)
