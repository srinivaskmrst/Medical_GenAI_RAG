"""Metadata extraction and normalization."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

from pydantic import BaseModel

from app.ingestion.loaders import DocumentLoader


class DocumentMetadata(BaseModel):
    document_id: str
    file_name: str
    file_type: str

    source: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None

    page_number: Optional[int] = None
    section: Optional[str] = None

    created_at: Optional[str] = None
    modified_at: Optional[str] = None

    version: Optional[str] = None

    document_type: Optional[str] = None
    medical_domain: Optional[str] = None
    workflow_id: Optional[str] = None
    screen_id: Optional[str] = None
    access_level: Optional[str] = None
    effective_date: Optional[str] = None


# Every synthetic PDF/DOCX in the corpus prints, near the top of the body:
# "Document ID: DOC_025 | Type: screen_instructions | Domain: radiology | Workflow: WF_001 | Screen: SCREEN_01"
_BANNER_PATTERN = re.compile(
    r"Document ID:\s*(?P<document_id>\S+)"
    r"\s*\|\s*Type:\s*(?P<document_type>\S+)"
    r"\s*\|\s*Domain:\s*(?P<domain>\S+)"
    r"(?:\s*\|\s*Workflow:\s*(?P<workflow_id>\S+))?"
    r"(?:\s*\|\s*Screen:\s*(?P<screen_id>\S+))?"
)
_DOC_ID_IN_FILENAME_PATTERN = re.compile(r"(DOC_\d+)", re.IGNORECASE)
_DISCLAIMER_MARKER = "SYNTHETIC TEST DATA"
_BANNER_MARKER = "Document ID:"


class MetadataExtractor:
    """Builds a DocumentMetadata record for a source file.

    JSON files carry their metadata as structured fields and are read
    directly. PDF/DOCX files carry it as an in-body banner line
    ("Document ID: ... | Type: ... | Domain: ..."), which is recovered here
    by loading the file's raw (pre-cleaning) text via DocumentLoader.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = DocumentLoader(file_path)

    def _file_timestamps(self) -> Tuple[Optional[str], Optional[str]]:
        try:
            stat = os.stat(self.file_path)
        except OSError:
            return None, None
        created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()
        modified_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
        return created_at, modified_at

    def _document_id_from_filename(self) -> Optional[str]:
        match = _DOC_ID_IN_FILENAME_PATTERN.search(os.path.basename(self.file_path))
        return match.group(1).upper() if match else None

    def _fields_from_json(self) -> dict:
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {
            "document_id": data.get("document_id"),
            "title": data.get("document_name") or data.get("screen_name"),
            "document_type": data.get("document_type"),
            "version": data.get("version"),
            "medical_domain": data.get("medical_domain"),
            "workflow_id": data.get("workflow_id"),
            "screen_id": data.get("screen_id"),
            "source": data.get("source"),
            "access_level": data.get("access_level"),
            "effective_date": data.get("effective_date"),
        }

    def _raw_text_lines(self) -> List[str]:
        try:
            segments = self.loader.CallingLoader()
        except (ValueError, FileNotFoundError):
            return []
        lines: List[str] = []
        for segment in segments:
            if segment:
                lines.extend(segment.split("\n"))
        return lines

    def _fields_from_body(self, raw_lines: List[str]) -> dict:
        raw_text = "\n".join(raw_lines)
        match = _BANNER_PATTERN.search(raw_text)
        fields = {k: v for k, v in match.groupdict().items() if v} if match else {}
        if "domain" in fields:
            fields["medical_domain"] = fields.pop("domain")
        return fields

    def _title_from_body(self, raw_lines: List[str]) -> Optional[str]:
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith(_DISCLAIMER_MARKER) or line.startswith(_BANNER_MARKER):
                break
            return line
        return None

    def extract(self) -> DocumentMetadata:
        """Load the file and return its DocumentMetadata."""
        file_name = os.path.basename(self.file_path)
        file_type = os.path.splitext(file_name)[1].lower()
        created_at, modified_at = self._file_timestamps()

        if file_type == ".json":
            fields = self._fields_from_json()
        else:
            raw_lines = self._raw_text_lines()
            fields = self._fields_from_body(raw_lines)
            fields.setdefault("title", self._title_from_body(raw_lines))

        document_id = (
            fields.get("document_id")
            or self._document_id_from_filename()
            or os.path.splitext(file_name)[0]
        )

        return DocumentMetadata(
            document_id=document_id,
            file_name=file_name,
            file_type=file_type,
            source=fields.get("source"),
            title=fields.get("title"),
            version=fields.get("version"),
            created_at=created_at,
            modified_at=modified_at,
            document_type=fields.get("document_type"),
            medical_domain=fields.get("medical_domain"),
            workflow_id=fields.get("workflow_id"),
            screen_id=fields.get("screen_id"),
            access_level=fields.get("access_level"),
            effective_date=fields.get("effective_date"),
        )
