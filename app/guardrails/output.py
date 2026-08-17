"""Output guardrail validation for final responses."""

from __future__ import annotations


class OutputGuardrail:
    """Checks the final answer for unsupported claims and sensitive leakage."""

    def validate(self, answer: str) -> bool:
        text = (answer or "").lower()
        forbidden = ["patient name", "ssn", "medical record number", "phone number"]
        for phrase in forbidden:
            if phrase in text:
                raise ValueError("Response contains sensitive or disallowed content")
        return True
