"""Input guardrail checks for unsafe or malicious prompts."""

from __future__ import annotations


class InputGuardrail:
    """Validate user input before embedding or retrieval."""

    def validate(self, query: str) -> bool:
        text = (query or "").strip().lower()
        if not text:
            raise ValueError("Query is empty")
        prohibited = [
            "ignore previous instructions",
            "ignore all previous",
            "system prompt",
            "developer message",
            "bypass",
            "bypass safety",
            "new instructions",
            "override",
            "act as",
            "roleplay",
        ]
        for phrase in prohibited:
            if phrase in text:
                raise ValueError("Input contains disallowed instructions")
        return True
