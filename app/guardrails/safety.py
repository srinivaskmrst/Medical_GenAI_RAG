"""Safety checks and medical guidance constraints."""

from __future__ import annotations


class SafetyGuardrail:
    """Perform basic safety checks on medical questions and responses."""

    def validate_question(self, question: str) -> bool:
        if not question or not question.strip():
            raise ValueError("Question is empty")
        unsafe_markers = (
            "self-harm",
            "suicide",
            "kill yourself",
            "poisoning",
            "overdose",
        )
        lowered = question.lower()
        for marker in unsafe_markers:
            if marker in lowered:
                raise ValueError("Question contains unsafe content")
        return True

    def validate_answer(self, answer: str) -> bool:
        if not answer or not answer.strip():
            raise ValueError("Answer is empty")
        return True
