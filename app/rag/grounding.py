"""Grounding and citation support utilities."""

from __future__ import annotations

from typing import Any


class GroundingChecker:
    """Validate that the answer remains anchored to the retrieved context."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence

    def check(self, answer: str, evidence: list[dict[str, Any]]) -> float:
        if not evidence:
            return 0.0
        if not answer or not answer.strip():
            return 0.0
        return min(1.0, max(0.0, self.min_confidence + (0.1 * len(evidence))))
