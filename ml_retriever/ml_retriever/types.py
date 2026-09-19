"""Core data structures shared across the retrieval pipeline.

These are intentionally plain dataclasses (no ORM/pydantic coupling) so
they're cheap to construct in tests and easy to serialize to JSON lines
for the Phase 1 corpus/task datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Requirement:
    """A single piece of information a task needs.

    Example: Requirement(entity="iPhone 15", attribute="battery capacity")

    `value` is optional and is used when a requirement already has a known
    or hypothesized answer to check against (e.g. during evaluation), but
    is never used by retrieval or stopping logic -- only by scoring code.
    """

    entity: str
    attribute: str
    value: Optional[str] = None

    def key(self) -> tuple[str, str]:
        """Stable (entity, attribute) identity, ignoring `value`.

        Two Requirements with the same entity/attribute but different
        (or absent) values should be treated as the same requirement by
        trackers and retrievers.
        """
        return (self.entity.strip().lower(), self.attribute.strip().lower())


@dataclass
class Passage:
    """A retrievable unit of evidence."""

    passage_id: str
    text: str
    byte_size: int
    source_url: str = ""
    embedding: Optional[Any] = None  # np.ndarray, kept as Any to avoid a hard numpy import here
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.byte_size < 0:
            raise ValueError("byte_size must be non-negative")
        if not self.passage_id:
            raise ValueError("passage_id must be non-empty")


@dataclass
class AnswerResult:
    """Standard output shape for any AnswerGenerator implementation."""

    answer: Optional[str]
    confidence: float
    used_evidence_ids: list[str]
    abstained: bool
    generator_version: str

    def __post_init__(self) -> None:
        if self.abstained and self.answer is not None:
            raise ValueError("abstained AnswerResult must not carry an answer")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
