"""Interfaces that later phases implement against.

Defined as `typing.Protocol` (structural typing, runtime-checkable) rather
than ABCs so that:
  - Phase 4's rewritten EvidenceCoverageTracker, and any future tracker,
    can satisfy `EvidenceTracker` without inheriting from a base class.
  - Test doubles in this phase's own test suite can be tiny and dependency-free.

Nothing in this module should ever import a model library or an LLM API
client -- it only describes shapes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import AnswerResult, Passage, Requirement


@runtime_checkable
class EvidenceTracker(Protocol):
    """Tracks, for a fixed set of Requirements, whether retrieved Passages
    provide sufficient evidence.

    Implementations MUST NOT use ground truth in any of these methods --
    only the Passages that have actually been added via `add_passage`.
    """

    def add_passage(self, passage: Passage) -> None:
        """Register a newly retrieved passage and update internal coverage
        state. Must be idempotent for the same passage_id."""
        ...

    def is_sufficient(self) -> bool:
        """True once every requirement is satisfied at the tracker's
        threshold (or there is nothing left to satisfy)."""
        ...

    def frac_satisfied(self) -> float:
        """Fraction of requirements currently satisfied, in [0, 1]."""
        ...

    def frac_remaining(self) -> float:
        """1 - frac_satisfied(). Kept as a separate method (rather than
        derived inline by callers) because it is used directly as a
        bandit/controller feature."""
        ...

    def coverage(self) -> float:
        """Overall coverage score in [0, 1]. May differ from
        frac_satisfied() if the tracker uses a continuous/partial-credit
        notion of coverage rather than a hard per-requirement threshold."""
        ...

    def unanswered_requirements(self) -> list[Requirement]:
        """Requirements not yet satisfied, in priority order if the
        implementation has one, otherwise in insertion order."""
        ...


@runtime_checkable
class AnswerGenerator(Protocol):
    """Produces a final answer from a question and the evidence gathered
    so far. Must only use the passed-in evidence -- no ground truth, and
    no re-fetching additional passages of its own."""

    def generate(self, question: str, evidence: list[Passage]) -> AnswerResult:
        ...
