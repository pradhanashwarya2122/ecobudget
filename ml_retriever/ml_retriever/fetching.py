"""The fetch contract between this ML workstream and the web/fetch layer.

plan.md Phase 6: decouple ML from the network by agreeing a single signature --
`fetch_passages_for_requirement(requirement) -> list[Passage]` -- that offline
(pre-downloaded corpus) and live fetching both implement. The pipeline only
ever talks to this interface, so Teammate 2's live fetcher can be dropped in
later without touching the ML code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Passage, Requirement


@runtime_checkable
class PassageSource(Protocol):
    def fetch_passages_for_requirement(self, requirement: Requirement, k: int = 5) -> list[Passage]:
        """Return up to k candidate passages for a single requirement,
        best-first. May hit the network or read a local corpus; callers don't
        care which."""
        ...


class CorpusPassageSource:
    """Offline-first source: ranks a pre-embedded in-memory corpus with the
    Phase 3 `RequirementRetriever`. The default for experiments; a live
    fetcher implements the same `fetch_passages_for_requirement` signature."""

    def __init__(self, retriever):
        self._retriever = retriever

    def fetch_passages_for_requirement(self, requirement: Requirement, k: int = 5) -> list[Passage]:
        return [scored.passage for scored in self._retriever.retrieve(requirement, k=k)]
