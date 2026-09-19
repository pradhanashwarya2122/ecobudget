"""These tests don't exercise any real ML logic (that's Phase 2-6). They
exist to lock down the *shape* of EvidenceTracker and AnswerGenerator so
later phases can't silently drift from the Phase 0 contract.
"""

from ml_retriever.interfaces import AnswerGenerator, EvidenceTracker
from ml_retriever.types import AnswerResult, Passage, Requirement


class DummyEvidenceTracker:
    """Minimal in-memory tracker: a requirement is "satisfied" once any
    added passage's text contains both the entity and the attribute as
    substrings. This is intentionally too simple for production (that's
    exactly the naive 'looks relevant' failure mode described in the
    project history) -- it's only here to validate the interface shape.
    """

    def __init__(self, requirements: list[Requirement]):
        self._requirements = list(requirements)
        self._satisfied: set[tuple[str, str]] = set()
        self._passages: list[Passage] = []

    def add_passage(self, passage: Passage) -> None:
        if any(p.passage_id == passage.passage_id for p in self._passages):
            return  # idempotent
        self._passages.append(passage)
        text_lower = passage.text.lower()
        for req in self._requirements:
            entity, attribute = req.key()
            if entity in text_lower and attribute in text_lower:
                self._satisfied.add(req.key())

    def is_sufficient(self) -> bool:
        return len(self._satisfied) == len(self._requirements)

    def frac_satisfied(self) -> float:
        if not self._requirements:
            return 1.0
        return len(self._satisfied) / len(self._requirements)

    def frac_remaining(self) -> float:
        return 1.0 - self.frac_satisfied()

    def coverage(self) -> float:
        return self.frac_satisfied()

    def unanswered_requirements(self) -> list[Requirement]:
        return [r for r in self._requirements if r.key() not in self._satisfied]


class DummyAnswerGenerator:
    """Trivial stand-in: abstains if there's no evidence, otherwise
    'answers' with the id of the first passage. Shape-only, not a real
    generator (that's Phase 6)."""

    def generate(self, question: str, evidence: list[Passage]) -> AnswerResult:
        if not evidence:
            return AnswerResult(
                answer=None,
                confidence=0.0,
                used_evidence_ids=[],
                abstained=True,
                generator_version="dummy-v0",
            )
        return AnswerResult(
            answer=f"answer-from-{evidence[0].passage_id}",
            confidence=0.5,
            used_evidence_ids=[evidence[0].passage_id],
            abstained=False,
            generator_version="dummy-v0",
        )


def test_dummy_tracker_satisfies_protocol():
    tracker = DummyEvidenceTracker([Requirement(entity="iphone 15", attribute="battery")])
    assert isinstance(tracker, EvidenceTracker)


def test_dummy_generator_satisfies_protocol():
    assert isinstance(DummyAnswerGenerator(), AnswerGenerator)


def test_tracker_starts_insufficient_with_requirements():
    req = Requirement(entity="iphone 15", attribute="battery")
    tracker = DummyEvidenceTracker([req])
    assert tracker.is_sufficient() is False
    assert tracker.frac_satisfied() == 0.0
    assert tracker.frac_remaining() == 1.0
    assert tracker.unanswered_requirements() == [req]


def test_tracker_becomes_sufficient_after_matching_passage():
    req = Requirement(entity="iphone 15", attribute="battery")
    tracker = DummyEvidenceTracker([req])
    tracker.add_passage(
        Passage(passage_id="p1", text="The iPhone 15 battery lasts 20 hours.", byte_size=40)
    )
    assert tracker.is_sufficient() is True
    assert tracker.frac_satisfied() == 1.0
    assert tracker.unanswered_requirements() == []


def test_entity_only_passage_does_not_satisfy_requirement():
    """Regression guard for Phase 4: a passage mentioning only the entity,
    not the attribute, must not count as evidence."""
    req = Requirement(entity="iphone 15", attribute="battery")
    tracker = DummyEvidenceTracker([req])
    tracker.add_passage(
        Passage(passage_id="p1", text="The iPhone 15 comes in five colors.", byte_size=40)
    )
    assert tracker.is_sufficient() is False
    assert tracker.frac_satisfied() == 0.0


def test_add_passage_is_idempotent():
    req = Requirement(entity="iphone 15", attribute="battery")
    tracker = DummyEvidenceTracker([req])
    passage = Passage(passage_id="p1", text="iPhone 15 battery info", byte_size=20)
    tracker.add_passage(passage)
    tracker.add_passage(passage)
    assert len(tracker._passages) == 1  # noqa: SLF001 -- white-box test of dummy internals


def test_generator_abstains_with_no_evidence():
    result = DummyAnswerGenerator().generate("what is the battery life?", [])
    assert result.abstained is True
    assert result.answer is None


def test_generator_answers_with_evidence():
    passage = Passage(passage_id="p1", text="battery is 20 hours", byte_size=20)
    result = DummyAnswerGenerator().generate("what is the battery life?", [passage])
    assert result.abstained is False
    assert result.used_evidence_ids == ["p1"]
