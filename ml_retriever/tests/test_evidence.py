"""Phase 4 tests: EvidenceCoverageTracker state machine, model-free.

Uses a transparent token-overlap scorer instead of MiniLM, so the "entity
mentioned but attribute absent -> not satisfied" case is a real behavioral
test (not a model download, and not a tautology about injected numbers).
"""

import re

import pytest

from ml_retriever.evidence import EvidenceCoverageTracker, gather_evidence
from ml_retriever.interfaces import EvidenceTracker
from ml_retriever.retriever import requirement_to_query
from ml_retriever.types import Passage, Requirement


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def lexical_score(req: Requirement, passage: Passage) -> float:
    """Jaccard overlap between the requirement query tokens and passage tokens.
    Transparent stand-in for embedding similarity: a passage must actually
    contain the attribute words to score high, so an entity-only passage stays
    low -- exactly the property Phase 4 cares about."""
    q = _tokens(requirement_to_query(req))
    p = _tokens(passage.text)
    if not q:
        return 0.0
    return len(q & p) / len(q | p)


def _p(pid, text, byte_size=100):
    return Passage(passage_id=pid, text=text, byte_size=byte_size)


REQ_PRICE = Requirement(entity="iPhone 15", attribute="price")


class TestProtocolConformance:
    def test_satisfies_evidence_tracker_protocol(self):
        t = EvidenceCoverageTracker([REQ_PRICE], score_fn=lexical_score)
        assert isinstance(t, EvidenceTracker)


class TestEmptyRequirements:
    def test_no_requirements_is_trivially_sufficient(self):
        t = EvidenceCoverageTracker([], score_fn=lexical_score)
        assert t.is_sufficient()
        assert t.frac_satisfied() == 1.0
        assert t.coverage() == 1.0
        assert t.unanswered_requirements() == []


class TestSufficiencyGating:
    def test_starts_unsatisfied(self):
        t = EvidenceCoverageTracker([REQ_PRICE], threshold=0.3, score_fn=lexical_score)
        assert not t.is_sufficient()
        assert t.frac_satisfied() == 0.0
        assert t.unanswered_requirements() == [REQ_PRICE]

    def test_relevant_passage_satisfies(self):
        t = EvidenceCoverageTracker([REQ_PRICE], threshold=0.3, score_fn=lexical_score)
        t.add_passage(_p("p1", "The iPhone 15 price starts at $799."))
        assert t.is_sufficient()
        assert t.frac_satisfied() == 1.0
        assert t.unanswered_requirements() == []

    def test_entity_only_passage_does_not_satisfy(self):
        # Mentions the entity but never the attribute ("price") -> must NOT satisfy.
        t = EvidenceCoverageTracker([REQ_PRICE], threshold=0.3, score_fn=lexical_score)
        t.add_passage(_p("p1", "The iPhone 15 weighs about 171 grams."))
        assert not t.is_sufficient()
        assert REQ_PRICE in t.unanswered_requirements()

    def test_weak_then_strong_passage_transitions(self):
        t = EvidenceCoverageTracker([REQ_PRICE], threshold=0.3, score_fn=lexical_score)
        t.add_passage(_p("weak", "The iPhone 15 weighs 171 grams."))
        assert not t.is_sufficient()
        t.add_passage(_p("strong", "The iPhone 15 price is $799."))
        assert t.is_sufficient()


class TestIdempotency:
    def test_same_passage_id_added_twice_is_noop(self):
        t = EvidenceCoverageTracker([REQ_PRICE], threshold=0.3, score_fn=lexical_score)
        t.add_passage(_p("p1", "The iPhone 15 price is $799."))
        before = t.best_score(REQ_PRICE)
        # same id, different (higher-overlap) text must be ignored
        t.add_passage(_p("p1", "iPhone 15 price price price $799 128gb configuration"))
        assert t.best_score(REQ_PRICE) == before


class TestMultiRequirementCoverage:
    REQS = [
        Requirement(entity="iPhone 15", attribute="price"),
        Requirement(entity="iPhone 15", attribute="battery"),
    ]

    def test_partial_coverage(self):
        t = EvidenceCoverageTracker(self.REQS, threshold=0.3, score_fn=lexical_score)
        t.add_passage(_p("p1", "The iPhone 15 price is $799."))
        assert t.frac_satisfied() == 0.5
        assert t.frac_remaining() == 0.5
        assert not t.is_sufficient()
        assert [r.attribute for r in t.unanswered_requirements()] == ["battery"]

    def test_full_coverage(self):
        t = EvidenceCoverageTracker(self.REQS, threshold=0.3, score_fn=lexical_score)
        t.add_passage(_p("p1", "The iPhone 15 price is $799."))
        t.add_passage(_p("p2", "The iPhone 15 battery is 3349 mAh."))
        assert t.is_sufficient()
        assert t.frac_satisfied() == 1.0

    def test_coverage_is_continuous_not_hard_count(self):
        # A below-threshold best score still contributes partial coverage,
        # so coverage() can be > 0 while frac_satisfied() is still 0.
        t = EvidenceCoverageTracker([REQ_PRICE], threshold=0.9, score_fn=lexical_score)
        t.add_passage(_p("p1", "The iPhone 15 price is $799."))
        assert t.frac_satisfied() == 0.0
        assert 0.0 < t.coverage() < 1.0


class TestDedup:
    def test_duplicate_requirements_collapse(self):
        reqs = [REQ_PRICE, Requirement(entity="iPhone 15", attribute="price")]
        t = EvidenceCoverageTracker(reqs, score_fn=lexical_score)
        assert len(t.requirements) == 1


class TestGatherEvidence:
    def test_wires_retriever_output_into_tracker(self):
        # Fake retriever returning a fixed passage per requirement, so the
        # Phase 3 -> Phase 4 wiring is covered without a model.
        class _Scored:
            def __init__(self, passage):
                self.passage = passage

        class FakeRetriever:
            def retrieve(self, req, k=5):
                return [_Scored(_p(f"{req.entity}-{req.attribute}",
                                   f"{req.entity} {req.attribute} evidence text"))]

        reqs = [REQ_PRICE]
        t = gather_evidence(reqs, FakeRetriever(), k=1, threshold=0.2, score_fn=lexical_score)
        assert t.is_sufficient()
