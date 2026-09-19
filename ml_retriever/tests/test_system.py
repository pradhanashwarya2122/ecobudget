"""Phase 6 pipeline test: EcoBudgetSystem orchestration, model-free."""

import re

from ml_retriever.answer import EvidenceAnswerGenerator
from ml_retriever.bandit import STOP
from ml_retriever.retriever import ScoredPassage, requirement_to_query
from ml_retriever.system import EcoBudgetSystem
from ml_retriever.types import Passage, Requirement


def lexical_score(req, passage):
    q = set(re.findall(r"[a-z0-9]+", requirement_to_query(req).lower()))
    p = set(re.findall(r"[a-z0-9]+", passage.text.lower()))
    return len(q & p) / len(q | p) if q else 0.0


class FakeDecomposer:
    version = "fake-decomposer"
    def __init__(self, reqs):
        self._reqs = reqs
    def decompose(self, question):
        return self._reqs


class FakeRetriever:
    """Returns a fixed good passage per requirement (value present) + a distractor."""
    def retrieve(self, req, k=5):
        good = Passage(passage_id=f"{req.entity}-{req.attribute}", byte_size=120,
                       text=f"{req.entity} {req.attribute} is 799")
        bad = Passage(passage_id=f"{req.entity}-other", byte_size=80,
                      text=f"{req.entity} weighs 171 grams")
        return [ScoredPassage(good, 0.9, 0.9), ScoredPassage(bad, 0.3, 0.3)]


REQS = [Requirement("iPhone 15", "price")]


def test_pipeline_gathers_evidence_and_answers():
    sys = EcoBudgetSystem(FakeDecomposer(REQS), FakeRetriever(), EvidenceAnswerGenerator(),
                          score_fn=lexical_score, threshold=0.2)
    res = sys.run("What is the price of the iPhone 15?")
    assert res.requirements == REQS
    assert not res.abstained
    assert "799" in res.answer
    assert res.bytes_used > 0
    assert res.decisions[-1] == STOP  # terminates on STOP
    assert res.versions["decomposer"] == "fake-decomposer"
    assert res.versions["policy"] == "heuristic"


def test_immediate_stop_policy_abstains_with_no_evidence():
    class StopPolicy:
        def select_action(self, ctx, explore=False):
            return STOP
    sys = EcoBudgetSystem(FakeDecomposer(REQS), FakeRetriever(), EvidenceAnswerGenerator(),
                          policy=StopPolicy(), score_fn=lexical_score, threshold=0.2)
    res = sys.run("price?")
    assert res.bytes_used == 0
    assert res.abstained  # no evidence gathered -> answerer abstains
    assert res.versions["policy"] == "bandit"
