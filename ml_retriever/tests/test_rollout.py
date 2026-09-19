"""Model-free tests for the Phase 5 episode environment and baselines,
using fake candidates + a lexical tracker scorer + the real judge."""

import re

from ml_retriever.answer import EvidenceAnswerGenerator
from ml_retriever.bandit import RETRIEVE, STOP
from ml_retriever.rollout import (
    Candidate,
    decide_heuristic,
    decide_normal,
    make_fixed_budget_decider,
    run_episode,
)
from ml_retriever.retriever import requirement_to_query
from ml_retriever.types import Passage, Requirement


def lexical_score(req: Requirement, passage: Passage) -> float:
    q = set(re.findall(r"[a-z0-9]+", requirement_to_query(req).lower()))
    p = set(re.findall(r"[a-z0-9]+", passage.text.lower()))
    return len(q & p) / len(q | p) if q else 0.0


def _passage(pid, text, b=100):
    return Passage(passage_id=pid, text=text, byte_size=b)


# Two-requirement task; each requirement has a "good" candidate (contains the
# value) ranked first and a distractor second.
TASK = {
    "id": "X1",
    "question": "Compare iPhone 15 and Galaxy S24 on price.",
    "decomposed_requirements": [
        {"entity": "iPhone 15", "attribute": "price"},
        {"entity": "Galaxy S24", "attribute": "price"},
    ],
    "ground_truth": {"required_facts": ["799", "899"], "match_threshold": 1.0},
    "answer_type": "comparison",
}

CANDIDATES = {
    ("iphone 15", "price"): [
        Candidate(_passage("a_price", "iPhone 15 price is 799 dollars", 120), 0.9),
        Candidate(_passage("a_weight", "iPhone 15 weighs 171 grams", 80), 0.3),
    ],
    ("galaxy s24", "price"): [
        Candidate(_passage("b_price", "Galaxy S24 price is 899 dollars", 120), 0.9),
        Candidate(_passage("b_weight", "Galaxy S24 weighs 168 grams", 80), 0.3),
    ],
}

AGEN = EvidenceAnswerGenerator()


def _run(decide, lam=0.5, threshold=0.2):
    return run_episode(TASK, CANDIDATES, lexical_score, AGEN, decide, lam=lam, threshold=threshold)


class TestBaselines:
    def test_normal_retrieves_one_per_requirement_then_stops(self):
        r = _run(decide_normal)
        assert r.n_retrieves == 2  # one passage per requirement
        assert r.success  # both price facts gathered

    def test_fixed_budget_retrieves_exactly_n(self):
        r = _run(make_fixed_budget_decider(1))
        assert r.n_retrieves == 1
        # only one requirement's value gathered -> comparison not fully satisfied
        assert not r.success

    def test_heuristic_stops_when_sufficient(self):
        # With a low threshold, the two good (price) candidates satisfy both
        # requirements; heuristic should stop right after gathering them.
        r = _run(decide_heuristic, threshold=0.2)
        assert r.success
        assert r.n_retrieves == 2

    def test_immediate_stop_fails_and_pays_no_bytes(self):
        r = _run(lambda ctx, st: STOP)
        assert r.total_bytes == 0
        assert not r.success


class TestRewardWiring:
    def test_success_with_fewer_bytes_beats_success_with_more(self):
        lean = _run(decide_normal)              # 2 retrieves
        heavy = _run(make_fixed_budget_decider(4))  # retrieves everything
        assert lean.success and heavy.success
        assert lean.reward > heavy.reward  # byte penalty rewards stopping early


class TestPerStepReward:
    def _rewards(self, res):
        # map action -> list of step rewards
        out = {RETRIEVE: [], STOP: []}
        for _ctx, action, r in res.trajectory:
            out[action].append(r)
        return out

    def test_useful_retrieve_positive_stop_success_terminal(self):
        # normal run: both RETRIEVEs fill a needed requirement (coverage up),
        # then STOP with success -> STOP reward 1.0, RETRIEVE rewards positive.
        res = run_episode(TASK, CANDIDATES, lexical_score, AGEN, decide_normal,
                          lam=0.5, threshold=0.2, reward_mode="per_step")
        r = self._rewards(res)
        assert r[STOP] == [1.0]
        assert all(x > 0 for x in r[RETRIEVE])  # coverage gain beat byte cost

    def test_redundant_retrieve_is_negative(self):
        # always-RETRIEVE pulls distractors after both requirements are satisfied;
        # those add no coverage and only cost bytes -> negative step reward.
        res = run_episode(TASK, CANDIDATES, lexical_score, AGEN, lambda c, s: RETRIEVE,
                          lam=0.5, threshold=0.2, reward_mode="per_step")
        r = self._rewards(res)
        assert any(x < 0 for x in r[RETRIEVE])  # at least one wasteful retrieval

    def test_terminal_mode_credits_same_reward_to_all_steps(self):
        res = run_episode(TASK, CANDIDATES, lexical_score, AGEN, decide_normal,
                          lam=0.5, threshold=0.2, reward_mode="terminal")
        step_rewards = [r for _c, _a, r in res.trajectory]
        assert len(set(step_rewards)) == 1
        assert step_rewards[0] == res.reward


class TestTermination:
    def test_retrieve_forced_to_stop_when_nothing_left(self):
        # A decider that always says RETRIEVE must still terminate once all
        # candidates are exhausted (no infinite loop).
        r = _run(lambda ctx, st: RETRIEVE)
        assert r.n_retrieves == 4  # all candidates across both requirements
        assert r.n_steps <= 6
