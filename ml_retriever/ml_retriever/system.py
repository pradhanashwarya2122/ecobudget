"""Phase 6: EcoBudgetSystem -- the full pipeline, end to end.

question -> decompose -> requirements -> retrieve-under-policy (STOP/RETRIEVE)
-> evidence -> generate answer -> measure bytes. No ground truth anywhere: this
is the deployed path, distinct from the training/eval rollout that judges
against gold. The answer generator is frozen and its `generator_version` is
recorded on every run for reproducibility (plan.md).

Retrieval reuses the Phase 3 retriever for similarity-aware candidates and the
Phase 5 rollout primitives for the stop/continue loop. If a trained policy is
given it drives retrieval; otherwise the simple sufficiency heuristic does.
Everything is injectable, so the orchestration is unit-tested without models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .bandit import RETRIEVE, STOP
from .evidence import EvidenceCoverageTracker, QAScorer
from .rollout import Candidate, EpisodeState, build_context, build_candidates, decide_heuristic
from .types import Passage, Requirement


@dataclass
class SystemResult:
    question: str
    requirements: list[Requirement]
    answer: Optional[str]
    abstained: bool
    confidence: float
    used_evidence_ids: list[str]
    bytes_used: int
    n_retrieves: int
    decisions: list[int]
    versions: dict


class EcoBudgetSystem:
    def __init__(
        self,
        decomposer,
        retriever,
        answer_generator,
        policy=None,
        normalizer=None,
        score_fn=None,
        k: int = 5,
        threshold: float = 0.3,
        max_steps: int = 30,
        answer_mode: str = "joint",
        decider=None,
    ):
        # decider override: a callable(ctx, state)->action. When set it drives
        # retrieval instead of policy/heuristic (used to run Phase 7's fixed /
        # normal conditions through the same pipeline). Frozen configs only.
        self._decider_override = decider
        self.decomposer = decomposer
        self.retriever = retriever
        self.answer_generator = answer_generator
        self.policy = policy
        self.normalizer = normalizer
        self.score_fn = score_fn or QAScorer()
        self.k = k
        self.threshold = threshold
        self.max_steps = max_steps
        # "joint": one answerer call with the question + all evidence.
        # "per_requirement": one call per requirement with only that
        #   requirement's evidence, phrased as a single-fact question, then
        #   concatenate the facts -- removes the answerer's within-passage
        #   attribute-localization burden.
        self.answer_mode = answer_mode

    def _decide(self, ctx, state: EpisodeState) -> int:
        if self._decider_override is not None:
            return self._decider_override(ctx, state)
        if self.policy is None:
            return decide_heuristic(ctx, state)
        x = self.normalizer.transform(ctx) if self.normalizer is not None else ctx
        return self.policy.select_action(x, explore=False)

    def run(self, question: str) -> SystemResult:
        requirements = self.decomposer.decompose(question)
        candidates = build_candidates(requirements, self.retriever, k=self.k)
        tracker = EvidenceCoverageTracker(requirements, threshold=self.threshold, score_fn=self.score_fn)
        max_bytes = sum(c.passage.byte_size for cs in candidates.values() for c in cs) or 1
        state = EpisodeState(requirements=requirements, candidates=candidates,
                             tracker=tracker, max_bytes=max_bytes)

        decisions: list[int] = []
        while state.step < self.max_steps:
            ctx = build_context(state)
            action = self._decide(ctx, state)
            if action == RETRIEVE and not state.has_retrievable():
                action = STOP
            decisions.append(int(action))
            if action == STOP:
                break
            req = state.retrieval_target()
            cand = state.next_candidate(req)
            tracker.add_passage(cand.passage)
            state.added_ids.add(cand.passage.passage_id)
            state.bytes_used += cand.passage.byte_size
            state.step += 1

        by_id = {c.passage.passage_id: c.passage for cs in candidates.values() for c in cs}
        evidence = [by_id[pid] for pid in state.added_ids if pid in by_id]

        if self.answer_mode == "per_requirement":
            facts, used_ids = [], []
            for req in requirements:
                req_pids = {c.passage.passage_id for c in candidates.get(req.key(), [])}
                req_evidence = [by_id[pid] for pid in state.added_ids if pid in req_pids]
                if not req_evidence:
                    continue
                q = f"What is the {req.attribute.replace('_', ' ')} of {req.entity}?"
                r = self.answer_generator.generate(q, req_evidence)
                if r.answer:
                    facts.append(r.answer)
                    used_ids.extend(r.used_evidence_ids)
            answer = ", ".join(facts) if facts else None
            result_answer, result_abstained = answer, (answer is None)
            result_conf = 1.0 if answer else 0.0
            result_ids = used_ids
        else:
            result = self.answer_generator.generate(question, evidence)
            result_answer, result_abstained = result.answer, result.abstained
            result_conf, result_ids = result.confidence, result.used_evidence_ids

        return SystemResult(
            question=question, requirements=requirements,
            answer=result_answer, abstained=result_abstained, confidence=result_conf,
            used_evidence_ids=result_ids, bytes_used=state.bytes_used,
            n_retrieves=len(state.added_ids), decisions=decisions,
            versions={
                "decomposer": getattr(self.decomposer, "version", "?"),
                "answer_generator": getattr(self.answer_generator, "version", "?"),
                "policy": getattr(self.policy, "version", "bandit") if self.policy is not None else "heuristic",
            },
        )
