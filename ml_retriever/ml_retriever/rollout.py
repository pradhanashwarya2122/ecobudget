"""Phase 5 environment: run one retrieval episode under a decision policy.

One episode = one task. At each step the decider chooses STOP or RETRIEVE_TOP1
given the context; RETRIEVE adds the next unseen candidate passage for the
highest-priority unanswered requirement to the evidence tracker; STOP generates
an answer from the gathered evidence and the judge scores it. The same runner
drives the learned bandit and every baseline (normal / fixed-budget /
heuristic), so Phase 7's comparison is apples-to-apples.

Kept model-agnostic: candidate passages per requirement are passed in
(the training script builds them from the Phase 3 retriever), and the tracker's
score_fn is injectable -- so the whole loop is unit-testable without any model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .bandit import RETRIEVE, STOP, compute_reward, featurize
from .evidence import EvidenceCoverageTracker
from .judge import judge_task
from .types import Passage, Requirement

# A decider maps (context_vector, state) -> action (STOP / RETRIEVE).
Decider = Callable[[np.ndarray, "EpisodeState"], int]


@dataclass
class Candidate:
    passage: Passage
    similarity: float


@dataclass
class EpisodeState:
    requirements: list[Requirement]
    candidates: dict[tuple[str, str], list[Candidate]]  # req.key() -> ranked candidates
    tracker: EvidenceCoverageTracker
    max_bytes: int
    step: int = 0
    bytes_used: int = 0
    added_ids: set[str] = field(default_factory=set)
    # how many candidates already consumed per requirement
    cursor: dict[tuple[str, str], int] = field(default_factory=dict)

    def top_unanswered(self) -> Optional[Requirement]:
        unanswered = self.tracker.unanswered_requirements()
        return unanswered[0] if unanswered else None

    def next_candidate(self, req: Requirement) -> Optional[Candidate]:
        cands = self.candidates.get(req.key(), [])
        i = self.cursor.get(req.key(), 0)
        while i < len(cands) and cands[i].passage.passage_id in self.added_ids:
            i += 1
        self.cursor[req.key()] = i
        return cands[i] if i < len(cands) else None

    def retrieval_target(self) -> Optional[Requirement]:
        """Which requirement a RETRIEVE pulls for. Prefers a still-unanswered
        requirement that has a candidate left (make progress); otherwise any
        requirement with a candidate left -- so a decider that keeps asking can
        over-retrieve redundant passages and pay for it in bytes. That wasted-
        byte possibility is exactly what the bandit learns to avoid."""
        for req in self.tracker.unanswered_requirements():
            if self.next_candidate(req) is not None:
                return req
        for req in self.requirements:
            if self.next_candidate(req) is not None:
                return req
        return None

    def has_retrievable(self) -> bool:
        return self.retrieval_target() is not None


@dataclass
class EpisodeResult:
    task_id: str
    success: bool
    score: float
    total_bytes: int
    n_steps: int
    n_retrieves: int
    reward: float  # episode objective (success - lam*total_bytes/max_bytes), for reporting
    answer: Optional[str]
    # (context_vector, action, step_reward) per step. step_reward is what the
    # policy trains on; it equals the episode objective in "terminal" mode, or a
    # localized per-step credit in "per_step" mode.
    trajectory: list[tuple[np.ndarray, int, float]]


def build_context(state: EpisodeState) -> np.ndarray:
    tracker = state.tracker
    n_req = len(state.requirements)
    unanswered = tracker.unanswered_requirements()

    top_sim = mean_sim = 0.0
    has_cand = 0.0
    req = state.top_unanswered()
    if req is not None:
        cands = state.candidates.get(req.key(), [])
        remaining = [c for c in cands if c.passage.passage_id not in state.added_ids]
        if remaining:
            has_cand = 1.0
            sims = [c.similarity for c in remaining]
            top_sim = max(sims)
            mean_sim = float(np.mean(sims))

    return featurize({
        "frac_satisfied": tracker.frac_satisfied(),
        "frac_remaining": tracker.frac_remaining(),
        "coverage": tracker.coverage(),
        "n_requirements": n_req,
        "n_unanswered": len(unanswered),
        "passages_added": len(state.added_ids),
        "bytes_used_frac": state.bytes_used / max(state.max_bytes, 1),
        "step": state.step,
        "next_cand_top_sim": top_sim,
        "next_cand_mean_sim": mean_sim,
        "has_unretrieved_candidate": has_cand,
    })


def _gathered_passages(state: EpisodeState) -> list[Passage]:
    by_id = {c.passage.passage_id: c.passage
             for cands in state.candidates.values() for c in cands}
    return [by_id[pid] for pid in state.added_ids if pid in by_id]


def run_episode(
    task: dict,
    candidates: dict[tuple[str, str], list[Candidate]],
    score_fn,
    answer_gen,
    decide: Decider,
    lam: float = 0.5,
    threshold: float = 0.3,
    max_steps: int = 30,
    reward_mode: str = "terminal",
) -> EpisodeResult:
    """reward_mode:
      - "terminal": every step is credited the episode objective
        (success - lam*total_bytes/max_bytes). Coarse Monte-Carlo credit.
      - "per_step": localized credit -- a RETRIEVE earns the marginal coverage
        it adds minus its own byte cost (redundant retrievals go negative), and
        STOP earns the realized judge success. This lets the policy tell a
        premature STOP or a wasteful RETRIEVE apart, which terminal credit cannot.
    """
    requirements = [Requirement(entity=r["entity"], attribute=r["attribute"])
                    for r in task["decomposed_requirements"]]
    tracker = EvidenceCoverageTracker(requirements, threshold=threshold, score_fn=score_fn)

    # max payload = every candidate of every requirement (fraction stays <= 1)
    max_bytes = sum(c.passage.byte_size for cands in candidates.values() for c in cands) or 1

    state = EpisodeState(requirements=requirements, candidates=candidates,
                         tracker=tracker, max_bytes=max_bytes)

    # each step recorded as (context, action, delta_coverage, incremental_bytes)
    steps: list[tuple[np.ndarray, int, float, int]] = []
    n_retrieves = 0

    while state.step < max_steps:
        ctx = build_context(state)
        action = decide(ctx, state)
        # Can't retrieve with nothing left -> force STOP so episodes terminate.
        if action == RETRIEVE and not state.has_retrievable():
            action = STOP

        if action == STOP:
            steps.append((ctx, STOP, 0.0, 0))
            break

        cov_before = state.tracker.coverage()
        req = state.retrieval_target()
        cand = state.next_candidate(req)
        tracker.add_passage(cand.passage)
        state.added_ids.add(cand.passage.passage_id)
        state.bytes_used += cand.passage.byte_size
        delta_cov = state.tracker.coverage() - cov_before
        steps.append((ctx, RETRIEVE, delta_cov, cand.passage.byte_size))
        n_retrieves += 1
        state.step += 1

    answer_result = answer_gen.generate(task["question"], _gathered_passages(state))
    answer_text = answer_result.answer or ""
    success, score = judge_task(answer_text, task)
    objective = compute_reward(success, state.bytes_used, max_bytes, lam)

    trajectory: list[tuple[np.ndarray, int, float]] = []
    for ctx, action, delta_cov, inc_bytes in steps:
        if reward_mode == "per_step":
            if action == STOP:
                step_reward = 1.0 if success else 0.0
            else:
                step_reward = delta_cov - lam * (inc_bytes / max_bytes)
        else:  # terminal
            step_reward = objective
        trajectory.append((ctx, action, step_reward))

    return EpisodeResult(
        task_id=task.get("id", "?"), success=success, score=score,
        total_bytes=state.bytes_used, n_steps=len(steps), n_retrieves=n_retrieves,
        reward=objective, answer=answer_text, trajectory=trajectory,
    )


def build_candidates(requirements, retriever, k: int = 5) -> dict[tuple[str, str], list[Candidate]]:
    """Per-requirement ranked candidate passages from the Phase 3 retriever.
    Cached by requirement key so shared requirements are retrieved once."""
    out: dict[tuple[str, str], list[Candidate]] = {}
    for req in requirements:
        if req.key() in out:
            continue
        out[req.key()] = [
            Candidate(passage=s.passage, similarity=s.similarity)
            for s in retriever.retrieve(req, k=k)
        ]
    return out


# --- baseline deciders (the conditions the bandit must be compared against) ---

def decide_full(ctx: np.ndarray, state: EpisodeState) -> int:
    """Retrieve-everything: pull every candidate for every requirement (the
    max-payload 'load all relevant evidence' reference)."""
    return RETRIEVE if state.has_retrievable() else STOP


def decide_one_per_req(ctx: np.ndarray, state: EpisodeState) -> int:
    """One passage per requirement, then stop (minimal requirement-aware
    retrieval). NOTE: this is top-1-per-requirement, NOT 'retrieve everything'."""
    added_reqs = {
        req.key()
        for req in state.requirements
        if any(c.passage.passage_id in state.added_ids
               for c in state.candidates.get(req.key(), []))
    }
    return RETRIEVE if len(added_reqs) < len(state.requirements) else STOP


def decide_heuristic(ctx: np.ndarray, state: EpisodeState) -> int:
    """Simple adaptive heuristic (the EcoBudgetController analog, Baseline 3):
    stop as soon as the tracker says evidence is sufficient."""
    return STOP if state.tracker.is_sufficient() else RETRIEVE


# Backward-compatible alias (the function is top-1-per-requirement; older
# scripts imported it as decide_normal). Phase 7 uses the accurate label.
decide_normal = decide_one_per_req


def make_fixed_budget_decider(max_retrieves: int) -> Decider:
    """Fixed-budget: retrieve a fixed number of passages regardless of need."""
    def decide(ctx: np.ndarray, state: EpisodeState) -> int:
        return RETRIEVE if len(state.added_ids) < max_retrieves else STOP
    return decide


def decide_adaptive_rag(ctx: np.ndarray, state: EpisodeState) -> int:
    """Phase E external baseline: a faithful analog of Adaptive-RAG (Jeong et
    al., NAACL 2024), which routes a query by predicted COMPLEXITY to a fixed
    strategy -- simple queries get single-step retrieval, complex queries get
    iterative multi-step retrieval. Here the complexity signal is the number of
    requirements (single-hop vs multi-hop), which is exactly what Adaptive-RAG's
    classifier is trained to predict:
      - 1 requirement  -> single-step: retrieve one passage, then stop.
      - >=2 requirements -> multi-step: retrieve iteratively until the evidence
        tracker is sufficient (adaptive), like its multi-hop branch.
    This is a reimplementation of the routing idea, not the original model, and
    is labelled as such in Phase 7. It differs from `decide_heuristic` (which is
    adaptive for ALL queries) and from `decide_one_per_req` (always one each)."""
    if len(state.requirements) <= 1:
        return RETRIEVE if len(state.added_ids) < 1 else STOP
    return STOP if state.tracker.is_sufficient() else RETRIEVE


def make_byte_budget_decider(budget_bytes: int) -> Decider:
    """Fixed-byte-budget with STRICT enforcement: only retrieve the next passage
    if it fits within the remaining budget, so the cumulative payload never
    exceeds `budget_bytes` (no off-by-one overrun)."""
    def decide(ctx: np.ndarray, state: EpisodeState) -> int:
        req = state.retrieval_target()
        if req is None:
            return STOP
        cand = state.next_candidate(req)
        if cand is None:
            return STOP
        if state.bytes_used + cand.passage.byte_size <= budget_bytes:
            return RETRIEVE
        return STOP
    return decide
