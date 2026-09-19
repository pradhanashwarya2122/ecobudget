from dataclasses import dataclass
from typing import Optional

from .actions import Action
from . import config


@dataclass
class RewardBreakdown:
    total: float
    correctness_component: float
    byte_cost_component: float
    retrieval_cost_component: float
    unnecessary_retrieval_component: float


def compute_reward(
    action: Action,
    step_index: int,
    bytes_of_step: int,
    ground_truth_correct_at_step: Optional[int],
) -> RewardBreakdown:
    """
    Reward for ONE controller decision. Used only offline, during
    bandit training/evaluation (Phase 6/7) — never inside decide()
    itself, which must not see ground truth.

    step_index: how many retrieval steps had already happened BEFORE
        this decision (i.e. Context.steps_taken at decision time).
    bytes_of_step: for RETRIEVE, the size of the piece of evidence
        about to be fetched. For STOP, always pass 0 — stopping fetches
        nothing new; earlier bytes were already charged when fetched.
    ground_truth_correct_at_step: earliest step_index at which enough
        evidence exists to answer correctly. Eval-only data.
    """
    has_enough_evidence = (
        ground_truth_correct_at_step is not None
        and step_index >= ground_truth_correct_at_step
    )

    correctness = 0.0
    byte_cost = 0.0
    retrieval_cost = 0.0
    unnecessary = 0.0

    if action == Action.STOP:
        correctness = (
            config.CORRECT_STOP_REWARD if has_enough_evidence
            else -config.WRONG_STOP_PENALTY
        )
    else:  # RETRIEVE
        byte_cost = -config.BYTE_COST_PER_KB * (bytes_of_step / 1000)
        retrieval_cost = -config.RETRIEVAL_STEP_COST
        if has_enough_evidence:
            unnecessary = -config.UNNECESSARY_RETRIEVAL_PENALTY

    total = correctness + byte_cost + retrieval_cost + unnecessary
    return RewardBreakdown(
        total=total,
        correctness_component=correctness,
        byte_cost_component=byte_cost,
        retrieval_cost_component=retrieval_cost,
        unnecessary_retrieval_component=unnecessary,
    )