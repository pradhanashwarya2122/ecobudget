from backend.adaptive_controller.rewards import compute_reward
from backend.adaptive_controller.actions import Action


def test_stopping_with_enough_evidence_is_rewarded():
    r = compute_reward(Action.STOP, step_index=2, bytes_of_step=0, ground_truth_correct_at_step=2)
    assert r.total > 0


def test_stopping_too_early_is_penalized():
    r = compute_reward(Action.STOP, step_index=1, bytes_of_step=0, ground_truth_correct_at_step=4)
    assert r.total < 0


def test_unnecessary_retrieval_is_penalized_more_than_needed_retrieval():
    needed = compute_reward(Action.RETRIEVE, step_index=1, bytes_of_step=2000, ground_truth_correct_at_step=4)
    unnecessary = compute_reward(Action.RETRIEVE, step_index=5, bytes_of_step=2000, ground_truth_correct_at_step=4)
    assert unnecessary.total < needed.total


def test_more_bytes_costs_more():
    small = compute_reward(Action.RETRIEVE, step_index=0, bytes_of_step=500, ground_truth_correct_at_step=4)
    large = compute_reward(Action.RETRIEVE, step_index=0, bytes_of_step=5000, ground_truth_correct_at_step=4)
    assert large.total < small.total