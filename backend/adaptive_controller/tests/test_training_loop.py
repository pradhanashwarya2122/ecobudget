from backend.adaptive_controller.bandit import LinUCBBandit
from backend.adaptive_controller.features import ContextVectorBuilder
from backend.adaptive_controller.task_generator import generate_task_pool, split_tasks
from backend.experiments.train_bandit import train_one_episode, ALPHA
from backend.adaptive_controller.actions import Action
import numpy as np


def test_train_one_episode_returns_float_reward():
    tasks = generate_task_pool(n_per_difficulty=3, seed=1)
    bandit = LinUCBBandit(n_features=9, alpha=ALPHA)
    builder = ContextVectorBuilder()
    reward, counts = train_one_episode(bandit, builder, tasks[0])
    assert isinstance(reward, float)


def test_train_one_episode_returns_action_counts():
    tasks = generate_task_pool(n_per_difficulty=3, seed=1)
    bandit = LinUCBBandit(n_features=9, alpha=ALPHA)
    builder = ContextVectorBuilder()
    reward, counts = train_one_episode(bandit, builder, tasks[0])
    assert Action.STOP in counts and Action.RETRIEVE in counts
    assert counts[Action.STOP] + counts[Action.RETRIEVE] >= 1


def test_training_updates_bandit_weights():
    """
    After enough episodes at production alpha, STOP must be chosen at
    least once. Uses n_per_difficulty=20 and two passes so the bandit
    accumulates enough history to explore both arms.
    """
    tasks = generate_task_pool(n_per_difficulty=20, seed=1)
    bandit = LinUCBBandit(n_features=9, alpha=ALPHA)
    builder = ContextVectorBuilder()

    A_before = bandit.A[Action.STOP].copy()

    # Two passes (epochs) to give the bandit enough signal to try STOP.
    for _ in range(2):
        for task in tasks:
            train_one_episode(bandit, builder, task)

    assert not np.allclose(A_before, bandit.A[Action.STOP]), (
        "A[STOP] never changed across 2 epochs of 60 tasks — "
        "bandit chose RETRIEVE every time. Check ALPHA and task generator."
    )


def test_no_task_has_ground_truth_at_step_zero():
    """
    Ensures the task generator never produces a task where stopping
    immediately (before any retrieval) is correct. If it did, 'always
    stop' would be an optimal policy and the bandit couldn't learn
    meaningful retrieval behaviour.
    """
    tasks = generate_task_pool(n_per_difficulty=20, seed=42)
    for t in tasks:
        assert t.ground_truth_correct_at_step >= 1, (
            f"Task {t.task_id} has ground_truth_correct_at_step=0 — "
            f"stopping immediately is trivially optimal for this task."
        )


def test_split_tasks_produces_nonempty_splits_for_training():
    tasks = generate_task_pool(n_per_difficulty=20, seed=42)
    train, val, test = split_tasks(tasks)
    assert len(train) > 0 and len(val) > 0 and len(test) > 0