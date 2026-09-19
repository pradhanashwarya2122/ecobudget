"""
Trains a LinUCB bandit and saves the weights from the epoch with the
best validation reward, not the final epoch. This prevents the bandit
from drifting toward a degenerate policy that scores well on train
but fails on unseen tasks.
"""
import os
import random
import numpy as np

from backend.adaptive_controller.bandit import LinUCBBandit
from backend.adaptive_controller.features import ContextVectorBuilder
from backend.adaptive_controller.rewards import compute_reward
from backend.adaptive_controller.simulator import SimulatedEpisode
from backend.adaptive_controller.task_generator import generate_task_pool, split_tasks
from backend.adaptive_controller.actions import Action
from backend.adaptive_controller.controller import Decision

N_FEATURES = 10
ALPHA      = 0.8
N_EPOCHS   = 20
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "results", "bandit_weights.npz")


def train_one_episode(bandit: LinUCBBandit, builder: ContextVectorBuilder, task, max_steps=None):
    builder.reset()
    episode = SimulatedEpisode(task)
    limit = max_steps or len(task.steps)
    total_reward = 0.0
    action_counts = {Action.STOP: 0, Action.RETRIEVE: 0}

    while not episode.done and episode.step_index < limit:
        context = episode.current_context()
        x = builder.build(context)
        # No evidence exists before the first retrieval.
        # Therefore STOP is not a valid action at step 0.
        if episode.step_index == 0:
            decision = Decision(
                action=Action.RETRIEVE,
                confidence=1.0,
                reason="mandatory initial retrieval: no evidence available yet",
            )
        else:
            decision = bandit.decide(x)

        action_counts[decision.action] += 1

        bytes_of_step = 0
        if decision.action == Action.RETRIEVE and episode.step_index < len(task.steps):
            bytes_of_step = task.steps[episode.step_index].bytes

        reward = compute_reward(
            action=decision.action,
            step_index=episode.step_index,
            bytes_of_step=bytes_of_step,
            ground_truth_correct_at_step=task.ground_truth_correct_at_step,
        ).total
        total_reward += reward
        bandit.update(x, decision.action, reward)

        if decision.action == Action.STOP:
            break
        if episode.step() is None:
            break

    return total_reward, action_counts


def _val_reward(bandit, builder, val_tasks):
    total = 0.0
    for task in val_tasks:
        builder.reset()
        episode = SimulatedEpisode(task)
        while not episode.done:
            context = episode.current_context()
            x = builder.build(context)
            # No evidence exists before the first retrieval.
            if episode.step_index == 0:
                decision = Decision(
                    action=Action.RETRIEVE,
                    confidence=1.0,
                    reason="mandatory initial retrieval: no evidence available yet",
                )
            else:
                decision = bandit.decide(x)

            bytes_of_step = 0
            if decision.action == Action.RETRIEVE and episode.step_index < len(task.steps):
                bytes_of_step = task.steps[episode.step_index].bytes
            total += compute_reward(
                decision.action, episode.step_index, bytes_of_step,
                task.ground_truth_correct_at_step,
            ).total
            if decision.action == Action.STOP or episode.step() is None:
                break
    return total


def _val_success_rate(bandit, builder, val_tasks):
    """Fraction of val tasks where bandit stops at or after ground truth step."""
    correct = 0
    for task in val_tasks:
        builder.reset()
        episode = SimulatedEpisode(task)
        while not episode.done:
            context = episode.current_context()
            x = builder.build(context)
            # No evidence exists before the first retrieval.
            if episode.step_index == 0:
                decision = Decision(
                    action=Action.RETRIEVE,
                    confidence=1.0,
                    reason="mandatory initial retrieval: no evidence available yet",
                )
            else:
                decision = bandit.decide(x)

            if decision.action == Action.STOP or episode.step() is None:
                break
        if episode.step_index >= task.ground_truth_correct_at_step:
            correct += 1
    return correct / len(val_tasks)


def train():
    tasks = generate_task_pool(n_per_difficulty=20, seed=42)
    train_tasks, val_tasks, test_tasks = split_tasks(tasks)

    bandit = LinUCBBandit(n_features=N_FEATURES, alpha=ALPHA)
    builder = ContextVectorBuilder()

    print(f"Training on {len(train_tasks)} tasks, validating on {len(val_tasks)}, "
          f"holding out {len(test_tasks)} for final test.\n")
    print(f"Hyperparameters: alpha={ALPHA}, n_epochs={N_EPOCHS}\n")

    shuffle_rng = random.Random(0)
    best_val_reward = float("-inf")
    best_val_success = 0.0
    best_weights = None
    best_epoch = 0

    for epoch in range(N_EPOCHS):
        epoch_tasks = train_tasks[:]
        shuffle_rng.shuffle(epoch_tasks)

        epoch_reward = 0.0
        epoch_action_counts = {Action.STOP: 0, Action.RETRIEVE: 0}
        for task in epoch_tasks:
            r, counts = train_one_episode(bandit, builder, task)
            epoch_reward += r
            for arm in epoch_action_counts:
                epoch_action_counts[arm] += counts[arm]

        vr = _val_reward(bandit, builder, val_tasks)
        vs = _val_success_rate(bandit, builder, val_tasks)

        # Select the highest-reward policy among checkpoints that maintain
        # at least 80% validation task success.
        MIN_VAL_SUCCESS = 0.80
        if vs >= MIN_VAL_SUCCESS and vr > best_val_reward:
            best_val_reward = vr
            best_val_success = vs
            best_epoch = epoch + 1
            best_weights = {
                arm: (bandit.A[arm].copy(), bandit.b[arm].copy())
                for arm in (Action.STOP, Action.RETRIEVE)
            }

        stop_frac = epoch_action_counts[Action.STOP] / max(
            epoch_action_counts[Action.STOP] + epoch_action_counts[Action.RETRIEVE], 1
        )
        marker = " ← best" if epoch + 1 == best_epoch else ""
        print(f"epoch {epoch + 1:>2}/{N_EPOCHS}  "
              f"train={epoch_reward:>10.2f}  "
              f"val_reward={vr:>8.2f}  val_success={vs:.0%}  "
              f"STOP={epoch_action_counts[Action.STOP]:>4}  "
              f"RETRIEVE={epoch_action_counts[Action.RETRIEVE]:>4}  "
              f"stop_frac={stop_frac:.2f}{marker}")

    # Restore best weights before saving
    for arm, (A, b) in best_weights.items():
        bandit.A[arm] = A
        bandit.b[arm] = b

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    save_dict = {}
    for arm in (Action.STOP, Action.RETRIEVE):
        save_dict[f"A_{arm.name}"] = bandit.A[arm]
        save_dict[f"b_{arm.name}"] = bandit.b[arm]
    np.savez(WEIGHTS_PATH, **save_dict)
    print(f"\nSaved best weights (epoch {best_epoch}, "
          f"val_success={best_val_success:.0%}, val_reward={best_val_reward:.2f}) "
          f"to {WEIGHTS_PATH}")

    return bandit, (train_tasks, val_tasks, test_tasks)


if __name__ == "__main__":
    train()