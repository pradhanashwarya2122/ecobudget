"""
Loads a trained bandit and evaluates it on the held-out test split,
producing a comparison table against fixed baselines on the SAME tasks.

LIMITATION: test set is synthetic (3 archetypes with noise), not
independent real-world tasks. Results show whether the bandit learned
the reward structure, not generalisation to unseen question types.
"""
import json
import os
import numpy as np

from backend.adaptive_controller.bandit import LinUCBBandit
from backend.adaptive_controller.baselines import FixedBudgetController, FixedThresholdController
from backend.adaptive_controller.controller import AdaptiveController
from backend.adaptive_controller.features import ContextVectorBuilder
from backend.adaptive_controller.simulator import SimulatedEpisode
from backend.adaptive_controller.task_generator import generate_task_pool, split_tasks
from backend.experiments.train_bandit import N_FEATURES, ALPHA, WEIGHTS_PATH
from backend.adaptive_controller.actions import Action
from backend.adaptive_controller.controller import Decision

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "evaluation.json")


def load_trained_bandit() -> LinUCBBandit:
    bandit = LinUCBBandit(n_features=N_FEATURES, alpha=ALPHA)
    data = np.load(WEIGHTS_PATH)
    for arm in (Action.STOP, Action.RETRIEVE):
        bandit.A[arm] = data[f"A_{arm.name}"]
        bandit.b[arm] = data[f"b_{arm.name}"]
    return bandit


def run_bandit_episode(bandit: LinUCBBandit, builder: ContextVectorBuilder, task):
    builder.reset()
    episode = SimulatedEpisode(task)
    while not episode.done:
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

        if decision.action == Action.STOP or episode.step() is None:
            break
    return {
        "final_bytes_used": episode.bytes_used,
        "final_resources_seen": episode.resources_seen,
        "stopped_at_step": episode.step_index,
    }


def _summarise(rows: list, label: str) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "policy": label,
        "n": n,
        "success_rate": sum(r["reached_sufficient_evidence"] for r in rows) / n,
        "mean_bytes": sum(r["final_bytes_used"] for r in rows) / n,
        "mean_steps": sum(r["stopped_at_step"] for r in rows) / n,
    }


def evaluate():
    tasks = generate_task_pool(n_per_difficulty=20, seed=42)
    _, _, test_tasks = split_tasks(tasks)

    bandit = load_trained_bandit()
    builder = ContextVectorBuilder()

    # "always retrieve everything" baseline: run every step, never stop early
    class AlwaysRetrieveController:
        def decide(self, context):
            from backend.adaptive_controller.controller import Decision
            return Decision(action=Action.RETRIEVE, confidence=0.0, reason="always retrieve")

    policies = {
        "always_retrieve":        AlwaysRetrieveController(),
        "fixed_budget_10kb":      FixedBudgetController(budget_bytes=10_000),
        "fixed_budget_25kb":      FixedBudgetController(budget_bytes=25_000),
        "fixed_threshold_0.75":   FixedThresholdController(confidence_threshold=0.75),
        "fixed_threshold_0.90":   FixedThresholdController(confidence_threshold=0.90),
        "heuristic_adaptive":     AdaptiveController(),
    }

    results = []

    for policy_name, policy in policies.items():
        for task in test_tasks:
            r = SimulatedEpisode(task).run(policy)
            results.append({
                "policy": policy_name,
                "task_id": r["task_id"],
                "difficulty": r["difficulty"],
                "final_bytes_used": r["final_bytes_used"],
                "stopped_at_step": r["stopped_at_step"],
                "ground_truth_correct_at_step": task.ground_truth_correct_at_step,
                "reached_sufficient_evidence": (
                    r["stopped_at_step"] >= task.ground_truth_correct_at_step
                ),
            })

    for task in test_tasks:
        r = run_bandit_episode(bandit, builder, task)
        results.append({
            "policy": "linucb_bandit",
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "final_bytes_used": r["final_bytes_used"],
            "stopped_at_step": r["stopped_at_step"],
            "ground_truth_correct_at_step": task.ground_truth_correct_at_step,
            "reached_sufficient_evidence": (
                r["stopped_at_step"] >= task.ground_truth_correct_at_step
            ),
        })

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # ── Overall summary table ────────────────────────────────────────────────
    by_policy = {}
    for r in results:
        by_policy.setdefault(r["policy"], []).append(r)

    col = "{:<24}{:<5}{:<14}{:<12}{:<10}"
    print("\n" + "=" * 65)
    print("OVERALL RESULTS (all difficulties, held-out test set)")
    print("=" * 65)
    print(col.format("policy", "n", "success_rate", "mean_bytes", "mean_steps"))
    print("-" * 65)
    for policy_name, rows in by_policy.items():
        s = _summarise(rows, policy_name)
        print(col.format(
            policy_name, s["n"],
            f"{s['success_rate']:.0%}",
            f"{s['mean_bytes']:.0f}",
            f"{s['mean_steps']:.2f}",
        ))

    # ── Per-difficulty breakdown for linucb_bandit ───────────────────────────
    print("\n" + "=" * 65)
    print("LINUCB BANDIT — breakdown by difficulty")
    print("=" * 65)
    print(col.format("difficulty", "n", "success_rate", "mean_bytes", "mean_steps"))
    print("-" * 65)
    bandit_rows = [r for r in results if r["policy"] == "linucb_bandit"]
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in bandit_rows if r["difficulty"] == diff]
        if not subset:
            continue
        s = _summarise(subset, diff)
        print(col.format(
            diff, s["n"],
            f"{s['success_rate']:.0%}",
            f"{s['mean_bytes']:.0f}",
            f"{s['mean_steps']:.2f}",
        ))

    # ── Bytes saved vs always_retrieve ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("DATA EFFICIENCY vs always_retrieve baseline")
    print("=" * 65)
    always_bytes = np.mean([r["final_bytes_used"]
                            for r in results if r["policy"] == "always_retrieve"])
    print(f"always_retrieve mean bytes: {always_bytes:.0f}")
    print()
    for policy_name, rows in by_policy.items():
        if policy_name == "always_retrieve":
            continue
        mean_bytes = np.mean([r["final_bytes_used"] for r in rows])
        saving_pct = (always_bytes - mean_bytes) / always_bytes * 100
        success = np.mean([r["reached_sufficient_evidence"] for r in rows])
        print(f"  {policy_name:<24}  bytes={mean_bytes:>7.0f}  "
              f"saved={saving_pct:>+5.1f}%  success={success:.0%}")

    print(f"\nFull results written to {RESULTS_PATH}")
    print(f"\nNote: {len(test_tasks)} synthetic test tasks from 3 archetypes "
          f"(easy/medium/hard) with randomised noise. Not real-world tasks.")

    return results


if __name__ == "__main__":
    evaluate()