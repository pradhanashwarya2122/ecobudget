"""
Runs every baseline policy over every sample task and writes a common
result table to backend/experiments/results/baselines.json.

This file format is the contract later phases (bandit training,
evaluation, ablation) all read from — keep it stable.
"""
import json
import os

from backend.adaptive_controller.baselines import FixedBudgetController, FixedThresholdController
from backend.adaptive_controller.controller import AdaptiveController
from backend.adaptive_controller.simulator import SimulatedEpisode
from backend.adaptive_controller.sample_tasks import ALL_SAMPLE_TASKS

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "baselines.json")

POLICIES = {
    "fixed_budget_10kb": FixedBudgetController(budget_bytes=10_000),
    "fixed_budget_25kb": FixedBudgetController(budget_bytes=25_000),
    "fixed_budget_50kb": FixedBudgetController(budget_bytes=50_000),
    "fixed_threshold_0.75": FixedThresholdController(confidence_threshold=0.75),
    "fixed_threshold_0.90": FixedThresholdController(confidence_threshold=0.90),
    "heuristic_adaptive": AdaptiveController(),
}


def run_all():
    results = []
    for policy_name, policy in POLICIES.items():
        for task in ALL_SAMPLE_TASKS:
            episode_result = SimulatedEpisode(task).run(policy)
            results.append({
                "policy": policy_name,
                "task_id": episode_result["task_id"],
                "difficulty": episode_result["difficulty"],
                "final_bytes_used": episode_result["final_bytes_used"],
                "final_resources_seen": episode_result["final_resources_seen"],
                "stopped_at_step": episode_result["stopped_at_step"],
                "ground_truth_correct_at_step": task.ground_truth_correct_at_step,
                "reached_sufficient_evidence": (
                    task.ground_truth_correct_at_step is not None
                    and episode_result["stopped_at_step"] >= task.ground_truth_correct_at_step
                ),
            })

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = run_all()
    print(f"{len(results)} rows written to {RESULTS_PATH}\n")
    header = f"{'policy':<22}{'task':<10}{'diff':<8}{'bytes':<8}{'steps':<7}{'sufficient':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['policy']:<22}{r['task_id']:<10}{r['difficulty']:<8}"
            f"{r['final_bytes_used']:<8}{r['stopped_at_step']:<7}{str(r['reached_sufficient_evidence']):<10}"
        )