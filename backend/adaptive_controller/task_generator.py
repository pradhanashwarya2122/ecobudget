"""
Generates a larger pool of synthetic tasks by parameterized variation
around the three hand-built difficulty archetypes.

Key design constraint: ground_truth_correct_at_step is NEVER 0.
A task solvable at step 0 means "always stop immediately" is optimal,
which prevents the bandit from learning any retrieval policy at all.
Every task requires at least one retrieval step before stopping is correct.
"""
import random
from typing import List

from .simulator import SimulatedTask, RetrievalStep

ARCHETYPES = {
    # Easy: answer available after 1 retrieval step. Starts uncertain,
    # crosses the coverage threshold at step 1.
    "easy": {
        "n_steps": 3,
        "base_bytes": 1250,
        "confidence_start": 0.45,
        "confidence_end": 0.92,
    },
    # Medium: answer available after 2-3 retrieval steps.
    "medium": {
        "n_steps": 5,
        "base_bytes": 1950,
        "confidence_start": 0.25,
        "confidence_end": 0.88,
    },
    # Hard: answer available after 4-5 retrieval steps.
    "hard": {
        "n_steps": 7,
        "base_bytes": 2350,
        "confidence_start": 0.10,
        "confidence_end": 0.82,
    },
}

# Coverage must reach this threshold before stopping is correct.
# Kept at 0.75 to match original intent; tasks are now shaped so
# step 0 never reaches it.
COVERAGE_THRESHOLD = 0.75


def _generate_one(difficulty: str, task_id: str, rng: random.Random) -> SimulatedTask:
    spec = ARCHETYPES[difficulty]
    n = spec["n_steps"]
    steps = []

    for i in range(n):
        progress = i / max(n - 1, 1)
        confidence = (
            spec["confidence_start"]
            + progress * (spec["confidence_end"] - spec["confidence_start"])
            + rng.uniform(-0.04, 0.04)
        )
        confidence = max(0.0, min(1.0, confidence))

        # Coverage lags confidence slightly and is noisier.
        coverage = max(0.0, min(1.0, confidence - 0.08 + rng.uniform(-0.03, 0.03)))
        relevance = max(0.0, min(1.0, 0.4 + progress * 0.4 + rng.uniform(-0.04, 0.04)))
        bytes_ = max(100, int(spec["base_bytes"] + rng.uniform(-300, 300)))

        steps.append(RetrievalStep(
            bytes=bytes_,
            relevance=relevance,
            answer_confidence=confidence,
            evidence_coverage=coverage,
        ))

    # Ground truth: first step where coverage crosses threshold.
    gt = next(
        (i for i, s in enumerate(steps) if s.evidence_coverage >= COVERAGE_THRESHOLD),
        n - 1,
    )

    # Enforce the key constraint: stopping must require at least one retrieval.
    # If the threshold was crossed at step 0 (due to noise), push it to step 1.
    gt = max(1, gt)

    return SimulatedTask(
        task_id=task_id,
        question=f"Synthetic {difficulty} question {task_id}",
        difficulty=difficulty,
        ground_truth_correct_at_step=gt,
        steps=steps,
    )


def generate_task_pool(n_per_difficulty: int = 20, seed: int = 42) -> List[SimulatedTask]:
    rng = random.Random(seed)
    tasks = []
    for difficulty in ARCHETYPES:
        for i in range(n_per_difficulty):
            tasks.append(_generate_one(difficulty, f"{difficulty}-gen-{i}", rng))
    return tasks


def split_tasks(tasks: List[SimulatedTask], train_frac=0.6, val_frac=0.2, seed=42):
    """Stratified split: preserves difficulty proportions in each split."""
    rng = random.Random(seed)
    by_difficulty = {}
    for t in tasks:
        by_difficulty.setdefault(t.difficulty, []).append(t)

    train, val, test = [], [], []
    for difficulty, group in by_difficulty.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])
    return train, val, test