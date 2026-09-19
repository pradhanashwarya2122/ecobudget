"""
LinUCB contextual bandit over two arms: STOP and RETRIEVE.

Reference: Li, Chu, Langford, Schapire (2010), "A Contextual-Bandit
Approach to Personalized News Article Recommendation."

This is intentionally the simplest contextual bandit that's still a
real research object — no deep RL, per Phase 6 spec.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

from .actions import Action
from .controller import Decision

ARMS = [Action.STOP, Action.RETRIEVE]


@dataclass
class LinUCBBandit:
    n_features: int
    alpha: float = 1.0  # exploration coefficient; higher = more exploration
    seed: int = 0       # controls tie-breaking RNG; set for reproducibility

    def __post_init__(self):
        self.A = {arm: np.identity(self.n_features) for arm in ARMS}
        self.b = {arm: np.zeros(self.n_features) for arm in ARMS}
        self._rng = np.random.default_rng(self.seed)

    def _score(self, arm: Action, x: np.ndarray) -> float:
        A_inv = np.linalg.inv(self.A[arm])
        theta = A_inv @ self.b[arm]
        mean = float(x @ theta)
        exploration = self.alpha * float(np.sqrt(x @ A_inv @ x))
        # Tiny random jitter breaks deterministic ties when arms are equally
        # scored (e.g. at step 0 before any updates). Jitter is 1e-6 scale —
        # negligible once real signal accumulates, but sufficient to prevent
        # insertion-order bias from freezing the policy from epoch 1.
        jitter = float(self._rng.uniform(-1e-6, 1e-6))
        return mean + exploration + jitter

    def choose_action(self, feature_vector: List[float]) -> Decision:
        x = np.array(feature_vector, dtype=float)
        scores = {arm: self._score(arm, x) for arm in ARMS}
        best_arm = max(scores, key=scores.get)

        total = sum(abs(s) for s in scores.values()) or 1.0
        confidence = abs(scores[best_arm]) / total

        return Decision(
            action=best_arm,
            confidence=min(confidence, 1.0),
            reason=(
                f"LinUCB scores: STOP={scores[Action.STOP]:.3f}, "
                f"RETRIEVE={scores[Action.RETRIEVE]:.3f}"
            ),
        )

    def update(self, feature_vector: List[float], action: Action, reward: float):
        x = np.array(feature_vector, dtype=float)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def decide(self, feature_vector: List[float]) -> Decision:
        return self.choose_action(feature_vector)