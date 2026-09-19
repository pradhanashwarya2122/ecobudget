"""Phase 5: contextual bandit policy -- learn when to RETRIEVE more vs STOP.

This is the project's central research piece, so it is a genuine online
contextual bandit, not a relabeled heuristic:
  - a per-action linear model (`SGDClassifier(loss="log_loss")`) predicts
    whether taking that action in a given context leads to an above-average
    episode reward;
  - `select_action` is epsilon-greedy over those predictions (with random
    cold-start while a model is unfit);
  - `update` turns the episode's scalar reward into an online training signal.

Action space (v1, per plan.md review-2): STOP / RETRIEVE_TOP1 only. Get this
loop solid before per-requirement actions (v2). The reward's `success` comes
from judge.py and is never exposed to the policy -- only the scalar reward
reaches `update`.

Feature math and the policy are pure/sklearn and fully unit-tested; nothing
here loads an embedding model or an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

STOP = 0
RETRIEVE = 1
ACTIONS = ("STOP", "RETRIEVE")

# Fixed feature order (mirrors backend/usefulness.py's FEATURE_NAMES convention
# so a learned linear model stays interpretable).
FEATURE_NAMES = [
    "frac_satisfied",
    "frac_remaining",
    "coverage",
    "n_requirements",
    "n_unanswered",
    "passages_added",
    "bytes_used_frac",       # cumulative bytes / max_bytes  (remaining-budget signal)
    "step",
    "next_cand_top_sim",     # best candidate similarity for the top unanswered requirement
    "next_cand_mean_sim",
    "has_unretrieved_candidate",
]
N_FEATURES = len(FEATURE_NAMES)


def featurize(features: dict) -> np.ndarray:
    """Pack a dict of named scalars into the fixed-order context vector.
    Missing keys default to 0.0 so callers can omit features they can't
    compute at a given step."""
    return np.array([float(features.get(name, 0.0)) for name in FEATURE_NAMES], dtype=float)


def compute_reward(success: bool, total_bytes: int, max_bytes: int, lam: float) -> float:
    """Episode reward: success minus a byte penalty (plan.md Phase 5).

    success - lam * total_payload_bytes / max_bytes. Credited to every step of
    the trajectory (Monte-Carlo credit assignment over a short episode), so the
    policy learns to stop once evidence is sufficient rather than paying for
    retrieval that no longer raises success."""
    penalty = lam * (total_bytes / max(max_bytes, 1))
    return (1.0 if success else 0.0) - penalty


class FeatureNormalizer:
    """z-score normalization with statistics fit on the training contexts
    (plan.md: raw byte-count features would otherwise dominate by variance)."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, contexts: list[np.ndarray]) -> "FeatureNormalizer":
        X = np.vstack(contexts)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0  # guard constant features
        return self

    def transform(self, context: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return context
        return (context - self.mean) / self.std


@dataclass
class _ActionModel:
    clf: object
    fitted: bool = False


class BanditPolicy:
    """Epsilon-greedy linear contextual bandit over {STOP, RETRIEVE}."""

    def __init__(self, epsilon: float = 0.1, seed: int = 0, learning_rate: str = "optimal"):
        from sklearn.linear_model import SGDClassifier

        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)
        self._models = [
            _ActionModel(SGDClassifier(loss="log_loss", learning_rate=learning_rate, eta0=0.01))
            for _ in ACTIONS
        ]
        self._reward_sum = 0.0
        self._reward_n = 0

    @property
    def reward_baseline(self) -> float:
        return self._reward_sum / self._reward_n if self._reward_n else 0.0

    def _value(self, action: int, context: np.ndarray) -> float:
        """Predicted P(this action yields above-baseline reward) in [0,1].
        Unfit models return 0.5 (neutral/optimistic-enough to get explored)."""
        m = self._models[action]
        if not m.fitted:
            return 0.5
        return float(m.clf.predict_proba(context.reshape(1, -1))[0, 1])

    def action_values(self, context: np.ndarray) -> list[float]:
        return [self._value(a, context) for a in range(len(ACTIONS))]

    def select_action(self, context: np.ndarray, explore: bool = True) -> int:
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(len(ACTIONS)))
        values = self.action_values(context)
        best = float(np.max(values))
        # random tie-break among argmax
        candidates = [a for a, v in enumerate(values) if v == best]
        return int(self._rng.choice(candidates))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self._reward_sum += reward
        self._reward_n += 1
        label = 1 if reward >= self.reward_baseline else 0
        weight = abs(reward - self.reward_baseline) + 1e-3  # emphasize strong signals
        m = self._models[action]
        m.clf.partial_fit(
            context.reshape(1, -1), [label], classes=[0, 1], sample_weight=[weight]
        )
        m.fitted = True

    def save(self, path) -> None:
        import joblib

        joblib.dump(
            {
                "epsilon": self.epsilon,
                "models": [(m.clf, m.fitted) for m in self._models],
                "reward_sum": self._reward_sum,
                "reward_n": self._reward_n,
                "feature_names": FEATURE_NAMES,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "BanditPolicy":
        import joblib

        blob = joblib.load(path)
        policy = cls(epsilon=blob["epsilon"])
        policy._models = [_ActionModel(clf, fitted) for clf, fitted in blob["models"]]
        policy._reward_sum = blob["reward_sum"]
        policy._reward_n = blob["reward_n"]
        return policy


# --- Phase E: standard contextual-bandit baselines --------------------------
# LinUCB and linear Thompson sampling are the textbook contextual bandits a
# reviewer expects us to compare against. Both are disjoint (one linear model per
# action), consume the SAME context vector and the SAME per-step reward as our
# SGD bandit, and expose the SAME select_action / update interface, so Phase 7
# runs them through the identical rollout and metrics. Pure numpy, unit-tested.

class LinUCBPolicy:
    """Disjoint LinUCB (Li et al., 2010). Per action a it keeps ridge stats
    A_a = I + sum x x^T and b_a = sum r x, forms theta_a = A_a^-1 b_a, and picks
    argmax over theta_a . x + alpha * sqrt(x^T A_a^-1 x). The sqrt term is an
    upper confidence bonus that shrinks as an action is tried in a region, so
    exploration is built in (no epsilon). At eval (explore=False) alpha is set to
    0, i.e. greedy on the predicted reward."""

    def __init__(self, alpha: float = 1.0, seed: int = 0, n_features: int = N_FEATURES):
        self.alpha = alpha
        self.d = n_features
        self._rng = np.random.default_rng(seed)
        self.A = [np.eye(self.d) for _ in ACTIONS]
        self.b = [np.zeros(self.d) for _ in ACTIONS]

    def _ucb(self, action: int, x: np.ndarray, alpha: float) -> float:
        A_inv = np.linalg.inv(self.A[action])
        theta = A_inv @ self.b[action]
        mean = float(theta @ x)
        bonus = alpha * float(np.sqrt(max(x @ A_inv @ x, 0.0)))
        return mean + bonus

    def action_values(self, context: np.ndarray) -> list[float]:
        return [self._ucb(a, context, self.alpha) for a in range(len(ACTIONS))]

    def select_action(self, context: np.ndarray, explore: bool = True) -> int:
        alpha = self.alpha if explore else 0.0
        values = [self._ucb(a, context, alpha) for a in range(len(ACTIONS))]
        best = max(values)
        candidates = [a for a, v in enumerate(values) if v == best]
        return int(self._rng.choice(candidates))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = context.reshape(-1)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def save(self, path) -> None:
        import joblib
        joblib.dump({"alpha": self.alpha, "d": self.d,
                     "A": self.A, "b": self.b, "feature_names": FEATURE_NAMES}, path)

    @classmethod
    def load(cls, path) -> "LinUCBPolicy":
        import joblib
        blob = joblib.load(path)
        p = cls(alpha=blob["alpha"], n_features=blob["d"])
        p.A, p.b = blob["A"], blob["b"]
        return p


class LinTSPolicy:
    """Linear Thompson sampling (Agrawal & Goyal, 2013), disjoint per action.
    Keeps B_a = I + sum x x^T and f_a = sum r x; the posterior over the reward
    weights is N(B_a^-1 f_a, v^2 B_a^-1). At each step it SAMPLES a weight vector
    per action and picks the argmax -- randomized exploration. At eval
    (explore=False) it uses the posterior mean (no sampling) for a deterministic
    greedy decision."""

    def __init__(self, v: float = 0.25, seed: int = 0, n_features: int = N_FEATURES):
        self.v = v
        self.d = n_features
        self._rng = np.random.default_rng(seed)
        self.B = [np.eye(self.d) for _ in ACTIONS]
        self.f = [np.zeros(self.d) for _ in ACTIONS]

    def _score(self, action: int, x: np.ndarray, sample: bool) -> float:
        B_inv = np.linalg.inv(self.B[action])
        mean = B_inv @ self.f[action]
        if sample:
            theta = self._rng.multivariate_normal(mean, (self.v ** 2) * B_inv)
        else:
            theta = mean
        return float(theta @ x)

    def action_values(self, context: np.ndarray) -> list[float]:
        return [self._score(a, context, sample=False) for a in range(len(ACTIONS))]

    def select_action(self, context: np.ndarray, explore: bool = True) -> int:
        values = [self._score(a, context, sample=explore) for a in range(len(ACTIONS))]
        best = max(values)
        candidates = [a for a, v in enumerate(values) if v == best]
        return int(self._rng.choice(candidates))

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        x = context.reshape(-1)
        self.B[action] += np.outer(x, x)
        self.f[action] += reward * x

    def save(self, path) -> None:
        import joblib
        joblib.dump({"v": self.v, "d": self.d,
                     "B": self.B, "f": self.f, "feature_names": FEATURE_NAMES}, path)

    @classmethod
    def load(cls, path) -> "LinTSPolicy":
        import joblib
        blob = joblib.load(path)
        p = cls(v=blob["v"], n_features=blob["d"])
        p.B, p.f = blob["B"], blob["f"]
        return p
