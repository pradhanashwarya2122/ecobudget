import os
import numpy as np
import joblib

_MODEL_PATH = "models/usefulness.joblib"

FEATURE_NAMES = [
    "similarity", "passage_size", "value_per_byte",
    "frac_satisfied", "frac_remaining", "coverage",
    "num_previous_loads", "cumulative_bytes",
]


def build_features(similarity, passage_size, tracker, num_loads, cumulative_bytes):
    vpb = similarity / max(passage_size, 1)
    return np.array([
        similarity,
        passage_size,
        vpb,
        tracker.frac_satisfied(),
        tracker.frac_remaining(),
        tracker.coverage(),
        num_loads,
        cumulative_bytes,
    ], dtype=float)


class UsefulnessModel:
    def __init__(self):
        self.model = None
        if os.path.exists(_MODEL_PATH):
            self.model = joblib.load(_MODEL_PATH)

    def predict(self, feature_matrix):
        feature_matrix = np.atleast_2d(feature_matrix)
        if self.model is None:
            sim = feature_matrix[:, 0]
            frac_remaining = feature_matrix[:, 4]
            return np.clip(sim * (0.5 + 0.5 * frac_remaining), 0, 1)
        return self.model.predict_proba(feature_matrix)[:, 1]

    def train(self, X, y):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.9, eval_metric="logloss",
        )
        self.model.fit(X, y)
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, _MODEL_PATH)
        return self