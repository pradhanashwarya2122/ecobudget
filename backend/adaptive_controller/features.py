"""
Turns a Context into a fixed-length numeric feature vector for the bandit.

Features are normalized to comparable scales and deliberately exclude
anything eval-only (ground truth).
"""

from typing import List

from .context import Context

FEATURE_NAMES = [
    "data_used_kb",
    "resources_seen",
    "relevance_score",
    "answer_confidence",
    "evidence_coverage",
    "steps_taken",
    "marginal_confidence_gain",
    "relevance_delta",
    "confidence_delta",
    "retrieval_need",
]


class ContextVectorBuilder:
    """
    Stateful across a single episode: tracks previous relevance/confidence
    to compute deltas. Call reset() at the start of each new episode.
    """

    def __init__(self):
        self._prev_relevance = None
        self._prev_confidence = None

    def reset(self):
        self._prev_relevance = None
        self._prev_confidence = None

    def build(self, context: Context) -> List[float]:
        relevance_delta = (
            0.0
            if self._prev_relevance is None
            else context.relevance_score - self._prev_relevance
        )

        confidence_delta = (
            0.0
            if self._prev_confidence is None
            else context.answer_confidence - self._prev_confidence
        )

        self._prev_relevance = context.relevance_score
        self._prev_confidence = context.answer_confidence

        retrieval_need = (
            (1.0 - context.answer_confidence)
            + (1.0 - context.evidence_coverage)
        ) / 2.0

        return [
            min(context.data_used / 25000.0, 1.0),
            min(float(context.resources_seen) / 10.0, 1.0),
            context.relevance_score,
            context.answer_confidence,
            context.evidence_coverage,
            min(float(context.steps_taken) / 10.0, 1.0),
            context.estimated_remaining_information or 0.0,
            relevance_delta,
            confidence_delta,
            retrieval_need,
        ]
