from dataclasses import dataclass

from .actions import Action
from .context import Context


@dataclass
class Decision:
    action: Action
    confidence: float
    reason: str


class AdaptiveController:
    """
    Baseline controller. Deliberately simple and transparent — this is
    Phase 1's placeholder, not the research contribution. It exists so
    the surrounding architecture (context in, decision out) is stable
    before Phase 6 swaps this logic for a contextual bandit.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        coverage_threshold: float = 0.7,
        max_steps: int = 6,
    ):
        self.confidence_threshold = confidence_threshold
        self.coverage_threshold = coverage_threshold
        self.max_steps = max_steps

    def decide(self, context: Context) -> Decision:
        if context.steps_taken >= self.max_steps:
            return Decision(
                action=Action.STOP,
                confidence=0.5,
                reason=f"hit max_steps ({self.max_steps}) without full confidence",
            )

        if (
            context.answer_confidence >= self.confidence_threshold
            and context.evidence_coverage >= self.coverage_threshold
        ):
            return Decision(
                action=Action.STOP,
                confidence=min(context.answer_confidence, context.evidence_coverage),
                reason="answer confidence and evidence coverage both above threshold",
            )

        return Decision(
            action=Action.RETRIEVE,
            confidence=1 - context.answer_confidence,
            reason="answer confidence or evidence coverage still below threshold",
        )