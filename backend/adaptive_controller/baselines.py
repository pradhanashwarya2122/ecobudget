from dataclasses import dataclass

from .actions import Action
from .controller import Decision


@dataclass
class FixedBudgetController:
    """Retrieves until a fixed byte budget is exhausted, then stops."""
    budget_bytes: int

    def decide(self, context) -> Decision:
        if context.data_used >= self.budget_bytes:
            return Decision(
                action=Action.STOP,
                confidence=1.0,
                reason=f"byte budget {self.budget_bytes} reached",
            )
        return Decision(
            action=Action.RETRIEVE,
            confidence=1.0,
            reason=f"under byte budget {self.budget_bytes}",
        )


@dataclass
class FixedThresholdController:
    """Stops once answer_confidence crosses a fixed threshold, regardless of coverage."""
    confidence_threshold: float

    def decide(self, context) -> Decision:
        if context.answer_confidence >= self.confidence_threshold:
            return Decision(
                action=Action.STOP,
                confidence=context.answer_confidence,
                reason=f"answer_confidence >= {self.confidence_threshold}",
            )
        return Decision(
            action=Action.RETRIEVE,
            confidence=1.0 - context.answer_confidence,
            reason=f"answer_confidence below {self.confidence_threshold}",
        )