from dataclasses import dataclass
from typing import List, Optional

from .context import Context
from .actions import Action


@dataclass
class RetrievalStep:
    """One unit of evidence the simulator can reveal."""
    bytes: int
    relevance: float
    answer_confidence: float
    evidence_coverage: float


@dataclass
class SimulatedTask:
    """A synthetic question plus the sequence of evidence retrieving it would reveal."""
    task_id: str
    question: str
    steps: List[RetrievalStep]
    difficulty: str = "unknown"
    # Eval-only. Never read by the controller or by current_context().
    ground_truth_correct_at_step: Optional[int] = None


class SimulatedEpisode:
    """
    Plays a SimulatedTask against a controller, one retrieval step at a
    time, tracking cumulative state so Context is built the same way it
    eventually will be from real retrieval (Phase 8).
    """

    def __init__(self, task: SimulatedTask):
        self.task = task
        self.step_index = 0
        self.bytes_used = 0
        self.resources_seen = 0
        self.history: List[RetrievalStep] = []
        self.done = False

    def current_context(self) -> Context:
        if self.history:
            latest = self.history[-1]
            relevance = latest.relevance
            answer_confidence = latest.answer_confidence
            evidence_coverage = latest.evidence_coverage
        else:
            relevance = answer_confidence = evidence_coverage = 0.0

        # Marginal gain: how much did confidence improve on the last step?
        # Approximates "is evidence still helping?" without access to task
        # internals. Computable in production from real retrieval responses.
        # Falls to zero (or negative) when further retrieval stops helping.
        if len(self.history) >= 2:
            confidence_gain = (
                self.history[-1].answer_confidence
                - self.history[-2].answer_confidence
            )
        elif len(self.history) == 1:
            confidence_gain = self.history[-1].answer_confidence
        else:
            confidence_gain = 0.0

        return Context(
            question=self.task.question,
            data_used=self.bytes_used,
            resources_seen=self.resources_seen,
            relevance_score=relevance,
            answer_confidence=answer_confidence,
            evidence_coverage=evidence_coverage,
            steps_taken=self.step_index,
            estimated_remaining_information=confidence_gain,
        )

    def step(self) -> Optional[RetrievalStep]:
        """Reveal the next piece of evidence. Returns None once exhausted."""
        if self.step_index >= len(self.task.steps):
            self.done = True
            return None

        piece = self.task.steps[self.step_index]
        self.history.append(piece)
        self.bytes_used += piece.bytes
        self.resources_seen += 1
        self.step_index += 1

        if self.step_index >= len(self.task.steps):
            self.done = True
        return piece

    def run(self, controller, max_steps: Optional[int] = None) -> dict:
        """
        Full episode loop: ask the controller to decide, act on it,
        return a trace. ground_truth_correct_at_step is deliberately
        left out of everything the controller touches — it's only for
        the evaluator to read afterward, from outside this method.
        """
        trace = []
        limit = max_steps or len(self.task.steps)

        while not self.done and self.step_index < limit:
            context = self.current_context()
            decision = controller.decide(context)
            trace.append({
                "step": self.step_index,
                "action": decision.action,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "bytes_used_so_far": self.bytes_used,
            })

            if decision.action == Action.STOP:
                break

            if self.step() is None:
                break

        return {
            "task_id": self.task.task_id,
            "difficulty": self.task.difficulty,
            "trace": trace,
            "final_bytes_used": self.bytes_used,
            "final_resources_seen": self.resources_seen,
            "stopped_at_step": self.step_index,
        }