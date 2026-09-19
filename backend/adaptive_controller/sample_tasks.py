"""
Hand-built synthetic tasks for offline development, standing in for
Teammate 1's retrieval output until that's ready. Each is shaped so an
easy task should let a reasonable controller stop early, and a hard
task should force it to keep retrieving.

ground_truth_correct_at_step is eval-only: the step at which
evidence_coverage first reaches 0.75, standing in for "evidence was
actually sufficient." It is never read by AdaptiveController.decide()
or SimulatedEpisode.current_context() — only by reward computation
(Phase 3) and evaluation (Phase 7).
"""
from .simulator import SimulatedTask, RetrievalStep


def easy_task() -> SimulatedTask:
    return SimulatedTask(
        task_id="easy-1",
        question="What is the capital of France?",
        difficulty="easy",
        ground_truth_correct_at_step=1,
        steps=[
            RetrievalStep(bytes=1200, relevance=0.90, answer_confidence=0.92, evidence_coverage=0.90),
            RetrievalStep(bytes=1300, relevance=0.85, answer_confidence=0.95, evidence_coverage=0.93),
        ],
    )


def medium_task() -> SimulatedTask:
    return SimulatedTask(
        task_id="medium-1",
        question="What were the main causes of the 2008 financial crisis?",
        difficulty="medium",
        ground_truth_correct_at_step=4,
        steps=[
            RetrievalStep(bytes=2000, relevance=0.60, answer_confidence=0.40, evidence_coverage=0.30),
            RetrievalStep(bytes=2200, relevance=0.65, answer_confidence=0.60, evidence_coverage=0.55),
            RetrievalStep(bytes=1900, relevance=0.70, answer_confidence=0.78, evidence_coverage=0.72),
            RetrievalStep(bytes=1800, relevance=0.72, answer_confidence=0.88, evidence_coverage=0.85),
        ],
    )


def hard_task() -> SimulatedTask:
    return SimulatedTask(
        task_id="hard-1",
        question="Compare the economic policies of three central banks during a rate-hike cycle.",
        difficulty="hard",
        ground_truth_correct_at_step=6,
        steps=[
            RetrievalStep(bytes=2500, relevance=0.50, answer_confidence=0.20, evidence_coverage=0.15),
            RetrievalStep(bytes=2600, relevance=0.55, answer_confidence=0.35, evidence_coverage=0.28),
            RetrievalStep(bytes=2400, relevance=0.58, answer_confidence=0.45, evidence_coverage=0.40),
            RetrievalStep(bytes=2300, relevance=0.60, answer_confidence=0.55, evidence_coverage=0.50),
            RetrievalStep(bytes=2200, relevance=0.65, answer_confidence=0.68, evidence_coverage=0.62),
            RetrievalStep(bytes=2100, relevance=0.70, answer_confidence=0.80, evidence_coverage=0.79),
        ],
    )


ALL_SAMPLE_TASKS = [easy_task(), medium_task(), hard_task()]
