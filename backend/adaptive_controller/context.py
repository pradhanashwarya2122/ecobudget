from dataclasses import dataclass
from typing import Optional


@dataclass
class Context:
    """
    Everything the controller is allowed to see when deciding whether
    to retrieve more evidence or stop.

    IMPORTANT: no field here may be derived from ground truth. Ground
    truth is only used later, in offline evaluation (Phase 7).
    """
    question: str
    data_used: int                                    # bytes retrieved so far
    resources_seen: int                                # passages/pages retrieved so far
    relevance_score: float                             # 0-1
    answer_confidence: float                           # 0-1
    evidence_coverage: float                            # 0-1, estimated
    steps_taken: int
    estimated_remaining_information: Optional[float] = None