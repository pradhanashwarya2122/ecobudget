"""EcoBudget ML + Retrieval package (Teammate 1).

Local-models-only: no LLM API client should ever be imported here.
See scripts/check_no_llm_api.py.
"""

from .types import Requirement, Passage, AnswerResult

__all__ = ["Requirement", "Passage", "AnswerResult"]
