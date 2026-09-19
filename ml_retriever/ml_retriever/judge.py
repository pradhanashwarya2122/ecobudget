"""Answer-type-aware task judges for Phase 5 rewards and Phase 7 metrics.

The dispatcher is deliberately data-driven: replacing either free-text judge
with a local NLI judge later changes ``JUDGES``, not the call sites.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

Judge = Callable[[str, dict[str, Any]], tuple[bool, float]]

# Formatting-only normalizer (applied symmetrically to gold and prediction).
# Covers ONLY surface variations the answerer should be allowed to produce:
# case/whitespace, currency/thousands separators, number-unit spacing, trailing
# .0 decimals, and leading articles. It does NOT do synonym or semantic
# matching, and the match threshold stays 1.0. See docs/eval_notes.md.
_ARTICLES = {"the", "a", "an"}


def _normalize(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[$,]", "", s)                 # "$1,099" -> "1099"
    s = re.sub(r"(\d)\.0+\b", r"\1", s)        # "799.00" -> "799", "10.0" -> "10" (before unit merge)
    s = re.sub(r"(\d)\s+([a-z])", r"\1\2", s)  # "120 hz" -> "120hz"
    s = re.sub(r"\s+", " ", s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)

STRUCTURED_ANSWER_TYPES = {
    "single_fact",
    "yes_no",
    "list",
    "comparison",
    "multi_part",
}
PROCEDURE_STEP_F1_THRESHOLD = 0.4
NARRATIVE_F1_THRESHOLD = 0.4


def _ground_truth(task: dict[str, Any]) -> str | dict[str, Any]:
    if "ground_truth" in task:
        return task["ground_truth"]
    return task["expected_answer"]


def _string_set_match(answer: str, expected: str) -> bool:
    """Substring/token-subset match after symmetric formatting normalization."""
    if not answer:
        return False
    na = _normalize(answer)
    ne = _normalize(expected)
    if not ne:
        return False
    if ne in na or na in ne:
        return True
    answer_tokens = set(na.split())
    expected_tokens = set(ne.split())
    return bool(expected_tokens) and expected_tokens.issubset(answer_tokens)


def token_f1(answer: str, gold: str) -> float:
    """Lowercased whitespace-token-set F1 for v1 free-text evaluation."""
    answer_tokens = set(answer.lower().split())
    gold_tokens = set(gold.lower().split())
    if not answer_tokens or not gold_tokens:
        return 0.0
    overlap = len(answer_tokens & gold_tokens)
    if not overlap:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def judge_structured(answer: str, task: dict[str, Any]) -> tuple[bool, float]:
    """Rule-based judge for scalar and required-fact structured answers."""
    gold = _ground_truth(task)
    if isinstance(gold, str):
        success = _string_set_match(answer, gold)
        return success, 1.0 if success else 0.0

    facts = gold["required_facts"]
    if not facts:
        return False, 0.0
    matched = sum(_string_set_match(answer, fact) for fact in facts)
    frac = matched / len(facts)  # fact-level hit rate (continuous), for fact_f1
    return frac >= gold.get("match_threshold", 1.0), frac


def judge_procedure(answer: str, task: dict[str, Any]) -> tuple[bool, float]:
    """Judge a procedure by the fraction of required steps it covers."""
    gold = _ground_truth(task)
    if not isinstance(gold, dict):
        raise ValueError("procedure tasks require required_facts ground truth")
    steps = gold["required_facts"]
    if not steps:
        return False, 0.0
    covered = sum(
        token_f1(answer, step) >= PROCEDURE_STEP_F1_THRESHOLD
        for step in steps
    )
    score = covered / len(steps)
    return score >= gold.get("match_threshold", 1.0), score


def judge_narrative(answer: str, task: dict[str, Any]) -> tuple[bool, float]:
    """Judge a narrative answer against its plain-string gold summary."""
    gold = _ground_truth(task)
    if not isinstance(gold, str):
        raise ValueError("narrative tasks require plain-string ground truth")
    score = token_f1(answer, gold)
    return score >= NARRATIVE_F1_THRESHOLD, score


JUDGES: dict[str, Judge] = {
    **{answer_type: judge_structured for answer_type in STRUCTURED_ANSWER_TYPES},
    "procedure": judge_procedure,
    "narrative": judge_narrative,
}


def judge_task(answer: str, task: dict[str, Any]) -> tuple[bool, float]:
    """Dispatch a generated answer to the task's versioned v1 judge."""
    try:
        judge = JUDGES[task["answer_type"]]
    except KeyError as exc:
        raise ValueError(f"Unsupported answer_type: {task.get('answer_type')!r}") from exc
    return judge(answer, task)
