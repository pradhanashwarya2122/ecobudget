"""Expands data/tasks_seed.json into data/tasks.json via deterministic,
template-based paraphrasing -- NOT an LLM call. This is the "generate
synthetic variations of manually-written seed tasks" approach from the
plan's review addendum, used to reach the 150-200 task target without a
proportional increase in manual annotation.

Every synthetic task keeps the exact same `decomposed_requirements`,
`expected_answer` / `ground_truth`, `topic`, `difficulty`, and
`answer_type` as its seed -- only the question wording changes, so ground
truth can never silently drift from the source task. Each synthetic task
also carries `synthetic: true` and `seed_task_id` so later phases (and the
train/val/test split) can group or filter on it.

Run: python scripts/generate_synthetic_tasks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ATTRIBUTE_PHRASES = {
    "price": "price",
    "battery": "battery capacity",
    "display_refresh_rate": "display refresh rate",
    "weight": "weight",
    "chip": "chip",
    "camera": "main camera resolution",
    "display": "display",
    "best_time_to_visit": "recommended travel season",  # avoids "...to visit of Jaipur"
    "daily_budget": "daily travel budget",
    "top_attraction": "top attraction",
    "typical_trip_length": "typical trip length",
    "price_per_night": "price per night",
    "rating": "guest rating",
    "amenities": "amenities",
    "room_count": "room count",
    "battery_life": "battery life",
    "noise_cancellation": "noise cancellation",
    "codec_support": "audio codec support",
    "height": "height",
    "floors": "floor count",
    "completion_year": "completion year",
    "architect": "architect",
    "elevation": "elevation",
    "location": "location",
    "first_ascent_year": "year of first ascent",
    "population": "population",
    "area": "land area",
    "capital": "capital",
    "official_language": "official languages",
    "cushioning": "cushioning",
    "best_for": "recommended use",
    "range": "driving range",
    "battery_capacity": "battery capacity",
}

SINGLE_FACT_TEMPLATES = [
    "What is the {attr} of {e}?",
    "Can you tell me the {attr} of {e}?",
    "How would you describe the {attr} of {e}?",
    "I'd like to know the {attr} of {e}.",
    "Please tell me {e}'s {attr}.",
    "What's {e}'s {attr}?",
    "Do you know the {attr} of {e}?",
    "Give me the {attr} for {e}.",
    "What {attr} does {e} have?",
    "Looking for the {attr} of {e} -- what is it?",
    "Could you look up the {attr} of {e}?",
    "I need the {attr} of {e} for a comparison.",
    "What does {e} offer in terms of {attr}?",
]

COMPARISON_1ATTR_TEMPLATES = [
    "Compare {e1} and {e2} on {attr}.",
    "Which is better in terms of {attr}: {e1} or {e2}?",
    "How does {e1}'s {attr} compare to {e2}'s?",
    "Between {e1} and {e2}, which has the better {attr}?",
    "{e1} vs {e2}: which wins on {attr}?",
    "I'm deciding between {e1} and {e2} -- which has the better {attr}?",
    "Can you compare {attr} between {e1} and {e2}?",
    "In terms of {attr}, how do {e1} and {e2} differ?",
    "Of {e1} and {e2}, which has better {attr}?",
]

COMPARISON_2ATTR_TEMPLATES = [
    "Compare {e1} and {e2} on {attr1} and {attr2}.",
    "How do {e1} and {e2} differ in {attr1} and {attr2}?",
    "I'm choosing between {e1} and {e2} -- how do they compare on {attr1} and {attr2}?",
    "Which is the better choice, {e1} or {e2}, considering {attr1} and {attr2}?",
    "Break down {e1} vs {e2} by {attr1} and {attr2}.",
    "What are the differences between {e1} and {e2} in {attr1} and {attr2}?",
    "Help me weigh {e1} against {e2} on {attr1} and {attr2}.",
]

YES_NO_TEMPLATES = [
    "Is {e1}'s {attr} higher than {e2}'s?",
    "Does {e1} have a greater {attr} than {e2}?",
    "Is {e1} ahead of {e2} on {attr}?",
]

LIST_TEMPLATES = [
    "List the {attr} offered by {e}.",
    "What {attr} are available at {e}?",
    "Name the {attr} at {e}.",
]

MULTI_PART_TEMPLATES = [
    "What are {e}'s {attr1} and {attr2}?",
    "Tell me the {attr1} and {attr2} for {e}.",
    "Provide {e}'s {attr1} along with its {attr2}.",
]

PROCEDURE_TEMPLATES = [
    "What are the steps to change a flat car tire?",
    "Walk me through replacing a flat tire on a car.",
    "How should I replace a flat vehicle tire safely?",
]

NARRATIVE_TEMPLATES = [
    "Give me a brief summary of {e}.",
    "What is the premise of {e}?",
    "Briefly describe the story of {e}.",
]


def _phrase(attribute: str) -> str:
    return ATTRIBUTE_PHRASES.get(attribute, attribute.replace("_", " "))


def _reset_rotation() -> None:
    # No rotation state to reset. Kept as a hook so build() stays explicit
    # about template selection being deterministic and seed-order-independent.
    pass


def _rotated(templates: list[str], n: int) -> list[str]:
    """Take the first `n` templates of a family -- the SAME `n` for every
    seed of that family.

    This is deliberately not a per-seed or rotating selection. Because every
    seed of a given comparison family (1-attr vs 2-attr) draws from the same
    first-`n` templates, any phrasing that appears in val also appears in
    train, as long as train contains at least one seed of that family --
    which the split guarantees. A rotating/hashed offset breaks that: it let
    a phrasing land only in val's seeds and never train's, recreating the
    exact train/val coverage gap this fix targets. Balancing phrasing beyond
    this (so all templates, not just the first `n`, appear in both splits)
    is a split-composition problem, not a generator one, and is left alone."""
    return templates[:n]


def _requirement_shape(reqs: list[dict]):
    entities = list(dict.fromkeys(r["entity"] for r in reqs))
    attributes = list(dict.fromkeys(r["attribute"] for r in reqs))
    return entities, attributes


def generate_variants(task: dict) -> list[str]:
    """Generates up to N_VARIANTS_PER_FAMILY paraphrases per seed task.

    Three variants per seed; with ~90 seeds after the Phase 1 scale-up this
    yields ~360 tasks (see Phase-1-lever-plan.md and the raised cap in
    tests/test_data.py).
    """
    N_VARIANTS_PER_FAMILY = 3

    reqs = task["decomposed_requirements"]
    entities, attributes = _requirement_shape(reqs)

    if task["answer_type"] == "yes_no":
        return [
            t.format(e1=entities[0], e2=entities[1], attr=_phrase(attributes[0]))
            for t in _rotated(YES_NO_TEMPLATES, N_VARIANTS_PER_FAMILY)
        ]

    if task["answer_type"] == "list":
        return [
            t.format(e=entities[0], attr=_phrase(attributes[0]))
            for t in _rotated(LIST_TEMPLATES, N_VARIANTS_PER_FAMILY)
        ]

    if task["answer_type"] == "multi_part":
        return [
            t.format(
                e=entities[0],
                attr1=_phrase(attributes[0]),
                attr2=_phrase(attributes[1]),
            )
            for t in _rotated(MULTI_PART_TEMPLATES, N_VARIANTS_PER_FAMILY)
        ]

    if task["answer_type"] == "procedure":
        return _rotated(PROCEDURE_TEMPLATES, N_VARIANTS_PER_FAMILY)

    if task["answer_type"] == "narrative":
        return [t.format(e=entities[0]) for t in _rotated(NARRATIVE_TEMPLATES, N_VARIANTS_PER_FAMILY)]

    if len(reqs) == 1:
        e = entities[0]
        attr = _phrase(attributes[0])
        return [t.format(e=e, attr=attr) for t in _rotated(SINGLE_FACT_TEMPLATES, N_VARIANTS_PER_FAMILY)]

    if len(entities) == 2 and len(attributes) == 1:
        e1, e2 = entities
        attr = _phrase(attributes[0])
        return [t.format(e1=e1, e2=e2, attr=attr) for t in _rotated(COMPARISON_1ATTR_TEMPLATES, N_VARIANTS_PER_FAMILY)]

    if len(entities) == 2 and len(attributes) == 2:
        e1, e2 = entities
        attr1, attr2 = (_phrase(a) for a in attributes)
        return [
            t.format(e1=e1, e2=e2, attr1=attr1, attr2=attr2)
            for t in _rotated(COMPARISON_2ATTR_TEMPLATES, N_VARIANTS_PER_FAMILY)
        ]

    # Fallback for shapes not covered by a template family: no synthetic
    # variants rather than a wrong-shaped guess.
    return []


def build(seed_tasks: list[dict]) -> list[dict]:
    _reset_rotation()
    all_tasks: list[dict] = []
    for task in seed_tasks:
        base = dict(task)
        base["synthetic"] = False
        all_tasks.append(base)

        variants = generate_variants(task)
        for i, question in enumerate(variants, start=1):
            variant = dict(task)
            variant["id"] = f"{task['id']}-s{i:02d}"
            variant["question"] = question
            variant["synthetic"] = True
            variant["seed_task_id"] = task["id"]
            all_tasks.append(variant)
    return all_tasks


def main() -> None:
    seed_path = ROOT / "data" / "tasks_seed.json"
    seed_tasks = json.loads(seed_path.read_text(encoding="utf-8"))

    all_tasks = build(seed_tasks)

    out_path = ROOT / "data" / "tasks.json"
    out_path.write_text(json.dumps(all_tasks, indent=2, ensure_ascii=False), encoding="utf-8")

    n_synthetic = sum(1 for t in all_tasks if t["synthetic"])
    print(f"Wrote {len(all_tasks)} tasks to {out_path} "
          f"({len(seed_tasks)} seed + {n_synthetic} synthetic)")


if __name__ == "__main__":
    main()
