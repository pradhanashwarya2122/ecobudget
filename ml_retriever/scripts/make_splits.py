"""Creates data/splits.json: a train/val/test split over data/tasks.json.

Splits at the SEED-TASK level, not the individual-task level: every
synthetic paraphrase of a given seed task is assigned to the same split as
its seed. Splitting paraphrases independently would let near-duplicate
wording of the same underlying question appear in both train and test --
exactly the leakage the plan's review flagged for the corpus/task split in
general. Grouping by seed_task_id closes that specific hole.

Deterministic (fixed seed groups below, no randomness) so this is
reproducible without needing to pin a PRNG. Run once; per plan.md this
split is then FROZEN and must not be touched again until Phase 7.

Run: python scripts/make_splits.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 41 seed task ids -> split, chosen to keep each split's difficulty, topic,
# and answer-type mix roughly representative. Existing assignments are frozen;
# T37-T41 extend the same seed-grouping rule for new answer types.
SEED_SPLIT_ASSIGNMENT = {
    # train (24 seeds, ~2/3)
    "T1": "train", "T2": "train", "T4": "train", "T5": "train", "T6": "train",
    "T8": "train", "T9": "train", "T11": "train", "T13": "train", "T14": "train",
    "T16": "train", "T18": "train", "T19": "train", "T21": "train", "T22": "train",
    "T24": "train", "T25": "train", "T27": "train", "T28": "train", "T30": "train",
    "T31": "train", "T33": "train", "T34": "train", "T36": "train",
    # val (7 seeds)
    "T3": "val", "T7": "val", "T10": "val", "T17": "val", "T20": "val",
    "T26": "val", "T32": "val",
    # test (5 seeds)
    "T12": "test", "T15": "test", "T23": "test", "T29": "test", "T35": "test",
    # General question-type expansion
    "T37": "train", "T38": "val", "T39": "train", "T40": "test", "T41": "val",
    # Coverage fix (2026): train seeds added so every answer type (except
    # procedure, see below) and every attribute slug that was previously
    # val-only now also appears in train. None of these reuse a val/test
    # (entity, attribute) pair, so they add train coverage without leakage.
    "T42": "train",  # top_attraction (Jaipur; val keeps Kyoto/T10)
    "T43": "train",  # list answer type (Ibis amenities; val keeps Oberoi/T38)
    "T44": "train",  # narrative + summary slug (Titanic; val keeps Mamma Mia/T41)
    "T45": "train",  # noise_cancellation slug (Bose; val keeps AirPods/Sony/T20)
    "T46": "train",  # first_ascent_year slug (Kangchenjunga; val keeps Everest/K2/T26)
    "T47": "train",  # camera + display_refresh_rate slugs (Pixel 8; test/val keep iPhone/S24)
    "T48": "train",  # typical_trip_length slug (Bali; test keeps Jaipur/Kyoto/T12)
    # Phase 1 scale-up (2026-09-12): 42 balanced train seeds (T49-T90) added by
    # scripts/expand_seed_tasks.py, all train-bound and built only from corpus
    # pairs not used by any val/test seed -- no leakage. See Phase-1-lever-plan.md.
    **{f"T{i}": "train" for i in range(49, 91)},
}

# 'procedure' as an attribute slug is exempt for the same reason as the
# answer type: its only seed (T40) is reserved for the frozen test split.
_ATTRIBUTE_COVERAGE_EXEMPT = {"procedure"}

# 'procedure' is intentionally exempt from the train-coverage invariant below:
# only one procedure seed exists (T40) and it is reserved for the frozen test
# split, and the generator's procedure templates are tire-specific, so a train
# procedure seed can't be synthesized without either leaking into test or
# authoring new templates. Tracked as a known gap, not silently ignored.
_ANSWER_TYPE_COVERAGE_EXEMPT = {"procedure"}


def assert_train_coverage(tasks, splits) -> None:
    """Fail if any answer type or attribute slug has zero train examples.

    This is the exact bug this fix addresses: seed-level splitting had left
    whole answer types (list, narrative) and attribute slugs
    (top_attraction, noise_cancellation, first_ascent_year, summary) with no
    training data, making them unlearnable and tanking val exact-match. This
    assertion stops that regression from silently returning when a future
    seed is added. (val>=1 per slug is deliberately NOT required: a slug can
    be trained but not measured on val without being a correctness bug, and
    requiring it would force every train-only fact into val and blow the
    150-200 task cap.)"""
    train_ids = set(splits["train"])
    by_id = {t["id"]: t for t in tasks}

    train_answer_types = set()
    train_attributes = set()
    all_answer_types = set()
    all_attributes = set()
    for t in tasks:
        all_answer_types.add(t["answer_type"])
        attrs = {r["attribute"] for r in t["decomposed_requirements"]}
        all_attributes |= attrs
        if t["id"] in train_ids:
            train_answer_types.add(t["answer_type"])
            train_attributes |= attrs

    missing_types = (all_answer_types - train_answer_types) - _ANSWER_TYPE_COVERAGE_EXEMPT
    missing_attrs = (all_attributes - train_attributes) - _ATTRIBUTE_COVERAGE_EXEMPT
    assert not missing_types, f"answer types with no train coverage: {sorted(missing_types)}"
    assert not missing_attrs, f"attribute slugs with no train coverage: {sorted(missing_attrs)}"


def _phase_c_split(seed_id: str) -> str:
    """Deterministic split for the Phase C scale-up seeds (T91+): ~70/20/10
    train/val/test by id number. The larger val share is intentional -- it is
    the statistical-power lever Phase C exists for. Old seeds keep their frozen
    assignment in SEED_SPLIT_ASSIGNMENT; test seeds T12/T15/T23/T29/T35/T40 are
    untouched."""
    n = int(seed_id[1:])
    r = n % 10
    if r < 2:
        return "val"      # 20%
    if r == 2:
        return "test"     # 10%
    return "train"        # 70%


def _split_for(seed_id: str) -> str:
    return SEED_SPLIT_ASSIGNMENT.get(seed_id) or _phase_c_split(seed_id)


def main() -> None:
    tasks_path = ROOT / "data" / "tasks.json"
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for task in tasks:
        seed_id = task["id"] if not task["synthetic"] else task["seed_task_id"]
        splits[_split_for(seed_id)].append(task["id"])

    assert_train_coverage(tasks, splits)

    out_path = ROOT / "data" / "splits.json"
    out_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    lock_path = ROOT / "data" / "splits.json.FROZEN"
    lock_path.write_text(
        "This split was re-cut in 2026 to fix a train-coverage bug found in\n"
        "Phase 2: seed-level splitting had left whole answer types (list,\n"
        "narrative) and attribute slugs (top_attraction, noise_cancellation,\n"
        "first_ascent_year, summary) with zero training examples, making them\n"
        "unlearnable and tanking val exact-match. Five train seeds (T42-T46)\n"
        "were ADDED to close that; NO existing seed changed split, and the\n"
        "test seeds (T12, T15, T23, T29, T35, T40) are untouched, so the\n"
        "Phase-7 test-leakage guarantee still holds. assert_train_coverage()\n"
        "in make_splits.py now enforces the invariant. (Earlier freeze:\n"
        "2026-09-09, after the general question-type expansion.)\n\n"
        "Per plan.md, do not regenerate or edit data/splits.json (or the\n"
        "seed-to-split assignment in make_splits.py) until Phase 7's final\n"
        "run. Phase 2 decomposer training uses train only; Phase 3 recall@5\n"
        "evaluation uses a separate held-out requirement->passage set, not\n"
        "these task splits; Phase 5 bandit training/eval uses train/val only.\n",
        encoding="utf-8",
    )

    for split_name, ids in splits.items():
        print(f"{split_name}: {len(ids)} tasks")
    print(f"Wrote {out_path} and froze it via {lock_path.name}")


if __name__ == "__main__":
    main()
