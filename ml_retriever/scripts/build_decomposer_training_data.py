"""Builds (question -> serialized requirements) training pairs for Phase 2.

Reads data/tasks.json + the frozen data/splits.json.FROZEN, and writes:
    data/decomposer_train.jsonl   (from the train split)
    data/decomposer_val.jsonl     (from the val split, for dev-time eval)

The test split is intentionally NOT written here. Per plan.md, it stays
untouched until Phase 7's final evaluation -- writing it out now would
make it too easy to accidentally eval-then-tune against it during Phase 2
development.

Each output line: {"question": "...", "target": "entity|attribute ## ..."}

Run after any change to data/tasks.json or data/splits.json.FROZEN:
    python scripts/build_decomposer_training_data.py
"""

import json
from pathlib import Path

from ml_retriever.decomposer import serialize_requirements
from ml_retriever.types import Requirement

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def build():
    tasks = {t["id"]: t for t in _load_json(DATA_DIR / "tasks.json")}
    # splits.json.FROZEN is a human-readable note (see make_splits.py);
    # the actual split assignment lives in splits.json.
    splits = _load_json(DATA_DIR / "splits.json")

    for split_name in ("train", "val"):
        rows = []
        for task_id in splits[split_name]:
            task = tasks[task_id]
            reqs = [
                Requirement(entity=r["entity"], attribute=r["attribute"])
                for r in task["decomposed_requirements"]
            ]
            rows.append(
                {
                    "task_id": task_id,
                    "question": task["question"],
                    "target": serialize_requirements(reqs),
                }
            )
        out_path = DATA_DIR / f"decomposer_{split_name}.jsonl"
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(rows)} {split_name} pairs to {out_path}")


if __name__ == "__main__":
    build()
