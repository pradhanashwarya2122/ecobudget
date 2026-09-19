"""Phase 1 validation: every task's decomposed_requirements must have at
least one corpus passage with a matching (entity, attribute). This is a
scripted check, not manual review -- see also the 10-random-task manual
review step in plan.md, which this script doesn't replace.

Run: python scripts/validate_corpus_task_coverage.py
Exits non-zero if any task has an uncovered requirement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_retriever.types import Requirement  # noqa: E402


def load_corpus_keys(path: Path) -> set[tuple[str, str]]:
    keys = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            entity = row["metadata"]["entity"]
            attribute = row["metadata"]["attribute"]
            keys.add(Requirement(entity=entity, attribute=attribute).key())
    return keys


def load_tasks(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def check(corpus_path: Path, tasks_path: Path) -> list[str]:
    corpus_keys = load_corpus_keys(corpus_path)
    tasks = load_tasks(tasks_path)

    violations: list[str] = []
    for task in tasks:
        for req in task["decomposed_requirements"]:
            key = Requirement(entity=req["entity"], attribute=req["attribute"]).key()
            if key not in corpus_keys:
                violations.append(
                    f"Task {task['id']}: no corpus passage for requirement "
                    f"(entity={req['entity']!r}, attribute={req['attribute']!r})"
                )
    return violations


def main() -> int:
    corpus_path = ROOT / "data" / "corpus.jsonl"
    tasks_path = ROOT / "data" / "tasks.json"

    violations = check(corpus_path, tasks_path)
    n_tasks = len(load_tasks(tasks_path))
    n_passages = sum(1 for _ in corpus_path.open(encoding="utf-8"))

    if violations:
        print(f"COVERAGE CHECK FAILED ({len(violations)} violation(s)):")
        for v in violations:
            print(f"  - {v}")
        return 1

    print(
        f"Coverage check passed: all {n_tasks} tasks have >=1 corpus "
        f"passage per requirement ({n_passages} passages in corpus)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
