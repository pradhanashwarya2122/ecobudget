"""Phase 5 reward fix (Option 2): realign task gold to STRUCTURED required_facts.

The v1 judge was designed for structured answers, but the dataset drifted:
comparison/yes_no/list tasks carried free-text verdict strings in
`expected_answer` (e.g. "Samsung S24 (up to 120Hz vs iPhone's 60Hz)") that the
concat-evidence answer generator can never reproduce -- so even an oracle that
retrieves everything scored ~0 and the bandit's reward was degenerate.

This realigns the DATA, not the judge: for comparison/yes_no/multi_part/list
seed tasks it sets `ground_truth.required_facts` to the answer values actually
present in each requirement's corpus passage (extracted with the same local QA
model used for evidence), so success becomes "did retrieval gather the evidence
that contains each grounded value" -- retrieval-dependent and achievable.
`expected_answer` is kept as a display-only field, no longer the success
criterion. single_fact already works via its value string and is left alone.
narrative stays free-text (needs the Phase 6 generative answerer) and is
documented as a known residual.

Run once, then regenerate tasks.json. Idempotent (recomputes each run).
"""
from __future__ import annotations

import json
from pathlib import Path

from ml_retriever.evidence import QAScorer
from ml_retriever.types import Passage, Requirement

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

REALIGN_TYPES = {"comparison", "yes_no", "multi_part", "list"}


def main() -> None:
    corpus_rows = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    by_key: dict[tuple[str, str], Passage] = {}
    for r in corpus_rows:
        m = r.get("metadata", {})
        if "entity" in m and "attribute" in m:
            key = (m["entity"].strip().lower(), m["attribute"].strip().lower())
            by_key.setdefault(key, Passage(
                passage_id=r["passage_id"], text=r["text"], byte_size=r["byte_size"],
                metadata=m,
            ))

    seeds = json.loads((DATA / "tasks_seed.json").read_text(encoding="utf-8"))
    scorer = QAScorer()

    realigned = skipped = 0
    for task in seeds:
        if task["answer_type"] not in REALIGN_TYPES:
            continue
        facts: list[str] = []
        ok = True
        for r in task["decomposed_requirements"]:
            req = Requirement(entity=r["entity"], attribute=r["attribute"])
            passage = by_key.get(req.key())
            if passage is None:
                ok = False
                break
            span = scorer.extract_answer(req, passage).strip()
            if not span:
                ok = False
                break
            if span not in facts:
                facts.append(span)
        if not ok or not facts:
            skipped += 1
            print(f"  WARN {task['id']} ({task['answer_type']}): could not derive facts; left as-is")
            continue
        task["ground_truth"] = {"required_facts": facts, "match_threshold": 1.0}
        realigned += 1

    (DATA / "tasks_seed.json").write_text(
        json.dumps(seeds, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Realigned {realigned} seed tasks to required_facts; {skipped} skipped.")
    print("Now regenerate: generate_synthetic_tasks -> make_splits -> build_decomposer_training_data")


if __name__ == "__main__":
    main()
