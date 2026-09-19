"""Phase 3 validation: recall@k of per-requirement retrieval vs the
whole-question baseline.

The concrete, checkable Phase 3 deliverable (plan.md): show that retrieving
per Requirement beats ranking the corpus once against the whole question,
especially on multi-requirement questions where a single query can only chase
one information need.

Gold labels come for free from the corpus: a Requirement (entity, attribute)
is satisfied by exactly the corpus passages whose metadata carries that same
(entity, attribute). No manual relevance labeling, and -- because retrieval is
training-free (no fit on the task splits) -- this does not touch the Phase
5/7 test-split freeze. We evaluate over the hand-written SEED tasks only (not
synthetic paraphrases, which would just duplicate the same requirements).

Two systems, scored at the (task, requirement) level with recall@k = 1 if any
gold passage for that requirement lands in the system's top-k:
  - per-requirement: embed each requirement separately (RequirementRetriever).
  - whole-question baseline: embed the whole question once; every requirement
    of that task is checked against that single top-k list.

Needs sentence-transformers + the pre-embedded corpus (run embed_corpus.py).

Usage:
    python scripts/eval_retriever.py            # recall@5, plain cosine
    python scripts/eval_retriever.py --k 5 --value_per_byte
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ml_retriever.retriever import (
    EntityAwareRetriever,
    RequirementRetriever,
    rank_passages,
)
from ml_retriever.types import Passage, Requirement

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_corpus() -> list[Passage]:
    passages = []
    with (DATA_DIR / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            passages.append(Passage(
                passage_id=r["passage_id"], text=r["text"], byte_size=r["byte_size"],
                source_url=r.get("source_url", ""), embedding=r.get("embedding"),
                metadata=r.get("metadata", {}),
            ))
    return passages


def gold_map(passages: list[Passage]) -> dict[tuple[str, str], set[str]]:
    gold = defaultdict(set)
    for p in passages:
        m = p.metadata
        if "entity" in m and "attribute" in m:
            gold[(m["entity"].strip().lower(), m["attribute"].strip().lower())].add(p.passage_id)
    return gold


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    return 1.0 if gold_ids & set(retrieved_ids[:k]) else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--value_per_byte", action="store_true")
    parser.add_argument("--cross_encoder", action="store_true",
                        help="Re-rank the bi-encoder pool with a cross-encoder (extra model download).")
    args = parser.parse_args()

    passages = load_corpus()
    if any(p.embedding is None for p in passages):
        raise SystemExit("Some passages have no embedding -- run scripts/embed_corpus.py first.")
    gold = gold_map(passages)

    tasks = json.loads((DATA_DIR / "tasks.json").read_text(encoding="utf-8"))
    seed_tasks = [t for t in tasks if not t.get("synthetic")]

    retriever = RequirementRetriever(
        passages, value_per_byte=args.value_per_byte, use_cross_encoder=args.cross_encoder
    )
    # Phase D: entity-aware two-stage retriever (gate to the requirement's
    # entity, then rank by the attribute phrase). Same protocol, drop-in.
    entity_aware = EntityAwareRetriever(
        passages, value_per_byte=args.value_per_byte, rank_on="attribute"
    )

    # accumulators: (all units) and (multi-requirement tasks only). Three
    # systems: whole-question baseline, per-requirement bi-encoder, and the
    # Phase D entity-aware retriever.
    def acc():
        return {"per_req": 0.0, "baseline": 0.0, "entity_aware": 0.0, "n": 0}
    overall, multi = acc(), acc()
    # de-dupe requirements so a frequent (entity, attribute) is not counted
    # once per task -- one honest measurement per distinct requirement.
    seen_reqs: set = set()

    for t in seed_tasks:
        reqs = [Requirement(entity=r["entity"], attribute=r["attribute"])
                for r in t["decomposed_requirements"]]
        is_multi = len(reqs) >= 2

        # whole-question baseline: one ranking for the whole task
        q_emb = retriever._encode(t["question"])
        baseline_ids = [s.passage.passage_id for s in
                        rank_passages(q_emb, passages, k=args.k, value_per_byte=args.value_per_byte)]

        for req in reqs:
            g = gold.get(req.key(), set())
            if not g:
                continue  # no corpus passage for this requirement (shouldn't happen)
            if req.key() in seen_reqs:
                continue
            seen_reqs.add(req.key())
            per_req_ids = [s.passage.passage_id for s in retriever.retrieve(req, k=args.k)]
            ea_ids = [s.passage.passage_id for s in entity_aware.retrieve(req, k=args.k)]
            r_per = recall_at_k(per_req_ids, g, args.k)
            r_base = recall_at_k(baseline_ids, g, args.k)
            r_ea = recall_at_k(ea_ids, g, args.k)
            for bucket in (overall, *( (multi,) if is_multi else () )):
                bucket["per_req"] += r_per
                bucket["baseline"] += r_base
                bucket["entity_aware"] += r_ea
                bucket["n"] += 1

    def summarize(b, label):
        n = b["n"] or 1
        pr, bl, ea = b["per_req"] / n, b["baseline"] / n, b["entity_aware"] / n
        return {
            "bucket": label, "n_requirements": b["n"],
            f"recall@{args.k}_whole_question_baseline": round(bl, 4),
            f"recall@{args.k}_per_requirement": round(pr, 4),
            f"recall@{args.k}_entity_aware": round(ea, 4),
            "gain_per_req_vs_baseline": round(pr - bl, 4),
            "gain_entity_aware_vs_per_req": round(ea - pr, 4),
        }

    result = {
        "k": args.k,
        "value_per_byte": args.value_per_byte,
        "overall": summarize(overall, "all_requirements"),
        "multi_requirement_tasks": summarize(multi, "multi_requirement_only"),
    }
    print(json.dumps(result, indent=2))

    out = DATA_DIR / "retriever_eval_results.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Appended to {out}")


if __name__ == "__main__":
    main()
