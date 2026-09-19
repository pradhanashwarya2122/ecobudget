"""Phase 5 validation: compare the trained bandit against the baseline
conditions (plan.md): normal retrieval, fixed budgets, and the simple adaptive
heuristic (the EcoBudgetController analog). Target: bandit ~= normal task
success with meaningfully less data than the fixed budgets.

Same environment/judge for every condition, so the comparison is valid.
Evaluated on the val split's judgeable tasks (never trained on).

Usage:
    python scripts/eval_bandit.py
"""

import json
from pathlib import Path

import joblib

from ml_retriever.answer import EvidenceAnswerGenerator
from ml_retriever.bandit import BanditPolicy
from ml_retriever.evidence import CachedScorer, QAScorer
from ml_retriever.retriever import EntityAwareRetriever
from ml_retriever.rollout import (
    build_candidates,
    decide_heuristic,
    decide_normal,
    make_fixed_budget_decider,
    run_episode,
)
from ml_retriever.types import Passage, Requirement

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
LAM = 4.0  # chosen operating point from scripts/sweep_lambda.py (per-step reward)
THRESHOLD = 0.3
K = 5


def load_corpus():
    rows = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    return [Passage(passage_id=r["passage_id"], text=r["text"], byte_size=r["byte_size"],
                    source_url=r.get("source_url", ""), embedding=r.get("embedding"),
                    metadata=r.get("metadata", {})) for r in rows]


def judgeable(split):
    tasks = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())}
    splits = json.loads((DATA / "splits.json").read_text())
    return [tasks[i] for i in splits[split] if "expected_answer" in tasks[i] or "ground_truth" in tasks[i]]


def reqs_of(t):
    return [Requirement(entity=r["entity"], attribute=r["attribute"]) for r in t["decomposed_requirements"]]


def main():
    corpus = load_corpus()
    retriever = EntityAwareRetriever(corpus)  # Phase D/G: entity-gate + attribute-rank
    scorer = CachedScorer(QAScorer())
    answer_gen = EvidenceAnswerGenerator()

    val = judgeable("val")
    cands = {t["id"]: build_candidates(reqs_of(t), retriever, k=K) for t in val}

    policy = BanditPolicy.load(MODELS / "bandit_policy.joblib")
    normalizer = joblib.load(MODELS / "bandit_normalizer.joblib")

    def bandit_decider(ctx, state):
        return policy.select_action(normalizer.transform(ctx), explore=False)

    conditions = {
        "normal": decide_normal,
        "fixed_budget@2": make_fixed_budget_decider(2),
        "fixed_budget@4": make_fixed_budget_decider(4),
        "heuristic": decide_heuristic,
        "bandit": bandit_decider,
    }

    print(f"{'condition':<18}{'success':>9}{'avg_bytes':>11}{'avg_retr':>10}")
    rows = {}
    for name, decide in conditions.items():
        succ = byts = retr = 0.0
        for t in val:
            res = run_episode(t, cands[t["id"]], scorer, answer_gen, decide,
                              lam=LAM, threshold=THRESHOLD)
            succ += res.success
            byts += res.total_bytes
            retr += res.n_retrieves
        n = len(val)
        rows[name] = {"success": succ / n, "avg_bytes": byts / n, "avg_retrieves": retr / n}
        print(f"{name:<18}{succ/n:>9.3f}{byts/n:>11.0f}{retr/n:>10.2f}")

    out = DATA / "bandit_eval_results.json"
    out.write_text(json.dumps({"n_val": len(val), "conditions": rows}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
