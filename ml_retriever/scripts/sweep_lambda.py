"""Phase 5: sweep the reward's byte-penalty weight lambda and plot the
success-vs-bytes trade-off (plan.md review-2 deliverable).

At low lambda the policy has no incentive to stop early (it over-retrieves);
too high and it stops before gathering enough. This trains a fresh bandit per
lambda (reward valid after build_structured_gold) and evaluates each on val
against the fixed baselines, so we can pick the lambda that matches the
project's goal: comparable success with meaningfully fewer bytes than the
fixed budgets. Baselines are lambda-independent (their success/bytes depend
only on the decision rule), so they're computed once as reference.

Usage: python scripts/sweep_lambda.py --epochs 8 --lams 0.5 1 2 4 8
"""
import argparse
import json
import random
from pathlib import Path

from ml_retriever.answer import EvidenceAnswerGenerator
from ml_retriever.bandit import BanditPolicy, FeatureNormalizer
from ml_retriever.evidence import CachedScorer, QAScorer
from ml_retriever.retriever import EntityAwareRetriever
from ml_retriever.rollout import (
    build_candidates, decide_heuristic, decide_normal,
    make_fixed_budget_decider, run_episode,
)
from ml_retriever.types import Passage, Requirement

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
THRESHOLD = 0.3
K = 5


def load_corpus():
    rows = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    return [Passage(passage_id=r["passage_id"], text=r["text"], byte_size=r["byte_size"],
                    embedding=r.get("embedding"), metadata=r.get("metadata", {})) for r in rows]


def judgeable(split, tasks, splits):
    return [tasks[i] for i in splits[split] if "expected_answer" in tasks[i] or "ground_truth" in tasks[i]]


def reqs_of(t):
    return [Requirement(entity=r["entity"], attribute=r["attribute"]) for r in t["decomposed_requirements"]]


def eval_condition(tasks, cands, scorer, ag, decide, lam):
    s = b = 0.0
    for t in tasks:
        res = run_episode(t, cands[t["id"]], scorer, ag, decide, lam=lam, threshold=THRESHOLD)
        s += res.success; b += res.total_bytes
    n = len(tasks)
    return s / n, b / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.5, 1, 2, 4, 8])
    ap.add_argument("--reward_mode", choices=["terminal", "per_step"], default="per_step")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    corpus = load_corpus()
    retriever = EntityAwareRetriever(corpus)  # Phase G: match production retriever
    scorer = CachedScorer(QAScorer())
    ag = EvidenceAnswerGenerator()
    tasks = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())}
    splits = json.loads((DATA / "splits.json").read_text())
    train = judgeable("train", tasks, splits)
    val = judgeable("val", tasks, splits)
    cands = {t["id"]: build_candidates(reqs_of(t), retriever, k=K)
             for t in train + val}

    # baselines (lambda-independent success/bytes)
    print("baselines (val):")
    for name, dec in [("normal", decide_normal), ("fixed@2", make_fixed_budget_decider(2)),
                      ("fixed@4", make_fixed_budget_decider(4)), ("heuristic", decide_heuristic)]:
        su, by = eval_condition(val, cands, scorer, ag, dec, lam=1.0)
        print(f"  {name:<12} success={su:.3f} bytes={by:.0f}")

    rng = random.Random(args.seed)
    # normalizer is reward-independent; fit once on random train rollouts
    ctxs = []
    for t in train:
        res = run_episode(t, cands[t["id"]], scorer, ag, lambda c, s: rng.randint(0, 1),
                          lam=1.0, threshold=THRESHOLD, reward_mode=args.reward_mode)
        ctxs.extend(c for c, *_ in res.trajectory)
    normalizer = FeatureNormalizer().fit(ctxs)

    print("\nlambda sweep (val):")
    print(f"  {'lambda':>7}{'success':>10}{'avg_bytes':>11}{'avg_retr':>10}")
    for lam in args.lams:
        policy = BanditPolicy(epsilon=0.2, seed=args.seed)
        for epoch in range(args.epochs):
            policy.epsilon = 0.2 * (1 - epoch / max(args.epochs - 1, 1))
            order = list(train); rng.shuffle(order)
            for t in order:
                res = run_episode(t, cands[t["id"]], scorer, ag,
                                  lambda c, s: policy.select_action(normalizer.transform(c), True),
                                  lam=lam, threshold=THRESHOLD, reward_mode=args.reward_mode)
                for c, a, r in res.trajectory:
                    policy.update(normalizer.transform(c), a, r)
        # eval (greedy)
        su = by = rt = 0.0
        for t in val:
            res = run_episode(t, cands[t["id"]], scorer, ag,
                              lambda c, s: policy.select_action(normalizer.transform(c), False),
                              lam=lam, threshold=THRESHOLD)
            su += res.success; by += res.total_bytes; rt += res.n_retrieves
        n = len(val)
        print(f"  {lam:>7.1f}{su/n:>10.3f}{by/n:>11.0f}{rt/n:>10.2f}")


if __name__ == "__main__":
    main()
