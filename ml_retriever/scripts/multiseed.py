"""Phase G rigor: multi-seed training variance for the learned stopping policies.

Trains our SGD bandit, LinUCB, and linear Thompson sampling across several seeds
and reports mean +/- std of val success and bytes, so no headline number rests on
one lucky run. To isolate POLICY-training randomness (exploration order and, for
Thompson, posterior sampling), a single feature normalizer is fit once and shared
across all seeds; only the policy seed varies. Uses the fast EvidenceAnswerGenerator
(no generative model), like sweep_lambda.py, so five seeds run in minutes. Train
and val only; the frozen test split is never touched here.

Usage: python scripts/multiseed.py --seeds 0 1 2 3 4 --epochs 8 --lam 0.5
"""
import argparse
import json
import random
import statistics as stats
from pathlib import Path

from ml_retriever.answer import EvidenceAnswerGenerator
from ml_retriever.bandit import BanditPolicy, FeatureNormalizer, LinTSPolicy, LinUCBPolicy
from ml_retriever.evidence import CachedScorer, QAScorer
from ml_retriever.retriever import EntityAwareRetriever
from ml_retriever.rollout import build_candidates, decide_heuristic, run_episode
from ml_retriever.types import Passage, Requirement

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LAM_DEFAULT = 0.5


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


def evaluate(val, cands, scorer, ag, decide, lam):
    s = b = 0.0
    for t in val:
        res = run_episode(t, cands[t["id"]], scorer, ag, decide, lam=lam, threshold=0.3)
        s += res.success; b += res.total_bytes
    n = len(val)
    return s / n, b / n


def train_policy(kind, seed, train, cands, normalizer, scorer, ag, lam, epochs):
    if kind == "bandit":
        policy = BanditPolicy(epsilon=0.2, seed=seed)
    elif kind == "linucb":
        policy = LinUCBPolicy(alpha=1.0, seed=seed)
    else:
        policy = LinTSPolicy(v=0.25, seed=seed)
    rng = random.Random(seed)
    for epoch in range(epochs):
        if kind == "bandit":
            policy.epsilon = 0.2 * (1 - epoch / max(epochs - 1, 1))
        order = list(train); rng.shuffle(order)
        for t in order:
            res = run_episode(t, cands[t["id"]], scorer, ag,
                              lambda c, s: policy.select_action(normalizer.transform(c), True),
                              lam=lam, threshold=0.3, reward_mode="per_step")
            for c, a, r in res.trajectory:
                policy.update(normalizer.transform(c), a, r)
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lam", type=float, default=LAM_DEFAULT)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    corpus = load_corpus()
    retriever = EntityAwareRetriever(corpus)
    scorer = CachedScorer(QAScorer())
    ag = EvidenceAnswerGenerator()
    train = judgeable("train"); val = judgeable("val")
    cands = {t["id"]: build_candidates(reqs_of(t), retriever, k=args.k) for t in train + val}

    # one shared normalizer (fit once, seed 0) so only policy training varies
    rng = random.Random(0)
    ctxs = []
    for t in train:
        res = run_episode(t, cands[t["id"]], scorer, ag, lambda c, s: rng.randint(0, 1),
                          lam=args.lam, threshold=0.3, reward_mode="per_step")
        ctxs.extend(c for c, *_ in res.trajectory)
    normalizer = FeatureNormalizer().fit(ctxs)

    print(f"Multi-seed variance over seeds {args.seeds} (val, evidence answerer, lam={args.lam})\n")
    results = {}
    for kind in ("bandit", "linucb", "lints"):
        succ, byts = [], []
        for seed in args.seeds:
            policy = train_policy(kind, seed, train, cands, normalizer, scorer, ag, args.lam, args.epochs)
            s, b = evaluate(val, cands, scorer, ag,
                            lambda c, st: policy.select_action(normalizer.transform(c), False), args.lam)
            succ.append(s); byts.append(b)
        results[kind] = (succ, byts)

    # deterministic reference
    hs, hb = evaluate(val, cands, scorer, ag, decide_heuristic, args.lam)

    def ms(xs):
        return stats.mean(xs), (stats.stdev(xs) if len(xs) > 1 else 0.0)

    print(f"{'policy':<12}{'success mean+/-std':>22}{'bytes mean+/-std':>22}")
    for kind in ("bandit", "linucb", "lints"):
        succ, byts = results[kind]
        sm, sd = ms(succ); bm, bd = ms(byts)
        print(f"{kind:<12}{sm:>10.3f} +/- {sd:<7.3f}{bm:>10.0f} +/- {bd:<7.0f}")
    print(f"{'heuristic':<12}{hs:>10.3f} (det.)   {hb:>10.0f} (det.)")


if __name__ == "__main__":
    main()
