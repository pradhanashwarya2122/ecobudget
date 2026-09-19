"""Phase G ablation: per-step vs terminal reward for the stopping bandit.

Trains a fresh SGD bandit under each reward mode on the train split and evaluates
on val with the fast EvidenceAnswerGenerator (same harness as multiseed.py and
sweep_lambda.py), so it does not touch the saved production policy. lambda=0.5,
entity-aware candidates. Averages over a few seeds so the comparison is not a
single lucky run.

Usage: python scripts/ablate_reward.py --seeds 0 1 2 --epochs 8 --lam 0.5
"""
import argparse
import json
import random
import statistics as st
from pathlib import Path

from ml_retriever.answer import EvidenceAnswerGenerator
from ml_retriever.bandit import BanditPolicy, FeatureNormalizer
from ml_retriever.evidence import CachedScorer, QAScorer
from ml_retriever.retriever import EntityAwareRetriever
from ml_retriever.rollout import build_candidates, run_episode
from ml_retriever.types import Passage, Requirement

DATA = Path(__file__).resolve().parent.parent / "data"


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    corpus = load_corpus()
    retr = EntityAwareRetriever(corpus)
    scorer = CachedScorer(QAScorer())
    ag = EvidenceAnswerGenerator()
    train = judgeable("train"); val = judgeable("val")
    cands = {t["id"]: build_candidates(reqs_of(t), retr, k=args.k) for t in train + val}

    # one shared normalizer (reward-independent), fit on random rollouts
    rng0 = random.Random(0); ctxs = []
    for t in train:
        res = run_episode(t, cands[t["id"]], scorer, ag, lambda c, s: rng0.randint(0, 1),
                          lam=args.lam, threshold=0.3, reward_mode="per_step")
        ctxs.extend(c for c, *_ in res.trajectory)
    normalizer = FeatureNormalizer().fit(ctxs)

    def train_eval(mode, seed):
        policy = BanditPolicy(epsilon=0.2, seed=seed); rng = random.Random(seed)
        for ep in range(args.epochs):
            policy.epsilon = 0.2 * (1 - ep / max(args.epochs - 1, 1))
            order = list(train); rng.shuffle(order)
            for t in order:
                res = run_episode(t, cands[t["id"]], scorer, ag,
                                  lambda c, s: policy.select_action(normalizer.transform(c), True),
                                  lam=args.lam, threshold=0.3, reward_mode=mode)
                for c, a, r in res.trajectory:
                    policy.update(normalizer.transform(c), a, r)
        s = b = 0.0
        for t in val:
            res = run_episode(t, cands[t["id"]], scorer, ag,
                              lambda c, st_: policy.select_action(normalizer.transform(c), False),
                              lam=args.lam, threshold=0.3)
            s += res.success; b += res.total_bytes
        n = len(val)
        return s / n, b / n

    print(f"Reward-mode ablation (val, evidence answerer, lam={args.lam}, seeds {args.seeds})\n")
    print(f"{'reward_mode':<12}{'success mean+/-std':>22}{'bytes mean+/-std':>22}")
    for mode in ("per_step", "terminal"):
        succ, byts = [], []
        for seed in args.seeds:
            sc, by = train_eval(mode, seed); succ.append(sc); byts.append(by)
        def ms(xs): return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)
        sm, sd = ms(succ); bm, bd = ms(byts)
        print(f"{mode:<12}{sm:>10.3f} +/- {sd:<7.3f}{bm:>10.0f} +/- {bd:<7.0f}")


if __name__ == "__main__":
    main()
