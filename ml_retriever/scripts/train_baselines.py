"""Phase E: train the standard contextual-bandit baselines (LinUCB, linear
Thompson sampling) under the SAME conditions as our SGD bandit, so Phase 7's
comparison is fair.

Same feature vector, same per-step reward, same lambda (0.5, the Phase G
selection), same entity-aware candidates, same train split, and the SAME feature
normalizer that train_bandit.py fit (loaded from models/bandit_normalizer.joblib
so every learned policy sees identical normalized features). Train split only;
val/test never trained on.

Usage:
    python scripts/train_baselines.py --epochs 8 --lam 0.5
    python scripts/train_baselines.py --epochs 8 --lam 0.5 --seed 3   # for multi-seed
"""

import argparse
import json
import random
from pathlib import Path

import joblib

from ml_retriever.answer import EvidenceAnswerGenerator
from ml_retriever.bandit import FeatureNormalizer, LinTSPolicy, LinUCBPolicy
from ml_retriever.evidence import CachedScorer, QAScorer
from ml_retriever.retriever import EntityAwareRetriever
from ml_retriever.rollout import build_candidates, run_episode
from ml_retriever.types import Passage, Requirement

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"


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


def train_one(policy, train_tasks, candidates, normalizer, scorer, ag, lam, epochs, seed, name):
    rng = random.Random(seed)
    for epoch in range(epochs):
        order = list(train_tasks); rng.shuffle(order)
        ep_reward = ep_success = ep_bytes = 0.0
        for t in order:
            def decider(ctx, state):
                return policy.select_action(normalizer.transform(ctx), explore=True)
            res = run_episode(t, candidates[t["id"]], scorer, ag, decider,
                              lam=lam, threshold=0.3, reward_mode="per_step")
            for ctx, action, step_reward in res.trajectory:
                policy.update(normalizer.transform(ctx), action, step_reward)
            ep_reward += res.reward; ep_success += res.success; ep_bytes += res.total_bytes
        n = len(order)
        print(f"[{name}] epoch {epoch}: avg_reward={ep_reward/n:.3f} "
              f"success={ep_success/n:.3f} avg_bytes={ep_bytes/n:.0f}")
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.0, help="LinUCB exploration width")
    ap.add_argument("--v", type=float, default=0.25, help="LinTS posterior scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="filename suffix, e.g. _seed3 for multi-seed")
    args = ap.parse_args()

    MODELS.mkdir(exist_ok=True)
    corpus = load_corpus()
    if any(p.embedding is None for p in corpus):
        raise SystemExit("Corpus not embedded -- run scripts/embed_corpus.py first.")
    retriever = EntityAwareRetriever(corpus)
    scorer = CachedScorer(QAScorer())
    ag = EvidenceAnswerGenerator()
    train_tasks = judgeable("train")
    candidates = {t["id"]: build_candidates(reqs_of(t), retriever, k=args.k) for t in train_tasks}

    # Reuse the bandit's normalizer for identical features; fit one if absent.
    norm_path = MODELS / "bandit_normalizer.joblib"
    if norm_path.exists():
        normalizer = joblib.load(norm_path)
    else:
        rng = random.Random(args.seed)
        ctxs = []
        for t in train_tasks:
            res = run_episode(t, candidates[t["id"]], scorer, ag,
                              lambda c, s: rng.randint(0, 1), lam=args.lam,
                              threshold=0.3, reward_mode="per_step")
            ctxs.extend(c for c, *_ in res.trajectory)
        normalizer = FeatureNormalizer().fit(ctxs)

    print(f"Training LinUCB + LinTS on {len(train_tasks)} tasks (seed={args.seed})")
    linucb = train_one(LinUCBPolicy(alpha=args.alpha, seed=args.seed), train_tasks,
                       candidates, normalizer, scorer, ag, args.lam, args.epochs, args.seed, "linucb")
    lints = train_one(LinTSPolicy(v=args.v, seed=args.seed), train_tasks,
                      candidates, normalizer, scorer, ag, args.lam, args.epochs, args.seed, "lints")

    linucb.save(MODELS / f"linucb_policy{args.suffix}.joblib")
    lints.save(MODELS / f"lints_policy{args.suffix}.joblib")
    print(f"Saved LinUCB + LinTS to {MODELS} (suffix={args.suffix!r})")


if __name__ == "__main__":
    main()
