"""Phase F characterization (pre-registered): when does adaptive stopping beat
a strong hand-tuned heuristic?

Motivation. Under the Phase D entity-aware retriever (recall@1 0.989) the bandit
CONVERGES to the heuristic -- with near-perfect retrieval, "retrieve top-1 per
requirement, then stop" is optimal and there is no uncertainty to exploit
(eval_notes.md, Phase G). The honest hypothesis this experiment PRE-REGISTERS
before looking at results:

    H: the bandit's advantage over the heuristic grows with retrieval
       uncertainty. At noise=0 they tie; as the correct passage stops being
       reliably rank-1, the heuristic's fixed "stop after top-1" increasingly
       answers from a wrong passage, while the adaptive policy learns to retrieve
       more and preserves success.

Method. Wrap the entity-aware retriever in a controlled-noise layer: with
probability `noise` per requirement, demote the (gold-ish) top candidate so the
right passage is no longer rank-1. For each noise level we TRAIN a fresh bandit
(per-step reward, lambda=0.5 -- the Phase G selection) on that level's candidates
and evaluate bandit / heuristic / one_per_req / fixed@4 on val. This is a
characterization sweep on VAL, not a tuning run; the frozen test split is not
touched. Uses the fast EvidenceAnswerGenerator (no generative model), like
sweep_lambda.py, so it runs in minutes.

Usage: python scripts/retrieval_noise_sweep.py --epochs 8 --noises 0 0.1 0.2 0.4 0.6
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
LAM = 0.5  # Phase G selection under the entity-aware retriever


class NoisyRetriever:
    """Wraps a retriever and, with probability `noise` per requirement, demotes
    the top-ranked passage so the correct one is no longer rank-1 -- a controlled
    injection of retrieval uncertainty. Deterministic given a seed + requirement,
    so a task's candidates are stable across conditions at a fixed noise level."""

    def __init__(self, base, noise: float, seed: int = 0):
        self.base = base
        self.noise = noise
        self.seed = seed

    def retrieve(self, requirement: Requirement, k: int = 5):
        ranked = self.base.retrieve(requirement, k=k)
        if len(ranked) < 2 or self.noise <= 0:
            return ranked
        # per-(seed, requirement) deterministic coin
        rng = random.Random(f"{self.seed}:{requirement.key()}")
        if rng.random() < self.noise:
            # push the top candidate down to a random lower slot
            top = ranked.pop(0)
            pos = rng.randint(1, len(ranked))
            ranked.insert(pos, top)
        return ranked


def load_corpus():
    rows = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    return [Passage(passage_id=r["passage_id"], text=r["text"], byte_size=r["byte_size"],
                    source_url=r.get("source_url", ""), embedding=r.get("embedding"),
                    metadata=r.get("metadata", {})) for r in rows]


def judgeable(split, tasks, splits):
    return [tasks[i] for i in splits[split] if "expected_answer" in tasks[i] or "ground_truth" in tasks[i]]


def reqs_of(t):
    return [Requirement(entity=r["entity"], attribute=r["attribute"]) for r in t["decomposed_requirements"]]


def eval_decider(val, cands, scorer, ag, decide):
    s = b = r = 0.0
    for t in val:
        res = run_episode(t, cands[t["id"]], scorer, ag, decide, lam=LAM, threshold=THRESHOLD)
        s += res.success; b += res.total_bytes; r += res.n_retrieves
    n = len(val)
    return s / n, b / n, r / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--noises", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4, 0.6])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    corpus = load_corpus()
    base = EntityAwareRetriever(corpus)
    scorer = CachedScorer(QAScorer())
    ag = EvidenceAnswerGenerator()
    tasks = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())}
    splits = json.loads((DATA / "splits.json").read_text())
    train = judgeable("train", tasks, splits)
    val = judgeable("val", tasks, splits)
    rng = random.Random(args.seed)

    print("PRE-REGISTERED HYPOTHESIS: bandit's edge over the heuristic grows with")
    print("retrieval noise (they tie at noise=0). VAL only; test split untouched.\n")
    print(f"{'noise':>6} | {'bandit s/B/r':>22} | {'heuristic s/B':>16} | "
          f"{'one_per_req s/B':>16} | {'fixed@4 s/B':>14} | {'bandit-heur Δs':>14}")

    for noise in args.noises:
        retr = NoisyRetriever(base, noise, seed=args.seed)
        cands = {t["id"]: build_candidates(reqs_of(t), retr, k=K) for t in train + val}

        # fit normalizer on random train rollouts (reward-independent)
        ctxs = []
        for t in train:
            res = run_episode(t, cands[t["id"]], scorer, ag, lambda c, s: rng.randint(0, 1),
                              lam=LAM, threshold=THRESHOLD, reward_mode="per_step")
            ctxs.extend(c for c, *_ in res.trajectory)
        normalizer = FeatureNormalizer().fit(ctxs)

        # train a fresh bandit on this noise level
        policy = BanditPolicy(epsilon=0.2, seed=args.seed)
        for epoch in range(args.epochs):
            policy.epsilon = 0.2 * (1 - epoch / max(args.epochs - 1, 1))
            order = list(train); rng.shuffle(order)
            for t in order:
                res = run_episode(t, cands[t["id"]], scorer, ag,
                                  lambda c, s: policy.select_action(normalizer.transform(c), True),
                                  lam=LAM, threshold=THRESHOLD, reward_mode="per_step")
                for c, a, r in res.trajectory:
                    policy.update(normalizer.transform(c), a, r)

        bs, bb, br = eval_decider(val, cands, scorer, ag,
                                  lambda c, s: policy.select_action(normalizer.transform(c), False))
        hs, hb, _ = eval_decider(val, cands, scorer, ag, decide_heuristic)
        os_, ob, _ = eval_decider(val, cands, scorer, ag, decide_normal)
        fs, fb, _ = eval_decider(val, cands, scorer, ag, make_fixed_budget_decider(4))
        print(f"{noise:>6.2f} | {bs:>6.3f}/{bb:>5.0f}/{br:>4.2f}         | "
              f"{hs:>6.3f}/{hb:>6.0f}       | {os_:>6.3f}/{ob:>6.0f}       | "
              f"{fs:>6.3f}/{fb:>5.0f}      | {bs-hs:>+14.3f}")


if __name__ == "__main__":
    main()
