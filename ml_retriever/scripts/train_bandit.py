"""Phase 5: train the contextual bandit (STOP / RETRIEVE_TOP1) online.

Loop (plan.md): for each training task, init a tracker, repeatedly ask the
policy until STOP -- retrieving/updating evidence on RETRIEVE -- then generate
an answer at STOP, judge it, compute the reward, and update the policy online.
The episode's scalar reward is credited to each step (Monte-Carlo). Step-level
context/action/reward are logged for debuggability.

Only tasks with ground truth are used (the judge needs it); the decomposer-only
scale-up seeds are skipped. Train split only -- val/test are never trained on.

Needs sentence-transformers (retriever) + the QA model (evidence scorer) + the
pre-embedded corpus. Saves the policy and the feature normalizer.

Usage:
    python scripts/train_bandit.py --epochs 8 --lam 0.5
"""

import argparse
import json
import random
from pathlib import Path

from ml_retriever.answer import EvidenceAnswerGenerator
from ml_retriever.bandit import BanditPolicy, FeatureNormalizer
from ml_retriever.evidence import CachedScorer, QAScorer
from ml_retriever.retriever import EntityAwareRetriever
from ml_retriever.rollout import build_candidates, run_episode
from ml_retriever.types import Passage, Requirement

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"


def load_corpus() -> list[Passage]:
    rows = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    return [Passage(passage_id=r["passage_id"], text=r["text"], byte_size=r["byte_size"],
                    source_url=r.get("source_url", ""), embedding=r.get("embedding"),
                    metadata=r.get("metadata", {})) for r in rows]


def judgeable_tasks(split_name: str) -> list[dict]:
    tasks = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())}
    splits = json.loads((DATA / "splits.json").read_text())
    out = []
    for tid in splits[split_name]:
        t = tasks[tid]
        if "expected_answer" in t or "ground_truth" in t:
            out.append(t)
    return out


def reqs_of(task: dict) -> list[Requirement]:
    return [Requirement(entity=r["entity"], attribute=r["attribute"])
            for r in task["decomposed_requirements"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epsilon", type=float, default=0.2, help="initial exploration; annealed to ~0")
    ap.add_argument("--reward_mode", choices=["terminal", "per_step"], default="per_step")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    MODELS.mkdir(exist_ok=True)

    corpus = load_corpus()
    if any(p.embedding is None for p in corpus):
        raise SystemExit("Corpus not embedded -- run scripts/embed_corpus.py first.")
    retriever = EntityAwareRetriever(corpus)  # Phase D/G: entity-gate + attribute-rank
    scorer = CachedScorer(QAScorer())  # shared cache across all episodes/epochs
    answer_gen = EvidenceAnswerGenerator()

    train_tasks = judgeable_tasks("train")
    print(f"Training on {len(train_tasks)} judgeable train tasks")

    # candidate sets per task (built once; requirement-level cache inside)
    candidates = {t["id"]: build_candidates(reqs_of(t), retriever, k=args.k) for t in train_tasks}

    # --- pre-pass: random rollouts to fit the feature normalizer ---
    rng = random.Random(args.seed)
    def random_decider(ctx, state):
        return rng.randint(0, 1)
    contexts = []
    for t in train_tasks:
        res = run_episode(t, candidates[t["id"]], scorer, answer_gen, random_decider,
                          lam=args.lam, threshold=args.threshold, reward_mode=args.reward_mode)
        contexts.extend(ctx for ctx, *_ in res.trajectory)
    normalizer = FeatureNormalizer().fit(contexts)

    # --- online training ---
    policy = BanditPolicy(epsilon=args.epsilon, seed=args.seed)
    log_path = MODELS / "bandit_train_log.jsonl"
    log = log_path.open("w")

    for epoch in range(args.epochs):
        # linear epsilon anneal to ~0 by the final epoch
        policy.epsilon = args.epsilon * (1 - epoch / max(args.epochs - 1, 1))
        order = list(train_tasks)
        rng.shuffle(order)
        ep_reward = ep_success = ep_bytes = 0.0
        for t in order:
            def decider(ctx, state):
                return policy.select_action(normalizer.transform(ctx), explore=True)
            res = run_episode(t, candidates[t["id"]], scorer, answer_gen, decider,
                              lam=args.lam, threshold=args.threshold, reward_mode=args.reward_mode)
            for ctx, action, step_reward in res.trajectory:
                policy.update(normalizer.transform(ctx), action, step_reward)
                log.write(json.dumps({
                    "epoch": epoch, "task_id": res.task_id, "action": int(action),
                    "step_reward": step_reward, "episode_reward": res.reward,
                    "success": res.success, "bytes": res.total_bytes,
                    "n_retrieves": res.n_retrieves,
                }) + "\n")
            ep_reward += res.reward
            ep_success += res.success
            ep_bytes += res.total_bytes
        n = len(order)
        print(f"epoch {epoch}: eps={policy.epsilon:.3f} "
              f"avg_reward={ep_reward/n:.3f} success={ep_success/n:.3f} "
              f"avg_bytes={ep_bytes/n:.0f}")

    log.close()
    policy.save(MODELS / "bandit_policy.joblib")
    import joblib
    joblib.dump(normalizer, MODELS / "bandit_normalizer.joblib")
    print(f"Saved policy + normalizer to {MODELS}; step log at {log_path}")


if __name__ == "__main__":
    main()
