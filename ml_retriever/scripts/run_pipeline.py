"""Phase 6 validation: run the full EcoBudgetSystem end-to-end on val tasks.

Decompose -> retrieve-under-(bandit)-policy -> generate answer -> measure bytes,
with NO ground truth in the loop. Reports, per task: the decomposed
requirements, the synthesized answer, bytes, and abstention; plus aggregate
auxiliary quality (token-F1 of the generated answer vs the display gold) and a
bytes comparison between the bandit policy and the heuristic. Every run records
component versions (decomposer / answer_generator / policy).

Usage: python scripts/run_pipeline.py --n 10
"""
import argparse
import json
from pathlib import Path

import joblib

from ml_retriever.answer import GenerativeAnswerGenerator
from ml_retriever.bandit import BanditPolicy, LinUCBPolicy, LinTSPolicy
from ml_retriever.decomposer import TaskDecomposer
from ml_retriever.evidence import CachedScorer, QAScorer
from ml_retriever.judge import judge_task, token_f1
from ml_retriever.retriever import EntityAwareRetriever
from ml_retriever.system import EcoBudgetSystem
from ml_retriever.types import Passage

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS = ROOT / "data", ROOT / "models"


def load_corpus():
    rows = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    return [Passage(passage_id=r["passage_id"], text=r["text"], byte_size=r["byte_size"],
                    embedding=r.get("embedding"), metadata=r.get("metadata", {})) for r in rows]


def gold_answer(t):
    """Unified gold: identical to what judge_task scores against, so token_f1
    and judge_success share ONE reference (fork A). required_facts for
    structured types, string for narrative, expected_answer only as fallback."""
    gt = t.get("ground_truth")
    if isinstance(gt, dict) and gt.get("required_facts"):
        return ", ".join(gt["required_facts"])
    if isinstance(gt, str):
        return gt
    return str(t.get("expected_answer") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--answerer", default="models/answer-base",
                    help="path to the generative answerer checkpoint (LoRA adapter dir); "
                         "models/answer-base is the reported model")
    ap.add_argument("--answer_mode", choices=["joint", "per_requirement"], default="joint")
    ap.add_argument("--policy", choices=["linucb", "bandit", "lints"], default="linucb",
                    help="stopping policy. Default linucb: it is stable across seeds "
                         "(multi-seed 0.913 +/- 0.007) unlike the SGD bandit "
                         "(0.790 +/- 0.283); see docs/phase_g_consolidation.md.")
    args = ap.parse_args()

    corpus = load_corpus()
    # Phase D: entity-aware two-stage retrieval (gate to entity, rank by
    # attribute) -- recall@1 0.914 -> 0.989, recall@5 -> 1.000 on seed reqs.
    retriever = EntityAwareRetriever(corpus)
    scorer = CachedScorer(QAScorer())
    decomposer = TaskDecomposer("models/decomposer-base-lora", adapter_path="models/decomposer-base-lora",
                                attribute_vocab=sorted({p.metadata["attribute"] for p in corpus}))
    answerer = GenerativeAnswerGenerator(args.answerer, adapter_path=args.answerer)
    _POLICIES = {"linucb": (LinUCBPolicy, "linucb_policy.joblib"),
                 "bandit": (BanditPolicy, "bandit_policy.joblib"),
                 "lints": (LinTSPolicy, "lints_policy.joblib")}
    _cls, _file = _POLICIES[args.policy]
    policy = _cls.load(MODELS / _file)
    policy.version = args.policy  # for SystemResult version reporting
    # all three policies share the same feature normalizer (fit once on train)
    normalizer = joblib.load(MODELS / "bandit_normalizer.joblib")

    tasks = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())}
    splits = json.loads((DATA / "splits.json").read_text())
    val = [tasks[i] for i in splits["val"] if ("expected_answer" in tasks[i] or "ground_truth" in tasks[i])]
    # de-dup to distinct questions for a readable 10-task run; seeds first
    seen, picked = set(), []
    for t in sorted(val, key=lambda t: t.get("synthetic", False)):
        if t["question"] in seen:
            continue
        seen.add(t["question"]); picked.append(t)
        if len(picked) >= args.n:
            break

    policy_sys = EcoBudgetSystem(decomposer, retriever, answerer, policy=policy,
                                 normalizer=normalizer, score_fn=scorer, answer_mode=args.answer_mode)
    heur_sys = EcoBudgetSystem(decomposer, retriever, answerer, score_fn=scorer,
                               answer_mode=args.answer_mode)

    from collections import defaultdict
    per_type = defaultdict(lambda: {"f1": 0.0, "succ": 0.0, "bytes": 0.0, "n": 0})
    f1_sum = succ = bytes_bandit = bytes_heur = 0.0
    print(f"{'='*70}\nEnd-to-end ({len(picked)} tasks) | policy={args.policy} | answerer={args.answerer} | mode={args.answer_mode}\n{'='*70}")
    for t in picked:
        res = policy_sys.run(t["question"])
        hres = heur_sys.run(t["question"])
        gold = gold_answer(t)
        ans = res.answer or ""
        s, fact_f1 = judge_task(ans, t)  # unified gold: success (binary) + fact_f1 (fraction)
        f1_sum += fact_f1; succ += s; bytes_bandit += res.bytes_used; bytes_heur += hres.bytes_used
        pt = per_type[t["answer_type"]]
        pt["f1"] += fact_f1; pt["succ"] += s; pt["bytes"] += res.bytes_used; pt["n"] += 1
        print(f"\n[{t['id']}/{t['answer_type']}] {t['question']}")
        print(f"  answer: {ans!r}  gold: {gold!r}")
        print(f"  judge_success={bool(s)} fact_f1={fact_f1:.2f} | bytes policy={res.bytes_used} heuristic={hres.bytes_used}")

    n = len(picked)
    print(f"\n{'='*70}")
    print(f"{'answer_type':<14}{'n':>3}{'fact_f1':>10}{'success':>9}{'avg_bytes':>11}")
    for at in sorted(per_type):
        d = per_type[at]; m = d["n"]
        print(f"{at:<14}{m:>3}{d['f1']/m:>10.3f}{d['succ']/m:>9.3f}{d['bytes']/m:>11.0f}")
    print(f"{'OVERALL':<14}{n:>3}{f1_sum/n:>10.3f}{succ/n:>9.3f}{bytes_bandit/n:>11.0f}")
    print(f"\navg bytes: bandit={bytes_bandit/n:.0f} heuristic={bytes_heur/n:.0f}")
    print(f"versions: {res.versions}")


if __name__ == "__main__":
    main()
