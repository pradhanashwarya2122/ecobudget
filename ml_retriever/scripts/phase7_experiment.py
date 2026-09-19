"""Phase 7: full cross-condition comparison (frozen config; see
docs/phase7_preregistration.md). val split only; test untouched.

Seven conditions x 30 val tasks through the frozen pipeline (base per-requirement
answerer). Decomposition, retrieval, QA scores, and per-(requirement,evidence)
answers are cached so only the stop/continue policy differs across conditions.
Reports judge_success, fact_f1, avg_bytes, avg_retrieval_actions, avg_latency
per condition -- per task type and overall (narrative/list n=1 excluded from the
headline) -- plus bootstrap 95% CIs for bandit-vs-baseline byte and success
differences.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import joblib

from ml_retriever.answer import GenerativeAnswerGenerator
from ml_retriever.bandit import RETRIEVE, STOP, BanditPolicy, LinUCBPolicy, LinTSPolicy
from ml_retriever.decomposer import TaskDecomposer
from ml_retriever.energy import (
    SCENARIOS, EnergyAccountant, RadioStateModel, query_op_counts,
)
from ml_retriever.evidence import CachedScorer, EvidenceCoverageTracker, QAScorer
from ml_retriever.judge import judge_task
from ml_retriever.retriever import EntityAwareRetriever
from ml_retriever.rollout import (
    EpisodeState, build_candidates, build_context, decide_heuristic,
    decide_one_per_req, decide_full, decide_adaptive_rag, make_byte_budget_decider,
)
from ml_retriever.types import Passage, Requirement

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS = ROOT / "data", ROOT / "models"
THRESHOLD, K, MAX_STEPS = 0.3, 5, 30
HEADLINE_TYPES = {"comparison", "single_fact", "multi_part", "yes_no"}  # n>=3 only


def load_corpus():
    rows = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    return [Passage(passage_id=r["passage_id"], text=r["text"], byte_size=r["byte_size"],
                    source_url=r.get("source_url", ""),  # needed for page-level payload dedup
                    embedding=r.get("embedding"), metadata=r.get("metadata", {})) for r in rows]


def gold_answer(t):
    gt = t.get("ground_truth")
    if isinstance(gt, dict) and gt.get("required_facts"):
        return ", ".join(gt["required_facts"])
    if isinstance(gt, str):
        return gt
    return str(t.get("expected_answer") or "")


def gather(reqs, candidates, scorer, decide):
    tracker = EvidenceCoverageTracker(reqs, threshold=THRESHOLD, score_fn=scorer)
    max_bytes = sum(c.passage.byte_size for cs in candidates.values() for c in cs) or 1
    state = EpisodeState(requirements=reqs, candidates=candidates, tracker=tracker, max_bytes=max_bytes)
    while state.step < MAX_STEPS:
        ctx = build_context(state)
        a = decide(ctx, state)
        if a == RETRIEVE and not state.has_retrievable():
            a = STOP
        if a == STOP:
            break
        req = state.retrieval_target(); cand = state.next_candidate(req)
        tracker.add_passage(cand.passage); state.added_ids.add(cand.passage.passage_id)
        state.bytes_used += cand.passage.byte_size; state.step += 1
    return state


def per_req_answer(reqs, candidates, state, by_id, answerer, cache):
    facts = []
    for req in reqs:
        req_pids = {c.passage.passage_id for c in candidates[req.key()]}
        ev_ids = tuple(sorted(pid for pid in state.added_ids if pid in req_pids))
        if not ev_ids:
            continue
        key = (req.entity, req.attribute, ev_ids)
        if key not in cache:
            ev = [by_id[pid] for pid in ev_ids]
            q = f"What is the {req.attribute.replace('_', ' ')} of {req.entity}?"
            cache[key] = answerer.generate(q, ev).answer or ""
        if cache[key]:
            facts.append(cache[key])
    return ", ".join(facts)


def bootstrap_ci(diffs, iters=2000, seed=0):
    rng = random.Random(seed)
    means = []
    n = len(diffs)
    for _ in range(iters):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--split", choices=["val", "test"], default="val",
                    help="evaluation split. 'test' is the FROZEN final run -- use exactly once.")
    args = ap.parse_args()
    if args.split == "test":
        print("!! FROZEN TEST SPLIT: this is the single final run. Configs must be frozen. !!\n")

    corpus = load_corpus()
    by_id = {p.passage_id: p for p in corpus}
    retriever = EntityAwareRetriever(corpus)  # Phase D/G: entity-gate + attribute-rank
    scorer = CachedScorer(QAScorer())
    decomposer = TaskDecomposer("models/decomposer-base-lora", adapter_path="models/decomposer-base-lora",
                                attribute_vocab=sorted({p.metadata["attribute"] for p in corpus}))
    answerer = GenerativeAnswerGenerator("models/answer-base", adapter_path="models/answer-base")
    policy = BanditPolicy.load(MODELS / "bandit_policy.joblib")
    normalizer = joblib.load(MODELS / "bandit_normalizer.joblib")
    # Phase E external baselines (trained by scripts/train_baselines.py)
    linucb = LinUCBPolicy.load(MODELS / "linucb_policy.joblib")
    lints = LinTSPolicy.load(MODELS / "lints_policy.joblib")

    tasks = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())}
    splits = json.loads((DATA / "splits.json").read_text())
    val = [tasks[i] for i in splits[args.split] if "expected_answer" in tasks[i] or "ground_truth" in tasks[i]]
    seen = set(); val = [t for t in val if not (t["question"] in seen or seen.add(t["question"]))][:args.n]

    # precompute decomposition + candidates once per task (shared across conditions)
    pre = {}
    for t in val:
        reqs = decomposer.decompose(t["question"])
        cand = build_candidates(reqs, retriever, k=K)
        pre[t["id"]] = (reqs, cand)

    conditions = {
        "one_per_req": decide_one_per_req,   # top-1 per requirement (minimal), NOT "retrieve everything"
        "full": decide_full,                 # retrieve every candidate (max payload)
        "fixed-250B": make_byte_budget_decider(250),
        "fixed-500B": make_byte_budget_decider(500),
        "fixed-1000B": make_byte_budget_decider(1000),
        "fixed-1500B": make_byte_budget_decider(1500),
        "heuristic": decide_heuristic,
        "adaptive_rag": decide_adaptive_rag,  # Phase E: Adaptive-RAG complexity-routing analog
        "linucb": lambda ctx, st: linucb.select_action(normalizer.transform(ctx), explore=False),
        "lints": lambda ctx, st: lints.select_action(normalizer.transform(ctx), explore=False),
        "bandit": lambda ctx, st: policy.select_action(normalizer.transform(ctx), explore=False),
    }

    accountant = EnergyAccountant()  # Phase A: 5G transfer + compute energy
    radio = RadioStateModel()                       # Phase F: RRC tail energy (tight loop, conservative)
    radio_fd = RadioStateModel(demote_between_fetches=True)  # Phase F: fast-dormancy sensitivity
    ans_cache = {}
    results = {c: {} for c in conditions}  # condition -> task_id -> metrics
    for cname, decide in conditions.items():
        for t in val:
            reqs, cand = pre[t["id"]]
            t0 = time.perf_counter()
            state = gather(reqs, cand, scorer, decide)
            answer = per_req_answer(reqs, cand, state, by_id, answerer, ans_cache)
            latency = time.perf_counter() - t0
            s, f1 = judge_task(answer, t)
            # logical op counts (caching-independent) -> compute vs 5G transfer energy
            ops = query_op_counts(len(reqs), len(state.added_ids), answer_mode="per_requirement")
            added = [by_id[pid] for pid in state.added_ids if pid in by_id]
            e = accountant.account_passages(ops, added, "text")  # Phase A term (snippet bytes)
            rec = {"success": float(s), "fact_f1": f1, "bytes": state.bytes_used,
                   "actions": len(state.added_ids), "latency": latency,
                   "compute_j": e["compute_j"], "transfer_j": e["transfer_j"],
                   "total_j": e["total_j"], "gco2e": e["gco2e"], "type": t["answer_type"]}
            # Phase B: total energy under each payload realism scenario
            for sc in SCENARIOS:
                rec[f"total_j_{sc}"] = accountant.account_passages(ops, added, sc)["total_j"]
                rec[f"transfer_j_{sc}"] = accountant.account_passages(ops, added, sc)["transfer_j"]
            # Phase F: radio-state / tail energy + modeled latency. Radio is
            # driven by #fetches (actions) and the transported bytes at the
            # realistic html_page scale (the real 5G regime, per Phase B).
            n_fetch = len(state.added_ids)
            html_bytes = accountant.payload.transfer_bytes(added, "html_page")
            r = radio.account(html_bytes, n_fetch)
            rfd = radio_fd.account(html_bytes, n_fetch)
            rec["radio_j"] = r["radio_j"]
            rec["radio_time"] = r["radio_active_time_s"]
            rec["radio_tail_j"] = r["tail_j"]
            rec["radio_j_fastdormancy"] = rfd["radio_j"]
            # modeled end-to-end latency = compute time + transfer time (html_page)
            rec["compute_s"] = accountant.compute.time_seconds(ops)
            rec["transfer_s"] = radio.transfer_time_s(html_bytes)
            rec["e2e_s"] = rec["compute_s"] + rec["transfer_s"]
            results[cname][t["id"]] = rec

    _AGG_KEYS = ("success", "fact_f1", "bytes", "actions", "latency",
                 "compute_j", "transfer_j", "total_j", "gco2e",
                 "radio_j", "radio_time", "radio_tail_j", "radio_j_fastdormancy",
                 "compute_s", "transfer_s", "e2e_s") + \
                tuple(f"total_j_{sc}" for sc in SCENARIOS) + tuple(f"transfer_j_{sc}" for sc in SCENARIOS)

    def agg(cname, types=None):
        rows = [m for m in results[cname].values() if types is None or m["type"] in types]
        n = len(rows)
        return {k: sum(m[k] for m in rows) / n for k in _AGG_KEYS} | {"n": n}

    print("=" * 78)
    print("PHASE 7 — headline (comparison, single_fact, multi_part, yes_no; "
          "procedure excluded, n>=3; n=%d)" % agg("bandit", HEADLINE_TYPES)["n"])
    print(f"{'condition':<13}{'success':>9}{'fact_f1':>9}{'avg_bytes':>11}{'actions':>9}{'latency_s':>10}")
    for c in conditions:
        a = agg(c, HEADLINE_TYPES)
        print(f"{c:<13}{a['success']:>9.3f}{a['fact_f1']:>9.3f}{a['bytes']:>11.0f}{a['actions']:>9.2f}{a['latency']:>10.2f}")

    print("\nENERGY per query (Phase A: 5G transfer vs compute), headline tasks:")
    print(f"{'condition':<13}{'compute_J':>11}{'transfer_J':>12}{'total_J':>10}{'compute%':>10}{'gCO2e':>10}")
    for c in conditions:
        a = agg(c, HEADLINE_TYPES)
        cf = 100 * a["compute_j"] / a["total_j"] if a["total_j"] else 0
        print(f"{c:<13}{a['compute_j']:>11.3f}{a['transfer_j']:>12.4f}{a['total_j']:>10.3f}{cf:>9.1f}%{a['gco2e']:>10.4f}")

    print("\nPAYLOAD SENSITIVITY (Phase B): avg total_J per query by scenario, headline tasks")
    print(f"{'condition':<13}" + "".join(f"{sc:>13}" for sc in SCENARIOS))
    for c in conditions:
        a = agg(c, HEADLINE_TYPES)
        print(f"{c:<13}" + "".join(f"{a['total_j_'+sc]:>13.2f}" for sc in SCENARIOS))
    # crossover: transfer share for the bandit at each scenario
    ab = agg("bandit", HEADLINE_TYPES)
    print("bandit transfer share:  " + "  ".join(
        f"{sc}={100*ab['transfer_j_'+sc]/ab['total_j_'+sc]:.1f}%" for sc in SCENARIOS))

    print("\nRADIO STATE / TAIL ENERGY (Phase F): per query, headline tasks")
    print("  (radio held in RRC_CONNECTED across the retrieval loop; tail after last fetch)")
    print(f"{'condition':<13}{'fetches':>9}{'radio_time_s':>14}{'radio_J':>10}{'tail_J':>9}{'radio_J[fastD]':>16}")
    for c in conditions:
        a = agg(c, HEADLINE_TYPES)
        print(f"{c:<13}{a['actions']:>9.2f}{a['radio_time']:>14.2f}{a['radio_j']:>10.2f}"
              f"{a['radio_tail_j']:>9.2f}{a['radio_j_fastdormancy']:>16.2f}")

    print("\nLATENCY (Phase F, modeled): compute + transfer(html_page) per query, headline tasks")
    print(f"{'condition':<13}{'compute_s':>11}{'transfer_s':>12}{'e2e_s':>9}")
    for c in conditions:
        a = agg(c, HEADLINE_TYPES)
        print(f"{c:<13}{a['compute_s']:>11.3f}{a['transfer_s']:>12.4f}{a['e2e_s']:>9.3f}")

    print("\nper-type judge_success / avg_bytes:")
    types = sorted({m["type"] for m in results["bandit"].values()})
    print(f"{'condition':<13}" + "".join(f"{ty[:10]:>18}" for ty in types))
    for c in conditions:
        cells = []
        for ty in types:
            a = agg(c, {ty})
            cells.append(f"{a['success']:.2f}/{a['bytes']:.0f}({a['n']})")
        print(f"{c:<13}" + "".join(f"{cell:>18}" for cell in cells))

    # bootstrap: bandit vs baselines on headline tasks (paired)
    ids = [tid for tid, m in results["bandit"].items() if m["type"] in HEADLINE_TYPES]
    print("\nbandit vs baseline (headline tasks, paired mean diff [95% bootstrap CI]):")
    for base in ("heuristic", "adaptive_rag", "linucb", "lints", "one_per_req",
                 "full", "fixed-1000B", "fixed-1500B"):
        db = [results["bandit"][i]["bytes"] - results[base][i]["bytes"] for i in ids]
        ds = [results["bandit"][i]["success"] - results[base][i]["success"] for i in ids]
        de = [results["bandit"][i]["total_j"] - results[base][i]["total_j"] for i in ids]  # text scale
        dh = [results["bandit"][i]["total_j_html_page"] - results[base][i]["total_j_html_page"] for i in ids]
        dr = [results["bandit"][i]["radio_j"] - results[base][i]["radio_j"] for i in ids]  # Phase F
        blo, bhi = bootstrap_ci(db); slo, shi = bootstrap_ci(ds)
        elo, ehi = bootstrap_ci(de); hlo, hhi = bootstrap_ci(dh); rlo, rhi = bootstrap_ci(dr)
        print(f"  vs {base:<12} bytes {sum(db)/len(db):+.0f}[{blo:+.0f},{bhi:+.0f}]  "
              f"success {sum(ds)/len(ds):+.3f}[{slo:+.3f},{shi:+.3f}]  "
              f"net_J(text) {sum(de)/len(de):+.2f}[{elo:+.2f},{ehi:+.2f}]  "
              f"net_J(html_page) {sum(dh)/len(dh):+.1f}[{hlo:+.1f},{hhi:+.1f}]  "
              f"radio_J {sum(dr)/len(dr):+.2f}[{rlo:+.2f},{rhi:+.2f}]")

    (DATA / "phase7_results.json").write_text(json.dumps(
        {c: agg(c) for c in conditions} | {"per_type": {c: {ty: agg(c, {ty}) for ty in types} for c in conditions}},
        indent=2))
    print(f"\nWrote {DATA/'phase7_results.json'}")


if __name__ == "__main__":
    main()
