# Phase 7 pre-registration (frozen before running)

**Timestamp:** 2026-09-15. Written BEFORE any Phase 7 result was observed.
No component below is retuned after this point; any change invalidates the run.

## Frozen configuration (config of record)
- **Decomposer:** `models/decomposer-base-lora` (flan-t5-base, LoRA r=32 α=64
  q,v,k,o, 20 epochs), with attribute snap-to-vocab (closed 35-attr vocabulary).
- **Retriever:** Phase 3 `RequirementRetriever`, MiniLM bi-encoder, k=5, no
  cross-encoder. The entity-dominance issue is logged as future work and is
  NOT fixed for this run — Phase 7 tests the policy given the current retriever.
- **Evidence tracker:** `QAScorer` (roberta-base-squad2), threshold 0.3.
- **Answerer:** `models/answer-base` (flan-t5-base + LoRA), **per-requirement
  invocation** (one single-fact call per requirement, facts concatenated).
  Frozen — no retrain, no prompt change, no invocation change during Phase 7.
- **Judge:** curated `required_facts` gold + formatting normalizer, threshold
  1.0 (see eval_notes.md).
- **Bandit:** trained policy `models/bandit_policy.joblib` (per-step reward,
  λ=4) + `bandit_normalizer.joblib`. Not retuned (λ/reward/exploration frozen).

## Data
- **Split: val only.** 30 distinct-question judgeable val tasks. **Test split
  is untouched** and reserved for a single final run later.
- Val composition: comparison 24, single_fact 4, narrative 1, list 1.

## Conditions (7)
1. **normal** — retrieve top-1 per requirement, then stop.
2. **fixed-250B**, 3. **fixed-500B**, 4. **fixed-1000B**, 5. **fixed-1500B** —
   retrieve until cumulative payload ≥ budget (or no candidates).
   *Deviation from plan, pre-registered:* the plan's 10/25/50/100KB budgets all
   exceed the max per-task payload (1642 B; median 1136 B) for this short-text
   corpus, so they would every one collapse to "retrieve everything." Budgets
   are rescaled to span this corpus's payload range (≈250 B ≈ 1–2 passages up to
   ≈1500 B ≈ near-full). Same experimental intent, valid units.
6. **heuristic** — stop when the tracker reports sufficiency (EcoBudgetController
   analog, Baseline 3).
7. **bandit** — the trained contextual-bandit policy (greedy, no exploration).

## Metrics (per condition; per task type AND overall)
`judge_success` (binary), `fact_f1` (per-fact hit rate), `avg_bytes`
(= selected payload bytes loaded), `avg_retrieval_actions`, `avg_latency_s`.
`judge_success` and `fact_f1` reported separately, never averaged together.

## Analysis
- Headline: bandit vs heuristic and bandit vs fixed budgets, on
  success-at-equal-or-lower-bytes.
- n is small (comparison n=24), so report **effect sizes with bootstrap 95% CIs**
  (mean paired difference) alongside any p-value; do not overclaim from p<0.05.
- **Exclusion:** narrative (n=1) and list (n=1) are under-powered — reported
  separately, excluded from the headline and from significance tests.

## Expected result (stated before running)
Bandit reduces bytes vs fixed budgets while degrading success less than the
fixed budgets do; bandit ≈ heuristic success at ≤ heuristic bytes. The residual
retrieval gap (oracle 1.0 vs pipeline 0.75 on comparisons) means absolute
success is capped by the retriever, not the policy — Phase 7 tests the policy.

## Amendment (2026-09-15, before re-running; no corrected result seen yet)
Two condition bugs found in the first val run and corrected:
1. **`normal` was mislabeled.** Verified `decide_normal` = retrieve **top-1 per
   requirement, then stop** (minimal requirement-aware retrieval), NOT "retrieve
   everything." Relabeled **`one_per_req`**, and a true **`full`** condition
   (retrieve every candidate = max payload) added as the retrieve-everything
   reference. Conditions are now 8: one_per_req, full, fixed-250/500/1000/1500B,
   heuristic, bandit. Relabel is based on verified behavior, not on the numbers.
2. **Fixed byte budgets overran** (fixed-250B loaded 307B): the check added the
   passage that crossed the cap. Now STRICT — a passage is only retrieved if it
   fits within the remaining budget, so payload never exceeds the budget.
No model/policy/gold/judge change; pre-registration otherwise stands.
