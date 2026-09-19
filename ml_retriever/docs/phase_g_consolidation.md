# Phase G consolidation: ablations, multi-seed variance, final test run

This document consolidates the rigor deliverables for Phase G. It gathers the
ablations into one place, reports multi-seed variance, and records the single
frozen test-split run. No em dashes are used. Every number is measured; the
source command is given for each block. Where a result is historical (from an
earlier phase, on an earlier model) it is labelled as such.

## 1. Consolidated ablation table

Each ablation isolates one design choice and shows what it buys.

### 1a. Retriever (recall on 175 unique seed requirements)
Source: `scripts/eval_retriever.py --k 1` and `--k 5`.

| retriever | recall@1 | recall@5 |
| --- | --- | --- |
| whole-question baseline | 0.589 | 0.931 |
| per-requirement bi-encoder | 0.914 | 0.983 |
| entity-aware, attribute-ranked (used) | 0.989 | 1.000 |

Takeaway: gating to the entity and ranking by the attribute is the single biggest
retrieval lever, and it fixes the list/amenities failure (recall@5 reaches 1.000).

### 1b. Decomposer snap-to-vocab (val, current Phase C model, n=240)
Source: `scripts/eval_decomposer.py` with and without `--no_attr_snap`.

| setting | exact-match | entity-EM | attribute_f1 |
| --- | --- | --- | --- |
| snap off | 0.496 | 0.946 | 0.822 |
| snap on (used) | 0.892 | 0.946 | 0.969 |

Takeaway: constrained decoding onto the closed 35-attribute vocabulary lifts
exact-match by about 0.40. Entity-EM is unchanged because snap is attribute-only
by design (a wrong entity is never rescued).

### 1c. Stopping policy: byte-penalty lambda sweep (val)
Source: `scripts/sweep_lambda.py` (fast evidence answerer).

| lambda | success | avg bytes | avg retrievals |
| --- | --- | --- | --- |
| 0.1 / 0.2 / 0.5 | 0.917 | 105 | 1.75 |
| 1.0 / 2.0 / 4.0 | 0.283 | 61 | 1.00 (collapsed to STOP) |

Takeaway: a sharp cliff between 0.5 and 1.0. We selected lambda = 0.5 (the
strongest byte incentive that stays stable), by a pre-registered rule.

### 1d. Stopping policy: reward mode (val, 3 seeds, evidence answerer)
Source: `scripts/ablate_reward.py --seeds 0 1 2 --lam 0.5`.

| reward mode | success (mean +/- std) | bytes (mean +/- std) |
| --- | --- | --- |
| per-step (used) | 0.706 +/- 0.366 | 90 +/- 26 |
| terminal | 0.989 +/- 0.019 | 256 +/- 144 |

Takeaway (honest): under the entity-aware retriever the two reward modes trade
off. Terminal credit reaches high success but over-retrieves (about 3x the bytes)
because coarse episode credit does not penalise a wasteful RETRIEVE. Per-step is
byte-lean but high variance across seeds, which is the same SGD-bandit instability
the multi-seed test shows. This is why we do not lean on the custom SGD bandit and
recommend LinUCB.

### 1e. Answer mode: joint vs per-requirement (historical, Phase 6)
Source: Phase 6 evaluation (see the answerer section of `eval_notes.md`).

| answer mode | comparison success |
| --- | --- |
| joint (one call, all evidence) | 0.375 |
| per-requirement (one call per requirement) | 0.750 |

Takeaway: answering each requirement separately removes the answerer's within-
passage attribute-localisation burden and roughly doubled comparison success.
This was established in Phase 6 and the pipeline uses per-requirement mode. It has
not been re-measured on the Phase C models; the design decision stands.

## 2. Multi-seed variance (val, 5 seeds)

Source: `scripts/multiseed.py --seeds 0 1 2 3 4 --epochs 8 --lam 0.5`. Shared
normalizer so only policy-training randomness varies.

| policy | success (mean +/- std) | bytes (mean +/- std) |
| --- | --- | --- |
| SGD bandit (ours) | 0.790 +/- 0.283 | 96 +/- 20 |
| LinUCB | 0.913 +/- 0.007 | 102 +/- 2 |
| LinTS | 0.897 +/- 0.014 | 101 +/- 2 |
| heuristic (deterministic) | 0.917 | 105 |

Takeaway: our SGD bandit is unstable across seeds; LinUCB and LinTS are stable and
match the deterministic heuristic. Recommended reported learned policy: LinUCB.

DEPLOYED DEFAULT: as of Phase G the deployed path (`run_pipeline.py`) defaults to
`--policy linucb`. LinUCB metrics of record: val multi-seed 0.913 +/- 0.007
success at 102 +/- 2 bytes (this section); frozen test 0.895 success / 0.944
fact_f1 / 108 bytes / 1.87 fetches (Section 3), tied with the bandit within noise.
`--policy bandit` and `--policy lints` remain available for comparison.

## 3. Single frozen test-split run (headline)

Source: `scripts/phase7_experiment.py --n 200 --split test` (run once, configs
frozen). Headline set: comparison, single_fact, multi_part, yes_no; procedure
excluded (n<3). n=124.

| condition | success | fact_f1 | avg bytes | fetches |
| --- | --- | --- | --- | --- |
| bandit | 0.895 | 0.944 | 106 | 1.84 |
| linucb | 0.895 | 0.944 | 108 | 1.87 |
| heuristic | 0.895 | 0.944 | 136 | 2.29 |
| one_per_req | 0.831 | 0.911 | 101 | 1.74 |
| full | 0.903 | 0.919 | 420 | 7.56 |

Bandit vs baselines, paired bootstrap 95% CI, measured 506 KB payload:
- vs full: bytes -314 [-350, -282], energy at page scale -66.3 J [-95.1, -40.8],
  radio -2.05 J [-2.17, -1.92]. All exclude 0.
- vs heuristic: bytes -30 [-51, -13] (about 22% fewer) at identical success.
- vs one_per_req: success +0.065 [+0.024, +0.113].
- vs LinUCB: tied within noise.

## 4. Status of Phase G deliverables

- Multi-seed training and variance: done (Section 2). Answerer multi-seed was not
  run because each answerer fine-tune takes about 3.5 hours on our laptop, making
  5 seeds prohibitive; the policy multi-seed (the scientifically important one for
  the stopping contribution) is complete. Stated as a resource limitation.
- Ablations: consolidated in Section 1 (retriever, snap, lambda, reward mode,
  answer mode).
- Frozen test run: done once (Section 3).
- Reproducibility appendix: `docs/reproducibility.md`.

## 5. Honest one-line status

Efficiency and energy contributions are solid and reproducible: task-sufficient
stopping matches full-page answer success at about 75% fewer bytes and lower 5G
transfer and radio energy, robust across cited coefficient ranges. The learned
policy should be reported as LinUCB (stable); the custom SGD bandit is retained as
an ablation, not the method.
