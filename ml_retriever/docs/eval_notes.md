# Evaluation notes

## Decomposer (Phase 2) — attribute snap-to-vocab policy

**What it is.** `TaskDecomposer(attribute_vocab=...)` snaps each generated
attribute to the nearest slug in the corpus's closed 35-attribute vocabulary
(`ml_retriever/decomposer.py: snap_attribute`). This is constrained decoding
over a known finite label set, not a repair of wrong decisions.

**Scope & threshold.**
- **Attribute-only.** Entities are **never** snapped — a wrong entity is a real
  failure the metric must show.
- Snap first takes an exact match on the normalized form (lowercase, drop glue
  words {of,the,a,an}, token-sort); otherwise the nearest slug by Levenshtein
  distance, accepted only if within `0.4 * max(len(pred_norm), len(slug_norm))`.
  Far-off predictions are left unchanged rather than force-snapped.

**Why it's honest here (verification, base+LoRA on the 36-example val split).**
Failure analysis of the 22 missed requirements (of 60) shows the errors are
overwhelmingly *spelling corruption of the correct attribute* (e.g.
`display_refresh_rate` → `displays_recover_rate`) on the second entity of
comparison questions — not entity-type-as-attribute confusion (that was ~1
outlier). Verification that snap does not rescue wrong intent:
- Conceptual errors (genuinely different attribute, e.g. `shoe`/`light_pair`
  for `weight`): **6/6 remain wrong after snap.**
- Dropped-entity requirements: **6/6 remain dropped after snap** (snap cannot
  create a missing requirement).
- 5 cases where snap mapped a garble to the gold slug were confirmed to be
  severe corruptions of the *correct* attribute (`displays_recover_rate` →
  `display_refresh_rate`, `noises_cancelment` → `noise_cancellation`): the
  question asks about that attribute, the garble shares the slug's word-roots,
  and the same attribute was emitted correctly for the pair's first entity.
  These are legitimate constrained-decoding fixes. **No wrong-intent false
  rescue was found.**

**Metrics (Phase 2, val, base+LoRA, config: flan-t5-base, LoRA r=32 α=64 q,v,k,o,
20 epochs, 300 train pairs).** HISTORICAL — this is the Phase 2 model on the old
36-example val split.
- exact-match, un-snapped: **0.472**
- exact-match, snapped: **0.75**
- per-requirement accuracy (intent), un-snapped: **0.63**
- entity_f1 ≈ 0.89, attribute_f1 ≈ 0.87 (snapped)

**CURRENT metrics (re-measured 2026-09-17 after the Phase C retrain on 1008 train
pairs, evaluated on the current 240-example val split, snapped).** The Phase C
retrain improved the decomposer substantially over the Phase 2 numbers above:
- exact-match accuracy: **0.892**
- entity exact-match: **0.946**
- per-requirement precision/recall/F1: **0.936**
- entity_f1 **0.979**, attribute_f1 **0.969**
- mean latency 0.23 s/query
Command: `python scripts/eval_decomposer.py --model_path models/decomposer-base-lora
--adapter_path models/decomposer-base-lora`.

## Phase 3 retrieval — known issue: entity dominance / attribute collision

For the requirement `(Oberoi Rajvilas Jaipur, amenities)`, the correct
amenities passage is **not in the bi-encoder top-5**: the same hotel's `rating`
passage ("5-star hotel…", sim 0.745) and `price_per_night` (0.694) outrank it,
and the only amenities passage in the top-5 is the *wrong hotel's* (Ibis, 0.543).
The MiniLM query "Oberoi Rajvilas Jaipur amenities" is dominated by the entity
name, so it can't localize the attribute within the entity. The cross-encoder
re-rank does **not** fix it (top-1 stays `rating`). This is the same
entity-dominance effect seen in Phase 4 sufficiency scoring.

Impact: the pipeline fed the T38 list task a `rating` passage, so the answerer
returned "5-star" — a retrieval failure, NOT an answerer failure (given the
correct evidence, both small and base enumerate the amenities correctly).

Proposed Phase 3 fix (not yet implemented): entity-aware retrieval — when a
requirement names an entity, restrict/boost candidates to passages of that
entity, then rank by attribute similarity. Offline we have entity metadata;
a live fetcher would fetch per-entity anyway. Tracked, not fixed here.

**Gate decision.** Snapped exact-match 0.75 is accepted as meeting the ≥0.65
decomposer gate, justified by the verification above: for a closed attribute
vocabulary, snapping is principled constrained decoding, and the residual
genuinely-wrong (conceptual + dropped) rate is ~11/60 ≈ 18% of requirements,
concentrated on second-entity comparison generation. The reported pipeline uses
the snapped decomposer.

## Answerer / judge (Phase 6, fork A)

**Gold format.** All comparison/yes_no/multi_part/list tasks are judged against
hand-curated `required_facts` (canonical values the source passages contain,
`scripts/curate_gold.py`) — NOT the `expected_answer` verdict string, which is
retained display-only. Training targets and both eval metrics use this one gold,
so train == eval.

**Judge formatting normalizer** (`judge._normalize`, symmetric on gold & pred):
- lowercase, collapse whitespace
- strip `$` and thousands commas: `$1,099` == `1099`
- number-unit spacing: `120Hz` == `120 Hz`
- trailing `.0` decimals: `799.00` == `799`, `10.0 ounces` == `10 ounces`
- drop leading articles: `the iPhone 15` == `iPhone 15`

Does NOT cover: synonyms/paraphrase (`oz` ≠ `ounces`), unit conversion, semantic
equivalence. Match threshold stays 1.0 (every required fact must match after
normalization).

**Metrics.** `judge_success` (binary: all facts matched) and `fact_f1`
(continuous per-fact hit rate = fraction of required_facts matched), reported
separately — the 0.5 gate is on `judge_success`, `fact_f1` shows how close.

## Retriever, entity-aware (Phase D)

**Problem (measured).** On the 175 unique seed-task requirements, the plain
per-requirement bi-encoder (`RequirementRetriever`) hits recall@1 0.914,
recall@5 0.983. The 15 recall@1 misses split into two failure modes:
- **entity-dominance (5):** a near-twin entity outranks the target because the
  entity name dominates the query embedding — `Samsung Galaxy S24` → `S24 Ultra`,
  `iPhone 15` → `iPhone 16`, `Pixel 8` → `Pixel 9`, `Jaipur` → an
  `Oberoi ... Jaipur` hotel passage.
- **attribute-confusion (10):** correct entity, wrong attribute wins — Empire
  State Building `height`/`architect`/`completion_year` all collapse onto
  `floors`; `Japan area` → `capital`. Three (Oberoi/Ibis `amenities`,
  `room_count`) fell out of top-5 entirely — the known list/amenities failure.

**Fix (`EntityAwareRetriever`).** Two stages: (1) **entity gate** — keep only
passages matching the requirement's entity (exact normalized match on
`metadata['entity']` offline; word-boundary text mention as a live-fetch
fallback; full-corpus fallback if the gate is empty, so recall never drops for
unknown entities). Normalization is case/space-only and does NOT drop tokens, so
`S24` and `S24 Ultra` stay distinct. (2) **attribute rank** — rank the gated
pool by similarity to the *attribute phrase alone* (`attribute_to_query`); within
one entity the name is constant, so the attribute discriminates.

**Result (seed reqs, n=175, `scripts/eval_retriever.py`).**
| system | recall@1 | recall@5 |
|---|---|---|
| whole-question baseline | 0.589 | 0.931 |
| per-requirement bi-encoder | 0.914 | 0.983 |
| **entity-aware (attr-rank)** | **0.989** | **1.000** |

Multi-requirement subset (n=104): recall@1 0.885 → **0.981**, recall@5 → **1.000**.
Ranking the *full* requirement within the gate instead of the attribute only
recovers just 0.943 recall@1 — confirming the entity name was the dominating
noise. recall@5 = 1.000 means every list/amenities gold passage is now retrieved
(roadmap Phase D acceptance gate met).

**Residual (honest).** 2 recall@1 misses remain, both same-entity semantic
near-synonyms with gold at rank 2: Empire State `completion_year` and Kyoto
`best_time_to_visit`. Not hacked around — a genuine attribute-embedding ceiling.

**Scope.** UPDATE: as of Phase G this is fully adopted. `run_pipeline.py`,
`train_bandit.py`, `train_baselines.py`, `phase7_experiment.py`, `eval_bandit.py`,
and `sweep_lambda.py` all build `EntityAwareRetriever`, the bandit was retrained
on entity-aware candidates, and Phase 7 was re-run. (This note previously said
adoption was deferred; that is no longer true.)

## Phase 7 re-run under the entity-aware retriever (Phase G consolidation)

After wiring `EntityAwareRetriever` into `train_bandit` / `phase7_experiment`,
the bandit had to be recalibrated: the entity gate shrinks the per-query byte
scale, so the old λ=4.0 (selected against the old retriever) over-penalized
retrieval and the policy collapsed to always-STOP (val success 0.28). Re-running
the pre-registered λ sweep under the new retriever showed a sharp cliff between
λ=0.5 (success 0.917, 105 B) and λ=1.0 (collapse). Selected **λ=0.5** by the
same rule as before (strongest stable byte incentive). VAL ONLY — test frozen.

**Corrected Phase 7 (val, n=92 headline: comparison, single_fact, multi_part, yes_no (procedure excluded), generative
answerer `models/answer-base`, λ=0.5 bandit):**

| condition | success | fact_f1 | avg_bytes | net_J text | net_J html_page |
|---|---|---|---|---|---|
| one_per_req | 0.946 | 0.973 | 124 | 5.93 | 18.33 |
| **heuristic** | 0.946 | 0.973 | 134 | 6.02 | 19.38 |
| **bandit (λ=0.5)** | 0.946 | 0.973 | 134 | 6.02 | 19.38 |
| fixed-500B | 0.957 | 0.978 | 426 | 12.15 | 29.63 |
| fixed-1000B | 0.957 | 0.957 | 535 | 13.16 | 37.08 |
| full | 0.957 | 0.957 | 552 | 13.30 | 38.65 |

**Bandit vs baselines (paired mean diff [95% bootstrap CI]):**
- vs heuristic: bytes +0 [0,0], success +0.000 [0,0] — **identical policy**.
- vs one_per_req: bytes +11 [+4,+18], success +0.000 — essentially tied.
- vs full: bytes −418 [−460,−382], success −0.011 [−0.076,+0.054] (equal),
  net_J(html_page) −19.3 [−23.9,−15.3].
- vs fixed-1000B: bytes −400 [−432,−373], net_J(html_page) −17.7 [−21.4,−14.5].

**Honest narrative shift (important).** Under the *old* retriever the bandit beat
the heuristic by ~29% bytes / −67 B / −0.82 J at equal success. Under the Phase D
entity-aware retriever (recall@1 0.989) the bandit **converges to the heuristic's
policy** — "retrieve the top-1 per requirement, then stop" is now near-optimal
because the first retrieved passage is almost always correct, so there is little
adaptive headroom left to exploit. This is a genuine Phase-D↔Phase-5 tension, not
hidden:
- The learned policy still **dominates fixed budgets and full-load** (matches
  their success at ~75% fewer bytes and −18 to −19 J/query at html_page scale,
  all CIs excluding 0) — the Pareto claim holds.
- Its edge *over a strong hand-designed heuristic* is a function of retrieval
  uncertainty: large when recall is imperfect, ~0 when retrieval is near-perfect.
- Revised paper framing: "a learned per-step-reward stopping policy recovers the
  hand-tuned heuristic's behavior without hand-tuning and dominates fixed
  budgets; its advantage over the heuristic grows with retrieval uncertainty."
  Do NOT claim the bandit beats the heuristic under the entity-aware retriever.

## Phase E: external baselines, multi-seed variance, and the frozen test run (Tier 1)

### Phase E: standard contextual-bandit baselines
Added LinUCB and linear Thompson sampling (`ml_retriever/bandit.py`,
`scripts/train_baselines.py`) sharing the exact feature vector, per-step reward,
lambda=0.5, entity-aware candidates, and feature normalizer as our SGD bandit,
plus an Adaptive-RAG complexity-routing analog (`rollout.decide_adaptive_rag`).

Validation (n=92 headline): all adaptive methods CONVERGE. bandit / linucb / lints
all reach 0.946 success; linucb and lints use 126 bytes vs our bandit's 134 (our
bandit is +9 bytes [+2,+17], success identical). adaptive_rag == heuristic. Every
adaptive method dominates fixed/full by the same margin. Conclusion: we cannot
claim our SGD bandit is a better ALGORITHM than standard bandits; it ties them.

### Multi-seed variance (5 seeds, val, `scripts/multiseed.py`)
Shared normalizer; only the policy seed varies, so this isolates policy-training
randomness.

| policy | success mean +/- std | bytes mean +/- std |
| --- | --- | --- |
| SGD bandit (ours) | 0.790 +/- 0.283 | 96 +/- 20 |
| LinUCB | 0.913 +/- 0.007 | 102 +/- 2 |
| LinTS | 0.897 +/- 0.014 | 101 +/- 2 |
| heuristic (deterministic) | 0.917 | 105 |

KEY FINDING: our online SGD bandit is UNSTABLE across seeds (it collapses on some
seeds); the single-seed 0.946 we had reported was a favorable seed. LinUCB and
LinTS are stable and higher on average, essentially matching the deterministic
heuristic. RECOMMENDED FRAMING PIVOT: make LinUCB the headline learned policy
(stable, principled, standard, matches heuristic, dominates fixed budgets); do not
present the custom SGD bandit as the contribution. The real contribution is the
task-sufficient framing plus the 5G radio/energy characterization (Phase F).

### Frozen test-split run (single, final) -- `phase7_experiment.py --split test`
Everything frozen (models, lambda=0.5, all conditions). n=124 headline tasks.
This is the one and only test run.

| condition | success | fact_f1 | avg_bytes | fetches |
| --- | --- | --- | --- | --- |
| bandit | 0.895 | 0.944 | 106 | 1.84 |
| linucb | 0.895 | 0.944 | 108 | 1.87 |
| lints | 0.895 | 0.944 | 107 | 1.84 |
| heuristic | 0.895 | 0.944 | 136 | 2.29 |
| adaptive_rag | 0.895 | 0.944 | 136 | 2.29 |
| one_per_req | 0.831 | 0.911 | 101 | 1.74 |
| fixed-500B | 0.927 | 0.944 | 353 | 6.93 |
| full | 0.903 | 0.919 | 420 | 7.56 |

Paired bootstrap (bandit vs baseline, 95% CI), test. Energy uses the MEASURED
506 KB html_page payload (Tier A), re-run 2026-09-17 on the free machine:
- vs heuristic: bytes -30 [-51,-13], success +0.000, net_J(html_page) -10.9
  [-23.0,-1.9], radio_J -0.17 [-0.27,-0.07]. On TEST the learned policies DO save
  bytes vs the heuristic (~22%) at identical success, CI excluding 0 (on val they
  had exactly tied). A small, real win.
- vs linucb / lints: bytes within noise (CI includes 0), success identical -- the
  learned policies are interchangeable on test.
- vs one_per_req: success +0.065 [+0.024,+0.113] -- adaptive stopping beats naive
  top-1-per-requirement (which under-retrieves) at trivial byte cost.
- vs full: bytes -314 [-350,-282], net_J(html_page) -66.3 [-95.1,-40.8], radio_J
  -2.05 [-2.17,-1.92] -- dominates decisively (all CIs exclude 0).

### Honest headline for the paper (test-set, defensible)
Task-sufficient adaptive stopping reaches 0.895 success / 0.944 fact_f1 at ~106
bytes, saves ~22% bytes versus a strong adaptive heuristic at equal success,
beats naive per-requirement retrieval on success, and dominates fixed budgets and
full-load by ~314 bytes and ~66 J/query at the measured realistic page scale (all
CIs exclude 0). Among learned policies, LinUCB is the stable recommended choice;
our custom SGD bandit matches it on a good seed but is high-variance across seeds.

## Tier A/B capstone: end-to-end on REAL live web pages

`scripts/end_to_end_web.py` fetches a real page (browser UA, fixture fallback),
splits its visible text into passages, and runs the actual pipeline (decompose ->
per-requirement retrieval -> adaptive stop -> answer) over them, measuring bytes
and energy against loading the whole page. Grounded 5G energy model (Tier A/B).

Measured (live gsmarena pages):
| question | full page | task-sufficient | byte reduction | transfer J (full -> task) |
| --- | --- | --- | --- | --- |
| iPhone 16 refresh rate | 54,012 B | 3,878 B | 92.8% | 9.89 -> 0.71 |
| S24 Ultra weight | 56,296 B | 1,228 B | 97.8% | 10.31 -> 0.23 |

Two honest takeaways:
1. POSITIVE: task-sufficient loading delivers a measured ~93-98% byte reduction
   and ~14x transfer-energy reduction on real pages. The core thesis holds on
   live web content, not just the curated corpus.
2. LIMITATION: answer QUALITY degrades on raw live-page text. The decomposer,
   retriever, and answerer were trained on a clean structured corpus ("The
   iPhone 16 has a display with a 120Hz refresh rate"), so on messy real-page
   spec dumps the answerer produces malformed output (though the correct value,
   e.g. S24 Ultra "232g", is present in the retrieved passage). This corpus-to-web
   domain gap is the honest ceiling of the current pipeline: byte savings transfer
   to the real web immediately; answer quality needs training on real-page chunks
   (or a cleaner extraction stage) to match. Reported, not hidden.

## Domain-gap fix attempt: noise-augmented answerer (honest result)

To close the corpus-to-web gap we retrained the answerer on noise-augmented data
(`build_answer_training_data.py --noise_aug 2`, 2,904 rows = 968 clean + 1,936
with the gold passage embedded in distractor spec-dump text), saved as
`models/answer-base-robust` (10 epochs, final train loss ~0.002). Re-running
`end_to_end_web.py --answerer models/answer-base-robust` on the same live
gsmarena pages:

| question | old answerer | robust answerer | gold |
| --- | --- | --- | --- |
| iPhone 16 refresh rate | `128, 4,98.92, 604.99, ...` (dump) | `1280p` (clean, wrong) | 60Hz |
| S24 Ultra weight | `232g ... 8.6mm ... 6.8"` (dump, right value buried) | `8.6mm` (clean, wrong) | 232 g |

**Honest read.** Noise augmentation FIXED the verbose spec-dump failure mode:
answers are now clean and concise. It did NOT fix correctness on raw live pages.
The remaining failure is attribute mis-selection: live pages carry no
(entity, attribute) metadata, so the entity gate falls back to a plain bi-encoder
over coarse text chunks that mix many specs, and the answerer confidently returns
the wrong field. This is a retrieval/segmentation problem on live content, not an
answerer-training problem, so the right fix is per-entity live fetching with
structured extraction (or finer chunking), not more answerer epochs.

**Decision.** We keep `models/answer-base` as the reported model (all frozen-test
numbers use it; byte/energy savings are unaffected). `answer-base-robust` is
retained as evidence that the formatting half of the gap is closable. The
correctness half remains a stated limitation.
