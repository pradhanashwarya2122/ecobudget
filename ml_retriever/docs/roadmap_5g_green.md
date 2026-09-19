# EcoBudget → 5G-Green Conference Paper: Implementation Roadmap

**Purpose.** Turn the current data-efficient-QA prototype into a defensible
"green / energy-aware retrieval for 5G edge" research contribution. Phases are
ordered by *how badly the paper needs them*. Phases A–B are **existential** for
any "green" claim; C–E make it a real paper; F–H make it a strong one.

**How to read a phase.** Each has: **Goal**, **Why (the reviewer question it
answers)**, **Tasks**, **Files**, **Acceptance gate** (the number/artifact that
must exist before moving on). Run phases in order; do not skip a gate.

**Standing rules (carried from prior work).** Local models only (no cloud LLM).
Frozen test split — one final run at the very end. Pre-register before each
experiment. Report `judge_success` and `fact_f1` separately, with bootstrap CIs.
Never tune a component after seeing the final-test numbers.

---

## Phase A — Compute-vs-transfer energy accounting (EXISTENTIAL)

**Goal.** Measure the energy the pipeline *spends running models per query* and
compare it to the transfer energy it *saves* by loading fewer bytes. Report
**net** energy.

**Why.** The single most damaging reviewer question: *"Does the energy saved by
fewer bytes exceed the energy of running flan-t5 + RoBERTa-QA + MiniLM per
query?"* Today this is unmeasured. If compute energy dominates, the green claim
is false. This must be answered before anything else.

**Tasks.**
1. Instrument per-query compute: count model forward passes (decompose,
   embed, QA-score per (req,passage), answer-generate) and their input/output
   token counts.
2. Attach an energy estimate per inference: use published joules-per-token /
   joules-per-inference figures for the specific models on a target device
   class (edge CPU / mobile SoC), or measure wall-clock × a device TDP proxy.
   Cite the source; make the coefficient a documented constant.
3. Attach a transfer-energy estimate per byte (Phase B refines this).
4. New module `ml_retriever/energy.py`: `energy_joules(compute_ops, bytes) ->
   {compute_j, transfer_j, total_j}` and a per-query logger.
5. Re-run Phase 7 conditions reporting **compute_j, transfer_j, total_j** per
   condition, not just bytes.

**Files.** `ml_retriever/energy.py` (new), `scripts/phase7_experiment.py`
(add energy columns), `backend/carbon.py` (reuse/retire).

**Acceptance gate.** A table showing, per condition, transfer joules saved vs
compute joules spent, and whether the bandit is **net-positive**. If it is not,
the paper's framing must change (e.g. "compute-bound regime") — state it
honestly.

---

## Phase B — A defensible energy model + realistic payloads (EXISTENTIAL)

**Goal.** Replace the single web-CO2 coefficient with a 5G-grounded energy model
applied to **realistic payload sizes**, not 200-byte snippets.

**Why.** Byte savings on 200-byte passages are trivially small and unconvincing.
5G energy is dominated by the radio access network (RAN). A per-bit model split
into access + transport (+ device modem) applied to KB–MB payloads is what makes
"bytes saved → joules saved → gCO2e" credible to a networking reviewer.

**Tasks.**
1. Literature-based per-bit energy for the 5G path: RAN energy/bit, transport
   energy/bit, device modem energy/bit. Document sources and assumptions.
2. Grid carbon-intensity factor (gCO2e/kWh) with a cited source; keep
   green-vs-grid host variants.
3. Realistic payloads: attach real page/asset byte sizes to passages (e.g. the
   HTML page a passage came from), so "loading a passage" costs a realistic
   transfer, not 200 B. Store `page_bytes` alongside `text` in the corpus.
4. Wire the model into `energy.py` (Phase A) so transfer_j uses the 5G per-bit
   model over `page_bytes`.

**Files.** `ml_retriever/energy.py`, `data/corpus.jsonl` (add `page_bytes`),
`scripts/build_seed_corpus.py` / a new sizing script, `docs/energy_model.md`.

**Acceptance gate.** `docs/energy_model.md` with every coefficient, unit, and
citation; Phase 7 re-run showing gCO2e per condition at realistic scale.

---

## Phase C — Dataset scale-up (needed for statistics)

**Goal.** Move from 92 passages / 30 val tasks to a scale where results are
statistically meaningful.

**Why.** n=28 headline tasks ⇒ wide CIs; the success difference vs heuristic is
not statistically distinguishable. Reviewers dismiss toy-scale evidence.

**Tasks.**
1. Adopt an open-domain passage dataset (MS MARCO / Natural Questions passages)
   adapted into the `Passage` schema, keeping the `(entity, attribute)` labels
   where possible or defining a comparable relevance label.
2. Grow tasks to hundreds (target ≥300 with ≥100 comparisons), keeping the
   train/val/test freeze discipline.
3. Re-embed corpus; re-cut splits with the coverage invariant.
4. Re-train decomposer / answerer / bandit on the larger data (multi-seed, see
   Phase G).

**Files.** `scripts/build_seed_corpus.py`, `scripts/*` regeneration chain,
`data/*`.

**Acceptance gate.** ≥300 tasks, ≥100 comparison val/test tasks; Phase 7 CIs
tighten enough that at least one bandit-vs-baseline claim is significant.

---

## Phase D — Fix the retriever (entity-aware retrieval)

**Goal.** Stop the entity-name from dominating the query so the correct passage
is actually retrieved.

**Why.** The known entity-dominance bug caps absolute accuracy (~0.68–0.82) and
made `list` tasks fail (correct passage not in top-5). "Our accuracy is limited
by a retriever bug" is a weak story.

**Tasks.**
1. Entity-aware retrieval: restrict/boost candidates to passages whose metadata
   (or text) matches the requirement's entity, then rank by attribute
   similarity. Offline uses entity metadata; live fetch fetches per-entity.
2. Re-evaluate recall@1/5 vs the current bi-encoder; confirm the amenities /
   list failure is fixed.
3. Re-run the oracle-vs-pipeline gap; the pipeline's retrieval loss should
   shrink.

**Files.** `ml_retriever/retriever.py`, `scripts/eval_retriever.py`.

**Acceptance gate.** recall@1 improves materially; `list` task retrieves its
correct passage; documented in `eval_notes.md`.

---

## Phase E — External and standard baselines

**Goal.** Compare against methods a reviewer expects, not just internal
conditions.

**Why.** "Why a linear SGD bandit and not LinUCB/Thompson? Why not published
adaptive-RAG?" must be answered with numbers.

**Tasks.**
1. Standard contextual-bandit baselines: LinUCB and Thompson sampling, same
   features/reward, same eval.
2. At least one published adaptive-retrieval / budget-aware RAG baseline (or a
   faithful reimplementation) as an external comparison.
3. Add these as Phase 7 conditions with the same metrics + CIs.

**Files.** `ml_retriever/bandit.py` (add algorithms), `scripts/phase7_experiment.py`.

**Acceptance gate.** Phase 7 table includes ≥2 standard bandit baselines and ≥1
external method; positioning of "our bandit" is defensible.

---

## Phase F — Radio-state / latency-energy modeling (5G novelty)

**Goal.** Make "fewer/earlier retrievals" a genuine 5G energy lever, not just a
byte count.

**Why.** 5G energy is dominated by *keeping the radio active* (RRC states), not
only bytes moved. Modeling that turns adaptive stopping into a real radio-energy
result and is the paper's novelty hook for a networking venue.

**Tasks.**
1. Add an RRC-state / tail-energy model: each retrieval keeps the modem active;
   stopping earlier lets it return to idle sooner. Parameterize with cited
   timers/energies.
2. Report radio active-time and tail energy per condition; show the bandit
   reduces active radio time.
3. Latency: report end-to-end latency with the compute/transfer split; discuss
   the latency–energy tradeoff.

**Files.** `ml_retriever/energy.py`, `scripts/phase7_experiment.py`,
`docs/energy_model.md`.

**Acceptance gate.** A radio-active-time / tail-energy result showing adaptive
stopping's advantage beyond raw bytes.

---

## Phase G — Rigor, reproducibility, final test run

**Goal.** Make every headline number reproducible and defensible.

**Tasks.**
1. Multi-seed training (≥5 seeds) for the bandit and answerer; report mean ± std
   / CI across seeds.
2. Ablations: reward (terminal vs per-step), λ sweep, joint vs per-requirement
   answerer, snap on/off — as a table (most already run; consolidate).
3. Fix all configs (pre-registration), then the **single final test-split run**.
4. Reproducibility appendix: commands, seeds, versions, model checkpoints.

**Files.** `scripts/*`, `docs/phase7_preregistration.md`, `docs/reproducibility.md`.

**Acceptance gate.** Multi-seed variance reported; test-split run done once;
reproducibility doc complete.

**STATUS: DONE.** Multi-seed variance reported (policy, 5 seeds); ablations
consolidated (retriever, snap, lambda, reward mode, answer mode) in
`docs/phase_g_consolidation.md`; single frozen test-split run completed once;
reproducibility appendix at `docs/reproducibility.md`. Answerer multi-seed was
not run (about 3.5 h per fine-tune on our laptop makes 5 seeds prohibitive);
stated as a resource limitation. Ablation scripts: `scripts/ablate_reward.py`,
`scripts/sweep_lambda.py`, `scripts/multiseed.py`, `scripts/eval_decomposer.py
--no_attr_snap`.

---

## Phase H — Paper assembly

**Goal.** The writeup.

**Tasks.**
1. Related work + explicit novelty (online per-step-reward contextual bandit for
   retrieval stopping tied to a 5G energy budget; radio-state-aware saving).
2. Limitations & threats to validity (retriever ceiling, dataset scale, energy
   model assumptions, compute-vs-transfer regime).
3. Figures: the Pareto success-vs-energy plot (bandit sweet spot), architecture
   diagram, per-type breakdown, ablations.
4. Results narrative: honest Pareto framing (dominates fixed budgets; matches
   heuristic at fewer bytes/joules; net-positive after compute energy — or the
   honest regime statement if not).

**Files.** paper repo / `docs/paper_outline.md`.

**Acceptance gate.** Full draft with every claim backed by a table/figure and a
stated CI; limitations section written before results are finalized.

---

## The one-line honest status
Core adaptive-stopping contribution: **solid**. Green/5G framing: **not yet
earned** — Phases A and B (compute-vs-transfer net energy + a real 5G per-bit
model on realistic payloads) are the gate between "efficient-QA prototype" and
"5G-green paper." Do A and B first; if the bandit is not net-positive on energy,
change the framing honestly rather than hide it.
