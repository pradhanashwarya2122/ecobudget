# Teammate 1 Plan — ML + Retrieval (EcoBudget)

**Status date:** 2026-09-09
**Scope:** Task decomposition, requirement-aware retrieval, evidence tracking, adaptive/bandit policy, answer generation, and the ML side of benchmarking.
**Constraint:** No LLM API calls anywhere in this workstream. `teammate-1-ml-retriever/` was scrapped because it depended on an LLM API — every component below must run on local models only (flan-t5, sentence-transformers, spaCy, RoBERTa QA, scikit-learn / xgboost / a small PyTorch policy). This rules out using Claude or any hosted model as a hidden dependency for decomposition, retrieval, or answering.

This plan starts clean. Nothing from the scrapped folder is assumed to exist. Where the `backend/` prototype (documented in `codebase-analysis.md`) already has a usable piece, it's noted as **reuse/adapt**; everything else is **build new**. New work lives in a fresh directory, e.g. `ml_retriever/`, kept separate from `backend/` until it's ready to replace pieces of `conditions.py`.

**Review addendum (2026-09-09):** an external review of this plan came back "ready to execute, no blocking issues," with eight non-blocking suggestions and two minor points. All are incorporated below (marked *[review]*), concentrated in Phase 1 (dataset size/sourcing), Phase 2 (LoRA), Phase 5 (action space, judge model, features, logging), Phase 6 (generative answer model, abstain logic, Teammate 2 interface), and the cross-cutting notes (logging, CI scope). The reviewer's three "most critical" items — dataset size, a generative answer model for comparisons, and a clearly defined judge — are each called out explicitly where they land.

**Second review pass (2026-09-09):** a follow-up review examined risk/mitigation for five items and one execution-order clarification, marked *[review-2]* below: generative-answer-model training cost (Phase 6), per-requirement action-space complexity (Phase 5), reward λ-tuning (Phase 5), data leakage across the train/val/test split (Phase 1/2/3/5), and the `backend/`↔`ml_retriever/` integration path (Phase 6/7 + cross-cutting). It also refined the execution order around Phase 3/4 sequencing.

---

## Phase 0 — Environment, Data Structures, Test Harness

**Goal:** A clean, testable foundation before any ML work starts.

- [ ] `ml_retriever/pyproject.toml` (or `requirements.txt`) — pin: sentence-transformers, transformers, torch, spacy (+`en_core_web_sm`), scikit-learn, xgboost, joblib, numpy, pytest. No API-client packages (openai, anthropic, etc.) — enforce this as a lint/CI check, not just a convention.
- [ ] `pytest` set up with a `tests/` directory mirroring module structure.
- [ ] Dataclasses in `ml_retriever/types.py`:
  - `Requirement(entity: str, attribute: str, value: Optional[str] = None)`
  - `Passage(passage_id: str, text: str, byte_size: int, embedding: Optional[np.ndarray], source_url: str, ...)`
- [ ] `EvidenceTracker` interface (Protocol/ABC) — must be satisfiable by both a rewritten `EvidenceCoverageTracker` and any future implementation. Contract: `is_sufficient`, `frac_satisfied`, `frac_remaining`, `coverage`, `unanswered_requirements()`.
- [ ] `AnswerGenerator` interface — `generate(question, evidence: list[Passage]) -> {answer, confidence, used_evidence_ids, abstained, generator_version}`.
- [ ] Unit tests for all of the above using dummy/fixture data (no real models needed yet).

**Reuse/adapt:** None directly reusable — `backend/` has no dataclasses, interfaces, or tests. `usefulness.py`'s `FEATURE_NAMES` ordering convention is worth carrying over as a pattern.

**Validation:** `pytest` green with 0 real-model dependencies in this phase.

---

## Phase 1 — Corpus & Task Dataset

**Goal:** Real data foundation — replace the 20-task Wikipedia benchmark as the primary dataset (it remains a legacy baseline, not the target corpus).

- [ ] Collect 200–500 passages from non-Wikipedia, decision-relevant sources (specs pages, gov/university sources, product pages) matching the project's new focus: comparisons, multi-attribute research, product selection, travel, specification lookup.
  - *[review, most-critical]* 200–500 may be small for robust retrieval. If sourcing enough real passages is too time-consuming, adapt a subset of an existing open-source passage dataset (e.g. MS MARCO or Natural Questions passages) into the `Passage` schema instead of collecting everything by hand — this also comes with pre-computed relevance labels usable for Phase 3's recall@5 eval. Treat this as a time-saving option, not a requirement to hit 500 manually.
- [ ] Store as `corpus.jsonl` matching the `Passage` dataclass fields.
- [ ] Pre-compute `all-MiniLM-L6-v2` embeddings per passage; persist alongside (e.g. `.npy` or embedded in the jsonl as base64/list).
- [ ] Build `tasks.json`: **150–200 tasks** (raised from 50–100) — *[review, most-critical]* a contextual bandit needs many episodes to learn a reliable policy, especially as the action space grows in Phase 5. To hit this without proportionally more manual annotation, generate synthetic variations of manually-written seed tasks (paraphrases, swapped entities/attributes) and mark them as `synthetic: true` in `tasks.json` so Phase 2/5 evaluation can separate synthetic from hand-written performance if needed. Each task: `question`, manually-annotated `decomposed_requirements` (list of `{entity, attribute}`), `expected_answer`/`required_facts`, optional `acceptable_passage_ids`, `topic`, `difficulty`, `answer_type`.
- [ ] Train/val/test split (for Phase 2 decomposer training and Phase 5 bandit training/eval).
  - *[review-2]* Freeze this split immediately after Phase 1 and do not touch the test set again until Phase 7 — Phases 2, 3, and 5 all draw on the same corpus/task pool, which is exactly the setup where test data leaks into development without anyone intending it. Concretely: Phase 2 decomposer training uses only the train split; Phase 3's recall@5 evaluation uses a separate held-out set of requirement→passage pairs (not the held-out *tasks* meant for Phase 5/7); Phase 5 bandit training/eval uses train/val only, never test.
- [ ] Validation: every task has ≥1 passage per requirement in the corpus (scripted check); manually review 10 random tasks for clarity.

**Reuse/adapt:** `backend/pages/*.html` + `benchmark_tasks_v1_FROZEN.json` stay as a **legacy baseline only** — useful for regression comparison against Version 1, not as the Phase 1 deliverable. `scorer.parse_resources` can be repurposed as one *source* of passage extraction if any HTML pages are included in the new corpus, but the corpus itself must be broader than Wikipedia per the project overview.

---

## Phase 2 — Task Decomposition Model

**Goal:** Replace manual/regex decomposition with a trained local model — this directly addresses "Problem discovered #4" (regex too primitive) from the project history.

- [ ] Build a synthetic + manually-labeled training set from Phase 1 tasks + templates (question → `{entity, attribute}` list).
- [ ] Fine-tune `flan-t5-small` and `flan-t5-base` (compare both — required deliverable, not optional).
  - *[review]* For `flan-t5-base`, prefer parameter-efficient fine-tuning (LoRA) over full fine-tuning to keep memory/time reasonable on a laptop-class machine. If LoRA is used, record its extra memory/time overhead alongside the small-vs-base comparison so the Phase 2 benchmarking numbers stay reproducible and comparable (full fine-tune vs LoRA isn't an apples-to-apples compute comparison otherwise).
- [ ] Implement `TaskDecomposer` class: `question -> list[Requirement]`.
- [ ] Evaluate on held-out set: exact-match accuracy on entity/attribute pairs, F1.
- [ ] Record inference time and CPU/GPU memory for both model sizes.
- [ ] Target: >90% decomposition accuracy (entity + attribute both correct), runs locally within reasonable time.

**Reuse/adapt:** `evidence.py`'s subject/attribute extraction via flan-t5-base (`extract_requirements`) is a working starting point for prompt design and requirement-set construction, but it's heuristic and unevaluated — it needs to become the trained, benchmarked `TaskDecomposer`, not stay ad hoc. `semantic_engine.py`'s `decompose_task` (unwired, uses a different `requirement`/`answer_type` key schema) should be reviewed for ideas but not carried forward as-is — schema must match the Phase 0 `Requirement` dataclass.

---

## Phase 3 — Requirement-Aware Retrieval

**Goal:** Retrieve passages per-requirement, not per-whole-question.

- [ ] Implement `RequirementRetriever`: `Requirement + corpus -> top-k Passages by similarity`, with a cross-encoder re-rank option.
- [ ] Value-per-byte score: `similarity / log(byte_size)` — note this is a **different formula** from the existing `scorer.rank_by_vpb`'s `utility / bytes`; don't reuse that formula, only the general "per-byte" concept.
- [ ] Update evidence-adding logic to call this per-requirement, replacing whole-question ranking.
- [ ] Validation: recall@5 on manually labeled requirement→passage pairs; compare directly against old whole-question ranking (`scorer.rank_by_vpb`) and expect improvement — this comparison is a concrete, checkable deliverable.

**Reuse/adapt:** `scorer.compute_utility` (MiniLM cosine similarity) is reusable as the similarity primitive. `scorer.rank_by_vpb` itself is the explicit baseline to beat, not a component to extend — its whole-question design is exactly what Phase 3 replaces.

---

## Phase 4 — Evidence Tracking with Requirements

**Goal:** Track which requirements are sufficiently evidenced, using only retrieved (not ground-truth) passages.

- [ ] Enhance `EvidenceCoverageTracker` to accept a `Requirement` set + selected `Passage`s.
- [ ] Per-requirement sufficiency via similarity/QA-confidence signals.
- [ ] Expose `unanswered_requirements()`.
- [ ] Unit tests: synthetic passages, correct status transitions; a passage with entity-only (no attribute) must NOT satisfy a requirement.

**Reuse/adapt — this is the most mature existing piece.** `backend/evidence.py`'s `EvidenceCoverageTracker` already tracks requirements, bytes/passage counters, coverage, and answers, and already satisfies the `is_sufficient`/`frac_satisfied`/`frac_remaining`/`coverage` contract that Phase 0's `EvidenceTracker` interface and `controller.py` expect. Work here is mostly: (a) swap its input from whole-question-ranked candidates to Phase 3's per-requirement retrieval output, (b) add the missing `unanswered_requirements()` method, (c) port it into the new dataclass-based interface, (d) add the entity-without-attribute negative test explicitly (not clearly covered today).

---

## Phase 5 — Contextual Bandit Policy

**Goal:** Learn when to retrieve more vs. stop. This is the project's central research contribution — explicitly must not be faked (per project overview: "don't add a bandit just for the label").

- [ ] Action space v1: `STOP` or `RETRIEVE_TOP1` (highest-priority unanswered requirement).
  - *[review]* Consider going further even in v1: one `RETRIEVE_{req_id}` action per unanswered requirement (plus `STOP`), capped at a max of 4–6 requirements per task to keep the action space tractable. This lets the policy learn to prioritize which attribute to fill in next, which is the eventual goal anyway and is a natural fit given Phase 3's requirement-aware retrieval already returns per-requirement candidates. If this proves too complex to get working first, fall back to `STOP`/`RETRIEVE_TOP1` for v1 but explicitly plan the extension as a **hierarchical policy** (first choose requirement, then passage) rather than a full redesign later.
  - *[review-2]* Concretely, treat `RETRIEVE_{req_id}` actions as a v2 step, not a v1 requirement: per-requirement actions increase the action dimension, and if the number of requirements varies across tasks the policy needs a variable-sized action mask, which is its own implementation problem on top of the bandit itself. Get the `STOP`/`RETRIEVE_TOP1` loop working end-to-end first (this is what unblocks Phase 6/7), then introduce per-requirement actions once that basic loop is stable and validated.
- [ ] Context vector: per-requirement status/confidence/passages-retrieved, cumulative bytes used, candidate similarity stats, query embedding (or reduced form).
  - *[review]* Add a remaining-budget feature: either literal remaining bytes (when a fixed-budget condition applies) or a normalized `data_used / average_task_data` figure, to help the policy generalize across tasks of different natural size.
  - *[review]* Normalize all features (e.g. z-scores fit on the training set) before feeding the linear model — raw byte-count features will otherwise dominate by variance alone.
- [ ] **Judge model** — *[review, most-critical, clarifies previously-vague step]* used to compute `success` inside the reward function:
  - v1 (default): rule-based judge — check whether the generated answer contains the required attribute value(s) via string/regex matching. Sufficient as a first pass, especially if answers are kept structured (JSON-like) rather than free text, and avoids an extra training step before the bandit can run at all.
  - v2 (only if v1 proves too brittle on free-text answers): fine-tune a small NLI model (e.g. `deberta-v3-xsmall`) on labelled `(question, evidence, correct_answer, incorrect_answer)` pairs built from Phase 1 tasks, producing a binary correctness label.
  - Whichever is used: the judge must be **identical across every experimental condition** (Normal, Fixed-*, heuristic, bandit) for the comparison in Phase 7 to be valid, and it must never be exposed to the policy during training — only the scalar reward derived from it is passed to `update`.
- [ ] Reward: `success - λ * incremental_payload_bytes / max_bytes`, success from the judge above; λ tunable.
  - *[review-2]* A single fixed λ may not generalize across tasks with very different natural data sizes (a comparison task legitimately needs more bytes than a single-fact lookup). During offline training, sweep several λ values and plot the resulting success-vs-bytes trade-off curves per task category, then pick the λ that matches the project's stated goal (comparable success, meaningful reduction) rather than picking one value up front. If a single λ doesn't hold up across task types, consider a non-linear penalty instead — e.g. only penalize bytes above some per-task-category threshold — as a more robust alternative to a flat linear term.
- [ ] `BanditPolicy` class: `select_action`, `update` (start with `SGDClassifier(loss='log_loss')` linear contextual bandit; later compare a small PyTorch policy).
- [ ] Training loop: for each training task, init tracker, repeatedly query policy until STOP, retrieve/update evidence on RETRIEVE, call `AnswerGenerator` at STOP, compute reward, update online.
  - *[review]* Log the context vector, chosen action (and action probabilities if available), and reward for every step — not just final per-task metrics. This is what makes bandit trajectories debuggable and comparable across policy versions; retrofitting it later means losing earlier runs' step-level data. See the logging note in Cross-cutting notes.
- [ ] Save trained policy; comparative results vs. fixed budgets and the existing heuristic `EcoBudgetController`.
- [ ] Validation: bandit ≈ normal-retrieval task success with significantly less data than fixed budgets; decisions are interpretable (more retrieval for complex tasks, early stop for simple ones).

**Reuse/adapt:** `backend/controller.py` (`EcoBudgetController`) + `usefulness.py` (`UsefulnessModel`) is a **heuristic/supervised usefulness scorer, not a bandit** — no action space, no online reward loop, no `select_action`/`update`. Keep it only as the "simple adaptive heuristic" baseline the bandit must beat (this maps directly to Baseline 3 in the project overview's condition list). Do not extend it into the bandit; build the bandit as a new, separate policy class per the interface above.

**Build new — no shortcuts:** nothing in `backend/` or elsewhere should be treated as "close enough" here. This is flagged specifically because it's the easiest phase to fake.

**Phase 5/6 sequencing clarification (2026-09-13, discovered during implementation):**
The bandit's reward needs an `AnswerGenerator` at STOP, and `success` comes from the v1 judge. The v1 judge only gives a valid, retrieval-dependent signal when task gold is **structured `required_facts`**, not free-text verdict strings — with a concat-evidence answer generator and verdict-string gold, even an oracle that retrieves everything scored ~0 on comparisons, making the reward degenerate. Resolution adopted:
- **Data realignment (done):** `scripts/build_structured_gold.py` sets `ground_truth.required_facts` (grounded in the corpus via QA extraction) for comparison/yes_no/multi_part/list tasks; `expected_answer` is kept display-only, never the success criterion. This lifted oracle success on comparisons from 0.00 to 0.95. `narrative` stays free-text and is a known residual pending Phase 6.
- **Generator/reward decoupling:** the Phase 6 generative `AnswerGenerator` is built separately and evaluated with its own metric (ROUGE / manual) — it must **not** change the reward used for bandit training, to avoid a moving target. The bandit trains against the structured-gold judge regardless of generator quality.
- **Do not tune λ/exploration until the reward is valid.** A λ sweep (`scripts/sweep_lambda.py`) is the mechanism for picking the success-vs-bytes operating point once the reward is valid.

**Phase 5 result (v1):**
- *Terminal (Monte-Carlo) credit — negative result:* crediting every step the episode objective (success − λ·bytes/max) made the bandit reach ceiling success (0.778, above naive `normal`'s 0.444) but over-retrieve (λ=0.5→741 bytes, λ=1→384) and collapse to always-STOP at λ≥2 — a knife-edge with no stable operating point that beats the baselines.
- *Per-step credit — resolved:* giving each RETRIEVE its marginal coverage gain minus its byte cost (redundant retrievals go negative) and STOP the realized success localizes the decision. The bandit now reaches ceiling success **0.778 at ~277 bytes on val — below the heuristic (294), fixed@2 (281), and fixed@4 (516)** — meeting Phase 5's goal (comparable success, less data than fixed budgets), and the λ landscape is a stable plateau (λ∈[1,4]) rather than a cliff. Confirms the earlier shortfall was credit assignment, not the bandit itself. `scripts/sweep_lambda.py --reward_mode {terminal,per_step}` reproduces both. Operating point: per-step, λ=4.
- *Remaining caveat:* ceiling success is 0.778 (not ~1.0) because of the placeholder concat-evidence answer generator + `narrative` free-text gold; Phase 6's generative answerer lifts the ceiling but must stay decoupled from the bandit reward.

---

## Phase 6 — AnswerGenerator & Full Pipeline Integration

**Goal:** Connect decomposition → retrieval → evidence → policy → answer into `EcoBudgetSystem`.

- [ ] `AnswerGenerator`: wrap RoBERTa QA for single-fact tasks; template-based filler for comparison tasks. Must emit `{answer, confidence, used_evidence_ids, abstained, generator_version}` per the Phase 0 interface — note the existing QA path doesn't expose `abstained` or `used_evidence_ids` today, so this is new wrapper logic, not a pass-through.
  - *[review, most-critical]* Templates are brittle for comparison tasks — they can dump raw values ("iPhone 15: $799, 3349 mAh, 60 Hz") without synthesizing a judgment like "better battery life," which is what the project's realistic multi-attribute tasks actually need. Prefer fine-tuning a small local generative model (`flan-t5-base` or `bart-base`) on Phase 1's task types to produce final answers directly from retrieved evidence, replacing both the extractive-QA and template-filler paths for comparison tasks. Keep RoBERTa extractive QA as the path for single-fact tasks if it's already working well there. Whichever generator is used, it **must be frozen during bandit training and evaluation** (the bandit should never be learning against a moving-target answerer), and its `generator_version` must be logged on every call so Phase 7 results are attributable to a specific frozen checkpoint.
  - *[review-2]* This adds a third fine-tuning burden on top of the decomposer (Phase 2) and the bandit (Phase 5) — don't default straight to `flan-t5-base`. Start with `flan-t5-small`; it may well be sufficient for structured, short-form comparison answers, and only move up to `base` if quality proves inadequate on held-out tasks. Use LoRA for both sizes (same rationale as Phase 2's decomposer) to keep memory/time manageable rather than full fine-tuning either one.
  - Define **abstain logic explicitly** before wiring this into the pipeline: e.g. abstain when confidence falls below a set threshold, or when no evidence passage covers a given requirement. This isn't cosmetic — it directly feeds the Phase 5 reward function (an abstained answer should read as an unsuccessful/`success=0` episode via the judge, not silently as "wrong answer").
- [ ] `EcoBudgetSystem` class: `question -> decompose -> init requirements -> retrieve via policy -> generate answer -> measure bytes`.
  - *[review-2]* Define this class's public API cleanly within `ml_retriever/` first. When it's time to run end-to-end experiments through `experiment_runner.py`, integrate via a **thin adapter in `backend/`** that imports and calls `EcoBudgetSystem`, rather than merging the two codebases or reaching into `ml_retriever/` internals from `conditions.py`. This keeps `ml_retriever/` decoupled and independently testable while still allowing Phase 7's full-condition experiment run to include the bandit.
- [ ] Integrate with Teammate 2's fetching, offline-first using pre-downloaded pages.
  - *[review]* Agree on and implement a single, explicit fetch contract rather than an ad hoc integration: `fetch_passages_for_requirement(requirement: Requirement) -> list[Passage]`, owned by Teammate 2's web pipeline but called from this side. This decouples ML work from the network/live-fetch layer — offline (pre-downloaded pages) and live fetching become interchangeable implementations behind the same signature.
- [ ] Integration tests on a handful of benchmark tasks end-to-end.
- [ ] Validation: 10-task end-to-end run, correct answers with low payload; every component logs outputs for reproducibility.

**Reuse/adapt:** `scorer.check_answerability` (RoBERTa QA) is the extractive-QA primitive to wrap — but treat its output as unreliable-when-confident per the documented failure cases (T9 "1 July 2016" instead of Colosseum completion year, T13 "Transpiration," T5 "Phillip Ratner"). The `AnswerGenerator` wrapper needs its own abstention logic on top, not just a confidence passthrough. `db.py`'s SQLite logging pattern is reusable for the "every component logs outputs" requirement, though its schema will need to grow to cover decomposition/policy-decision logs, not just resource-load rows.

---

## Phase 7 — Benchmarking, Analysis, Documentation

**Goal:** Research-ready final results.

- [ ] Full run on the frozen (new, Phase 1) test set across: Normal, Fixed 10/25/50/100 KB, heuristic adaptive, bandit adaptive.
- [ ] Metrics: task success (judge), selected payload bytes, sunk page-acquisition bytes (if live), end-to-end transfer bytes, latency, retrieval step count.
- [ ] Statistical analysis: paired tests, bootstrap.
- [ ] Document model choices, compute resources, failure cases.
- [ ] Update README and `codebase-analysis.md`.
- [ ] Validation: reproducible from frozen benchmark + saved models; results support (or honestly don't support) the claim that the learned policy reduces bytes while maintaining success.

**Reuse/adapt:** `backend/experiment_runner.py` already runs Normal + 4 fixed budgets + EcoBudget and writes a CSV — structurally the right shape for this phase. Two concrete fixes needed before it can serve Phase 7: (1) it currently calls `check_task_success` (string-only ground truth) — must switch to `check_task_success_v2` (fact-set) since the new benchmark will use structured ground truth; (2) it needs a bandit condition added alongside the existing five. The current 11-column CSV schema is a reasonable base to extend with the Phase 7 metric list above (latency, retrieval steps, etc. aren't currently columns).

---

## Cross-cutting notes

- **No LLM API dependency, anywhere.** Every phase above uses local models only (flan-t5 fine-tuned locally, sentence-transformers, spaCy, RoBERTa QA, scikit-learn/xgboost/PyTorch for the bandit). This was the reason `teammate-1-ml-retriever/` was scrapped — enforced today by `scripts/check_no_llm_api.py` (banned imports + known API hosts). *[review]* A further check for `requests`/`httpx` calls to arbitrary non-local URLs was suggested too, but flagged by the reviewer as possibly too strict (would also catch Teammate 2's legitimate live-fetch code and Phase 1 corpus-sourcing scripts) — treat as optional/deferred rather than adding it to the required check now.
- **Adopt structured run logging from the start, not retroactively.** *[review]* Rather than only logging final metrics per task/condition (current Phase 6/7 framing), log at the step level too: for the bandit, that means context vector, chosen action, and reward per step (see Phase 5); for the pipeline generally, decomposition output, retrieved passages per requirement, and generator calls. A plain structured CSV/JSON log per run is enough to start — an experiment-tracking tool (e.g. Weights & Biases) is a reasonable upgrade later if comparing many policy runs becomes unwieldy, but isn't required to begin.
- **Don't optimize against the old 20-task Wikipedia benchmark** — it's retained only as a regression/legacy baseline (per project overview, Section 15).
- **Freeze the Phase 1 benchmark before Phase 7's final run** — no task changes after seeing results.
- **Evidence tracking and controller must never see ground truth** during retrieval/stopping decisions — only at evaluation time. The Phase 5 judge model follows the same rule: it computes reward, but is never itself exposed to the policy during training.
- **Phase 4 and parts of Phase 7 have real reusable code today; Phase 3, 5, and most of Phase 6 do not** — sequence work accordingly rather than assuming uniform effort across phases.

---

## Suggested execution order given current state

1. Phase 0 (blocks everything).
2. Phase 1 (blocks 2, 3, 5, 7) — freeze the train/val/test split as the last step of this phase and don't revisit it until Phase 7.
3. Phase 4 port/adapt (fast — reuses existing `evidence.py` logic). *[review-2]* Get this stable *before* fully wiring in Phase 3's per-requirement retrieval output — the tracker expects a stream of selected passages, so having its interface locked first makes testing the new retriever's output much simpler. In practice this means: start Phase 4 in parallel with Phase 3's embedding/ranking implementation work, but treat Phase 4's interface as frozen before connecting Phase 3's retriever to it.
4. Phase 2, then Phase 3 (3 depends on corpus + wants decomposed requirements to retrieve against).
5. Phase 5 (depends on 3 + 4 working together) — hardest phase, budget the most time here. Start with the simple `STOP`/`RETRIEVE_TOP1` action space; treat per-requirement actions as a v2 extension once the loop is validated.
6. Phase 6 integration — build `EcoBudgetSystem` in `ml_retriever/` first, add the `backend/` adapter last.
7. Phase 7 benchmarking and writeup.
