# Phase 1 Lever Plan — scale & balance to push Phase 2 toward its target

**Status date:** 2026-09-12
**Why this exists:** Phase 2's decomposer passed all its *deliverables* (both
models trained/benchmarked, `TaskDecomposer` built, LoRA tuned) but sits at
**44% val exact-match against a >90% target**. Diagnosis: the shortfall is
driven by **task/annotation volume and answer-type imbalance**, not model
capacity (entity_f1 is already ~0.81). 48 seeds → 132 train pairs is thin, and
answer types are badly skewed (comparison 21, single_fact 19, but yes_no 1,
procedure 1, list/multi_part/narrative 2 each).

**Key framing — two independent levers:**
- **Track A — seed-task volume/balance drives Phase 2 accuracy.** The
  decomposer trains on `question → (entity, attribute)` pairs (seed tasks),
  never on passage text. Growing and balancing seeds is what moves 44%.
- **Track B — passage volume drives Phase 1's 200–500 target and Phase 3
  retrieval**, not Phase 2. Separate, later work.

---

## Step 1 — Resolve builder authority (blocker, do first)

`scripts/build_seed_corpus.py` / `build_seed_tasks.py` are the nominal source
of truth but are **stale**: the 6 corpus rows + 7 seeds (T42–T48) added during
the decomposer fix went straight into `data/corpus.jsonl` /
`data/tasks_seed.json`, plus embeddings. Re-running either builder would wipe
that.

**Decision:** `data/corpus.jsonl` and `data/tasks_seed.json` are now the
**canonical source of truth**. The two original builders get a deprecation
header ("historical seed generator — do NOT re-run; canonical data lives in
data/*.jsonl"). New data is added by an **idempotent expansion script**
(`scripts/expand_seed_tasks.py`) that reads the canonical files, appends new
seeds by id, and never clobbers.

## Step 2 — Track A: grow & balance seeds, retrain, CHECKPOINT

**Author ~45 new seed tasks (T49+) entirely from existing corpus (entity,
attribute) pairs** — no new passages, no fabricated specs. All new seeds go to
the **train** split; they must not reuse an (entity, attribute) key already in
a val/test seed (leakage guard). Rough per-type targets (capped by corpus
entity diversity):

| answer_type | now | target | source |
|---|---|---|---|
| single_fact | 19 | ~25 | any safe pair |
| comparison | 21 | ~25 | 2 entities sharing an attribute; balance 1-attr/2-attr |
| yes_no | 1 | ~14 | 2 entities, numeric attribute (price/weight/height/…) |
| multi_part | 2 | ~14 | 1 entity with ≥2 attributes |
| list | 2 | ~8 | amenities / codec_support entities |
| narrative | 2 | ~6 | +4 well-known film `summary` passages (plots are public, not fabricated specs) |
| procedure | 1 | 1 | **left as-is** — templates are tire-hardcoded and its seed is test-reserved; scaling needs generator+schema surgery. Stays a documented gap. |

Mechanics:
- `expand_seed_tasks.py` enumerates safe pairs and emits deterministic base
  questions per type, dedups against existing questions, assigns T49+ ids.
- Add the 4 narrative film passages to `corpus.jsonl` (and to the expansion
  script's data, so it's reproducible).
- Assign all new seeds to `train` in `make_splits.py`'s `SEED_SPLIT_ASSIGNMENT`.
- **Raise the task cap** in `tests/test_data.py` (`test_task_count_in_target_range`)
  and update the stale "41 seeds/164 tasks" comment in
  `generate_synthetic_tasks.py` — the recommended sequence explicitly sanctions
  raising the 150–200 cap.
- Regenerate: `generate_synthetic_tasks.py` → `make_splits.py` →
  `build_decomposer_training_data.py` (coverage invariant must still pass).
- Embed any new passages: `embed_corpus.py`.
- Retrain both models (small full-FT; base+LoRA with the tuned config
  r=32/alpha=64/lr=5e-4/q,v,k,o, 20 epochs).
- **CHECKPOINT:** eval both; report before/after. Decide from the trajectory
  whether >90% is reachable with more data, needs constrained decoding, or the
  target/metric should be revised. **Do not assume 90% is reachable.**

## Step 3 — Track B: corpus to 200–500 (Phase 3 readiness) — PAUSE FIRST

Scoped but **not executed without a go-ahead**, because it needs a decision +
network/licensing: adapt a subset of MS MARCO / Natural Questions passages into
the `Passage` schema (plan's own suggestion) for bulk, plus targeted on-domain
passages. This serves Phase 3 recall@5, not Phase 2, so it waits until after
the Step-2 checkpoint.

---

## Guardrails (do not violate)
- No new fabricated product specs presented as sourced facts; new seeds reuse
  existing corpus pairs. Only film-plot `summary` passages are added, and plots
  are public knowledge.
- Test seeds (T12, T15, T23, T29, T35, T40) stay frozen; new seeds are
  train-only and never reuse a val/test (entity, attribute) key.
- `check_no_llm_api` / `test_no_llm_api` stay green; everything local-only.
- The train-coverage invariant (`assert_train_coverage`) must pass after regen.
- `python -m pytest -q` green throughout.

## Verification
- Coverage: every answer_type (except procedure) and attribute slug has train>0.
- No val-only comparison phrasing (parity check).
- Before/after table for both models: strict EM, EM-normalized, entity_f1,
  attribute_f1, entity_exact_match.
- Full pytest green.
