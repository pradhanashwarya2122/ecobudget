# Phase 1 data — corpus & task dataset

## What's here

| File | Contents |
|---|---|
| `corpus.jsonl` | 86 real, sourced passages across **22 entities in 10 topics**, one JSON object per line matching the `Passage` schema. |
| `tasks_seed.json` | 41 hand-written seed tasks spanning all seven answer types across the same 10 topics. |
| `tasks.json` | Seed tasks + 123 template-generated paraphrase variants = **164 tasks total** (target range: 150–200). Every variant carries `synthetic: true` and `seed_task_id`, and is guaranteed identical `decomposed_requirements`/ground truth to its seed (enforced by `tests/test_data.py`). |
| `splits.json` + `splits.json.FROZEN` | Re-frozen train/val/test split (104/36/24 tasks), **grouped by seed task** so paraphrases of the same seed never cross splits. Do not regenerate until Phase 7. |

## Topic coverage (fixed 2026-09-09 — see below)

An earlier version of this corpus only covered phones and laptops — comparisons
and product-spec lookup, but not the other categories the project targets.
It's now spread across all five categories from `project_overview`'s
suggested-categories table:

| project_overview category | Topics here | Entities |
|---|---|---|
| Product comparison / specification lookup | `phones`, `laptops`, `electronics` | iPhone 15, Galaxy S24, MacBook Air 13 M3, Dell XPS 13, AirPods Pro 2, Sony WF-1000XM5 |
| Travel | `travel` | Jaipur, Kyoto, Oberoi Rajvilas Jaipur, Ibis Jaipur City Centre |
| Specification lookup (non-product) | `buildings`, `geography` | Burj Khalifa, Empire State Building, Mount Everest, K2 |
| Multi-attribute research | `countries` | India, Japan |
| Decision making | `running_shoes` | Nike Pegasus 41, Hoka Clifton 9 |
| Sustainability | `vehicles` | Tesla Model 3, Nissan Leaf |

The general-question expansion adds `film` (Mamma Mia) and a `tire`
procedure entity without removing any of the categories above. The task
answer-type enum is closed over `single_fact`, `yes_no`, `list`,
`comparison`, `multi_part`, `procedure`, and `narrative`; every type has a
hand-written seed.

`tests/test_data.py::test_topic_diversity_covers_all_five_categories`
enforces that every category has at least one topic present, specifically
so the corpus can't silently narrow back to gadgets-only again.

## How this data was built (in order)

```bash
python scripts/build_seed_corpus.py          # writes corpus.jsonl
python scripts/build_seed_tasks.py            # writes tasks_seed.json
python scripts/generate_synthetic_tasks.py    # writes tasks.json (seed + synthetic)
python scripts/make_splits.py                 # writes splits.json, freezes it
python scripts/validate_corpus_task_coverage.py  # scripted coverage check
```

## Honest status vs. the plan.md targets

- **Corpus size**: 86 passages, not yet 200–500. Every passage traces to
  an actual sourced fact with a `source_url` (gathered via web search on
  2026-09-09), spread across 9 distinct topics rather than concentrated in
  one — but still a fraction of the 200–500 target. Per the plan's review
  addendum, closing the remaining gap should draw on an existing
  open-source passage dataset (e.g. a subset of MS MARCO or Natural
  Questions adapted into this `Passage` schema) rather than manually
  sourcing more by hand — and any such addition should preserve this
  topic spread, not just add more of the existing categories.
- **Task count**: 164 tasks, within the 150–200 target. 41 are
  hand-written seeds (spread across 10 topics and all seven answer types);
  123 are deterministic, template-based paraphrases capped at 3 variants per seed
  to keep the total in range as the seed set grew (**not** LLM output —
  see `scripts/generate_synthetic_tasks.py`'s docstring and the no-LLM-API
  guard in `tests/test_no_llm_api.py`, which this script also passes).
- **Embeddings**: `scripts/embed_corpus.py` is written but has not been
  run. This sandbox can't run it right now for two separate reasons: it
  ran out of disk space installing `sentence-transformers`, and even
  installed, `huggingface.co` isn't on this sandbox's network allowlist
  (only package registries are). Run it on a normal machine with
  `pip install sentence-transformers` first.
- **Manual review**: the required "review 10 random tasks for clarity"
  step was actually performed twice (once before and once after the
  topic-diversity fix) and caught a real issue the first time — the
  `best_time_to_visit` attribute produced grammatically awkward questions
  like *"What is the best time to visit of Jaipur?"* under the "X of Y"
  templates. Fixed by rephrasing that attribute to "recommended travel
  season"; all data was regenerated after the fix. The general-question
  expansion was reviewed again on 2026-09-09: 10 randomly sampled tasks
  were clear. Direct review of the new variants found that the India
  multi-part templates used singular "official language"; this was changed
  to plural "official languages" before the final regeneration.
- **Procedure coverage limitation**: a multi-step procedure is represented by
  one `Requirement(entity="tire", attribute="procedure")`. The evidence
  tracker therefore can mark that requirement satisfied after finding one
  reasonably similar passage, even if the passage does not cover every
  procedural step. This is an accepted v1 tracker-granularity limitation.
- **Judging v1/v2 path**: v1 dispatches rule-based string/set matching for
  structured answer types and token-F1 overlap for procedure/narrative.
  Procedure checks each required step at F1 >= 0.4 and applies its existing
  `match_threshold`; narrative passes at a fixed F1 >= 0.4. Both free-text
  dispatch entries can later be replaced by a local NLI judge without
  changing the calling contract, which always returns `(success, score)`.

## Validated by

`tests/test_data.py`: schema validity, byte-size correctness,
task-count range, synthetic-variant ground-truth fidelity, full
requirement→passage coverage, topic diversity across all five project
categories, and — specifically guarding the leakage concern from the
plan's review — that no seed group is split across train/val/test.
