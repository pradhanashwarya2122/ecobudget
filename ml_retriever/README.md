# EcoBudget: task-sufficient retrieval for greener 5G question answering

This is our team's research project. We set out to answer a simple but
surprisingly under-studied question: when a system answers a question from the
web, how little does it actually need to load? Most retrieval systems either pull
a fixed amount of context or grab whole pages and let the model sort it out. Both
move far more data than the answer requires. On a 5G link that waste is not free,
because moving bytes costs radio energy and carbon, and every fetch keeps the
phone's modem awake. EcoBudget decides, per question, exactly how much evidence it
needs, stops as soon as it has enough, and then measures the bytes and energy it
saved.

Everything here runs on local models. There is no hosted LLM API anywhere in the
pipeline, and we enforce that in code so the energy story reflects the system we
actually run, not a service we call. Every number in this document was produced on
our own hardware and can be reproduced with the commands below.

A note on how we worked: the whole team built this on one shared laptop (Apple
Silicon, using the MPS backend), so the training times we quote are what we
measured there. A CUDA GPU would be considerably faster.

## Table of contents

- [The idea in plain terms](#the-idea-in-plain-terms)
- [How the project is organised](#how-the-project-is-organised)
- [Architecture: what the pipeline is made of](#architecture-what-the-pipeline-is-made-of)
- [Datasets](#datasets)
- [Models and training](#models-and-training)
- [Results](#results)
- [Local models only](#local-models-only)
- [Setup](#setup)
- [Running the system](#running-the-system)
- [Reproducing everything from scratch](#reproducing-everything-from-scratch)
- [Reproducing the individual experiments](#reproducing-the-individual-experiments)
- [The four paper figures](#the-four-paper-figures)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)
- [What is left](#what-is-left)

## The idea in plain terms

We treat retrieval as a task-sufficiency problem. Instead of asking "can the
system answer," we ask "what is the smallest amount of loading that still answers
correctly," and then we check whether the energy saved by loading less actually
beats the energy spent running the models. The pipeline does four things:

1. It breaks a question into small, explicit information needs. We call each one a
   requirement, written as an (entity, attribute) pair, for example
   (iPhone 16, price).
2. For each requirement it retrieves candidate passages, gating them to the right
   entity first and then ranking by the attribute.
3. A stopping policy decides after each retrieval whether it already has enough
   evidence (STOP) or should pull one more passage (RETRIEVE).
4. It generates the answer from only the evidence it actually gathered, and it
   records the bytes moved and the energy that implies on a 5G access path.

## How the project is organised

EcoBudget has three workstreams that the team built together:

- `ml_retriever/` (this package): the reproducible machine-learning system, which
  is what this README documents. Decomposition, retrieval, evidence tracking, the
  adaptive stopping policy, answer generation, the energy and radio model, and the
  full experiment harness live here.
- `backend/`: an earlier FastAPI prototype that fetches real web pages with
  Playwright, extracts passages, and estimates transfer CO2e. We reuse its real
  fetched pages (`backend/pages/`) to ground our energy model in measured page
  sizes, and we drive our end-to-end demo against live pages the same way.
- `frontend/`: the project landing page (`frontend/landing.html`), including a
  full-screen hero video and an interactive iPhone demo that animates the
  task-sufficient loading idea using our measured numbers.

## Architecture: what the pipeline is made of

The pipeline is a chain of small, independent, dependency-injected components. We
wrote each one so its core logic is unit tested without downloading a model, which
is why the test suite runs in seconds and needs no network.

| component | file | what it does |
| --- | --- | --- |
| Decomposer | `ml_retriever/decomposer.py` | Fine-tuned flan-t5-base with a LoRA adapter. Turns a question into (entity, attribute) requirements, snapped to a closed vocabulary of 35 attributes so the output is reliable. |
| Retriever | `ml_retriever/retriever.py` | A MiniLM bi-encoder over a pre-embedded corpus. Our production retriever is the entity-aware two-stage version: gate candidates to the requirement's entity, then rank by the attribute phrase. |
| Evidence tracker | `ml_retriever/evidence.py` | Scores whether the gathered passages actually satisfy each requirement, using a RoBERTa question-answering confidence signal rather than raw similarity, because similarity alone is fooled by passages that mention the entity but not the attribute. |
| Stopping policy | `ml_retriever/bandit.py` | The decision layer. Includes our per-action contextual bandit (a linear SGD classifier) trained with a per-step reward, plus the LinUCB and linear Thompson sampling baselines, all sharing the same features and reward. |
| Rollout / deciders | `ml_retriever/rollout.py` | The episode loop and every baseline decider: full load, one-per-requirement, fixed budgets, the coverage heuristic, and an Adaptive-RAG analog. |
| Answer generator | `ml_retriever/answer.py` | Fine-tuned flan-t5-base with a LoRA adapter, called once per requirement, with an abstain path. |
| Judge | `ml_retriever/judge.py` | An answer-type-aware checker with a symmetric formatting normalizer. Reports judge_success (binary) and fact_f1 (fraction of required facts matched) separately. |
| Energy model | `ml_retriever/energy.py` | Compute energy from model FLOPs, a 5G per-bit transfer model, payload realism scenarios grounded in measured pages, and an RRC radio-state and tail-energy model. |
| Orchestrator | `ml_retriever/system.py` | The end-to-end deployed path that wires every component above together. |

Data flow for one query: question, then decompose into requirements, then
per-requirement retrieval under the stopping policy (STOP or RETRIEVE after each
step), then evidence sufficiency check, then answer generation from only the
gathered evidence, then measurement of bytes, compute energy, and 5G radio energy.

## Datasets

Everything the models learn from lives in `data/`. We built the data by hand and
with templated generation so that training targets match evaluation exactly (no
mismatch between what we train on and what the judge scores).

**Corpus (`data/corpus.jsonl`): 186 passages.** Each passage is a short factual
snippet with a stable (entity, attribute) label in its metadata, a source URL, a
byte size, and a 384-dimension MiniLM embedding. The corpus spans phones, laptops,
electronics, travel, buildings, geography, countries, running shoes, electric
vehicles, earbuds, and mountains. The original passages carry real source URLs
(gsmarena, Wikipedia, Britannica, Toms Guide, and similar); the Phase C expansion
passages use a synthetic provenance URL because their values come from a curated
specifications table rather than a single scraped page.

**Tasks (`data/tasks.json`): 1,376 tasks.** These are 344 hand-authored seed
tasks plus templated synthetic paraphrases of them. Task types are single_fact,
comparison, yes_no, multi_part, list, narrative, and procedure. Each task carries
its decomposed requirements and its gold required_facts. Comparisons and other
multi-fact types are judged against curated required_facts taken straight from the
specifications, so the gold is canonical.

**Splits (`data/splits.json`, frozen): train 1,008 / validation 240 / test 128.**
We split at the seed level, so every synthetic paraphrase of a seed lands in the
same split as its seed. This closes a leakage hole where near-duplicate wording of
the same question could otherwise appear in both train and test. A coverage
invariant, enforced in `scripts/make_splits.py` and in the tests, guarantees that
every answer type and every attribute slug appears in train (procedure is the one
documented exception, since its single seed is reserved for the frozen test set).
The split is frozen: we ran the final test set exactly once, at the very end.

**Measured payloads (`data/measured_payloads.json`).** To make the energy numbers
realistic rather than assumed, we measured the real byte sizes of the 19 pages the
`backend/` prototype fetched with Playwright. The median rendered HTML page is
506 KB and the median extracted text is about 92 KB. Our payload model uses the
measured 506 KB median for the html_page scenario instead of a guessed value.

**Per-model training data.** `data/decomposer_train.jsonl` and
`data/answer_train.jsonl` (plus their validation counterparts) are generated from
the train split only. The answerer training data can optionally be noise-augmented
(see below).

## Models and training

We fine-tuned three models and use two frozen pretrained models. All fine-tuning
uses LoRA adapters on flan-t5-base so it fits on a laptop.

**Decomposer (`models/decomposer-base-lora`).** flan-t5-base with a LoRA adapter
(r=32, alpha=64, target modules q,v,k,o), 20 epochs, learning rate 5e-4, on 1,008
training examples. This was the long one: about 15 hours on our shared Apple
Silicon laptop, final eval loss around 0.015. The output is snapped to the closed
35-attribute vocabulary, which is a principled form of constrained decoding for a
fixed schema.

```bash
python scripts/train_decomposer.py --model google/flan-t5-base \
    --output_dir models/decomposer-base-lora --epochs 20 --lora \
    --lora_r 32 --lora_alpha 64 --lr 5e-4 --lora_target_modules q,v,k,o
```

**Answerer (`models/answer-base`).** flan-t5-base with a LoRA adapter, 12 epochs,
on 968 training pairs, about 24 minutes on our laptop. Called once per requirement
with an abstain path.

```bash
python scripts/train_answer_generator.py --model google/flan-t5-base \
    --output_dir models/answer-base --epochs 12 --lora
```

**Noise-augmented answerer (`models/answer-base-robust`).** When we ran the
pipeline on raw live web pages, the answerer produced messy output, because it had
only ever seen clean structured passages. To close that corpus-to-web domain gap
we rebuilt the answerer training data with noise augmentation, wrapping each gold
passage in distractor spec-dump text so the model learns to extract the wanted
value from noise. This produces 2,904 training rows (968 clean plus 1,936
augmented), with validation kept clean.

```bash
python scripts/build_answer_training_data.py --noise_aug 2 --distractors 6
python scripts/train_answer_generator.py --model google/flan-t5-base \
    --output_dir models/answer-base-robust --epochs 10 --lora
```

**Stopping policy (`models/bandit_policy.joblib` + `models/bandit_normalizer.joblib`).**
Our contextual bandit is a per-action linear SGD classifier trained online with a
per-step reward that trades marginal evidence coverage against byte cost. We
selected the byte-penalty weight lambda by a pre-registered sweep and settled on
lambda = 0.5.

```bash
python scripts/train_bandit.py --epochs 8 --lam 0.5 --reward_mode per_step
```

**External baselines (`models/linucb_policy.joblib`, `models/lints_policy.joblib`).**
LinUCB and linear Thompson sampling, trained under identical conditions (same
features, same per-step reward, same normalizer, same candidates) so the
comparison is fair.

```bash
python scripts/train_baselines.py --epochs 8 --lam 0.5
```

**Frozen pretrained models we do not train.** The query encoder is
all-MiniLM-L6-v2 (used to embed the corpus and queries), and the evidence scorer
is roberta-base-squad2 (used for the sufficiency signal). Both are used as-is.

The trained checkpoints are not committed to git (they are large and are
gitignored), so on a fresh clone you regenerate them with the commands above.

## Results

All values are measured on our frozen test set or on real pages, unless a line
says validation.

**Retriever (175 unique seed requirements).** Our entity-aware, attribute-ranked
retriever raised recall@1 from 0.914 to 0.989 and recall@5 from 0.983 to a perfect
1.000. Reaching recall@5 of 1.000 means every list and amenities passage is now
retrieved, which was a known failure before.

**Frozen test set (124 headline tasks).** The adaptive methods reach 0.895
success and 0.944 fact_f1 using about 106 bytes per query, against 420 bytes for
full-page loading (about 75 percent fewer bytes) at statistically equal success.
Adaptive stopping also beats naive one-per-requirement retrieval on success by 6.5
points, and dominates the fixed and full baselines by about 300 bytes per query,
all with confidence intervals that exclude zero.

**Energy at realistic page scale (validation, measured 506 KB pages).** With the
measured page weight, adaptive stopping saves about 109 joules per query versus
full-page loading (95 percent confidence interval roughly 75 to 148), and transfer
is about 95 percent of total energy at that scale.

**5G radio and tail energy.** Fewer, earlier fetches shorten radio active time. In
the conservative model the per-query radio saving versus full load is a couple of
joules; under aggressive fast-dormancy release it grows to roughly 100 joules,
robust across all 18 coefficient settings we swept.

**Real live web pages.** Running the full pipeline end to end on live pages, we
measured about 93 to 98 percent fewer bytes than a full-page load, and roughly 14
times less transfer energy.

**An honest finding we report rather than hide.** Once the entity-aware retriever
made the first result almost always correct, our learned bandit converged to the
same policy as a strong hand-tuned heuristic, and across five random seeds our SGD
bandit was unstable (success 0.790 plus or minus 0.283) while LinUCB (0.913 plus
or minus 0.007) and the heuristic (0.917) were stable. Our contribution is the
task-sufficiency framework and the 5G energy and radio characterisation, not a new
bandit algorithm; among learned policies we recommend LinUCB.

We currently have 172 passing tests and the no-hosted-LLM-API guard is green.

## Local models only

This package must never depend on a hosted LLM API. `scripts/check_no_llm_api.py`
scans for banned imports (openai, anthropic, and similar) and known API hosts, and
`tests/test_no_llm_api.py` runs it as part of the suite.

```bash
python scripts/check_no_llm_api.py
```

## Setup

We use Python 3.13 in a virtual environment. Any Python 3.10 or newer works.

```bash
cd ml_retriever
python -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
pip install -e ".[dev,train]"
```

The main libraries are torch, transformers, sentence-transformers, peft,
scikit-learn, numpy, and joblib. On Apple Silicon torch uses the MPS backend
automatically. A CUDA GPU is much faster for the training steps.

## Running the system

First, confirm the suite is green:

```bash
source .venv/bin/activate
python -m pytest -q          # expect 172 passing, runs in seconds, no downloads
```

Then run the working pipeline on a few tasks. This uses the trained models in
`models/` and prints, per task, the answer, the gold, the judge result, and the
bytes used by the bandit versus the heuristic:

```bash
python scripts/run_pipeline.py --n 20 \
    --answerer models/answer-base \
    --answer_mode per_requirement
```

## Reproducing everything from scratch

Run these in order to rebuild the data and retrain the models end to end. The
training steps download the base flan-t5 weights the first time, so they need
network access. The committed `data/` already holds the frozen corpus, tasks, and
splits we used, so to match our exact numbers you can skip steps 1 and 2 and go
straight to embedding and training. Running steps 1 and 2 rebuilds the data
deterministically for a clean-room reproduction.

```bash
source .venv/bin/activate

# 1. Build the corpus and tasks, then scale up
python scripts/build_seed_corpus.py
python scripts/build_seed_tasks.py
python scripts/expand_dataset_phase_c.py
python scripts/generate_synthetic_tasks.py
python scripts/curate_gold.py

# 2. Cut the frozen splits and build the per-model training data
python scripts/make_splits.py
python scripts/build_decomposer_training_data.py
python scripts/build_answer_training_data.py            # add --noise_aug 2 for the robust answerer

# 3. Embed the corpus (needs network for the MiniLM download)
python scripts/embed_corpus.py

# 4. Ground the payload model in real pages (regenerate backend/pages first if needed)
python scripts/measure_real_pages.py

# 5. Train the models
python scripts/train_decomposer.py --model google/flan-t5-base \
    --output_dir models/decomposer-base-lora --epochs 20 --lora \
    --lora_r 32 --lora_alpha 64 --lr 5e-4 --lora_target_modules q,v,k,o
python scripts/train_answer_generator.py --model google/flan-t5-base \
    --output_dir models/answer-base --epochs 12 --lora
python scripts/train_bandit.py --epochs 8 --lam 0.5 --reward_mode per_step
python scripts/train_baselines.py --epochs 8 --lam 0.5
```

## Reproducing the individual experiments

```bash
source .venv/bin/activate

# Retriever recall: whole-question vs per-requirement vs entity-aware
python scripts/eval_retriever.py --k 1
python scripts/eval_retriever.py --k 5

# Main experiment on validation: success, bytes, energy, radio, latency, and
# bootstrap CIs for every condition including LinUCB, LinTS, and Adaptive-RAG
python scripts/phase7_experiment.py --n 100 --split val

# Byte-penalty selection: the lambda sweep and its cliff
python scripts/sweep_lambda.py --epochs 8 --lams 0.1 0.25 0.5 1 2 4

# When does adaptivity help: the pre-registered retrieval-noise sweep
python scripts/retrieval_noise_sweep.py --epochs 8 --noises 0 0.1 0.2 0.4 0.6

# Multi-seed variance: stability of each learned policy across seeds
python scripts/multiseed.py --seeds 0 1 2 3 4 --epochs 8 --lam 0.5

# Reward-mode ablation (Phase G): per-step vs terminal credit
python scripts/ablate_reward.py --seeds 0 1 2 --epochs 8 --lam 0.5

# Decomposer snap on/off ablation (Phase G)
python scripts/eval_decomposer.py --model_path models/decomposer-base-lora \
    --adapter_path models/decomposer-base-lora --no_attr_snap

# Radio-coefficient sensitivity across the cited 5G ranges
python scripts/radio_sensitivity.py

# End to end on real live web pages: measured byte and energy savings
python scripts/end_to_end_web.py

# The single frozen test-split run. Run this once, only after configs are locked.
python scripts/phase7_experiment.py --n 200 --split test

# Generate the four paper figures
python scripts/make_figures.py
```

## The four paper figures

`scripts/make_figures.py` writes four 300-dpi figures to `figures/`, using the
frozen test results plus the measured payload and real-page data:

- `fig1_pareto.png`: success versus energy. The adaptive methods cluster at the
  low-energy, high-success corner while fixed budgets and full-page load spread
  out to the right.
- `fig2_crossover.png`: compute versus 5G transfer energy across payload realism;
  transfer overtakes compute at real page scale.
- `fig3_radio_tail.png`: 5G RRC radio and tail energy by condition, tight loop
  versus fast dormancy.
- `fig4_realpage_savings.png`: measured byte savings on real live web pages.

## Repository layout

```
ml_retriever/
  pyproject.toml
  ml_retriever/               the package
    types.py                  Requirement, Passage, AnswerResult
    interfaces.py             tracker and generator protocols
    decomposer.py             question to requirements (flan-t5 + LoRA)
    retriever.py              bi-encoder + entity-aware retriever
    evidence.py               QA-confidence evidence coverage
    bandit.py                 SGD bandit, LinUCB, linear Thompson sampling
    rollout.py                episode loop, candidate building, deciders
    answer.py                 generative and extractive answer generators
    judge.py                  answer-type-aware judge and normalizer
    energy.py                 compute, 5G transfer, payloads, radio state
    system.py                 end-to-end orchestrator
  data/                       corpus, tasks, frozen splits, training data, measured payloads
  docs/
    roadmap_5g_green.md       the full plan with acceptance gates
    energy_model.md           every energy coefficient and its source
    eval_notes.md             per-phase measured results and decisions
    phase7_preregistration.md pre-registration for the main experiment
  figures/                    the four generated paper figures
  models/                     trained checkpoints and policy artifacts (gitignored)
  scripts/                    build, train, and evaluation scripts
  tests/                      172 tests, model-free where possible
```

## Limitations

We state these plainly so nothing is oversold.

- The contribution is the task-sufficiency framework plus the 5G energy and radio
  characterisation, not a new learning algorithm. Among learned policies we
  recommend LinUCB, which is stable across seeds; our custom SGD bandit ties it on
  a good seed but is high-variance, so we do not present it as the method.
- Answer quality degrades on raw live-page text (the corpus-to-web domain gap).
  We tried a noise-augmented answerer (`models/answer-base-robust`): it fixed the
  verbose spec-dump failure mode (answers became clean and concise) but did not
  fix correctness, because on live pages the pipeline has no (entity, attribute)
  metadata and picks the wrong field from coarse chunks. The real fix is
  per-entity live fetching with structured extraction, not more answerer training.
  We keep `models/answer-base` as the reported model; the byte and energy savings
  are unaffected by this.
- The corpus is domain-narrow (products, travel, specifications) and some of its
  source URLs are synthetic, so external validity beyond these domains is
  untested. The measured page-weight distribution comes from 19 real pages.
- Narrative tasks have no training coverage and currently fail; they are excluded
  from the headline metrics and reported as a gap rather than hidden.
- Energy is a first-order FLOPs-and-per-bit model with cited coefficients and
  reported sensitivity ranges, not a hardware power measurement.

## Phase G consolidation (done)

The rigor phase is complete. See `docs/phase_g_consolidation.md` for the
consolidated ablation table (retriever, decomposer snap, byte-penalty lambda,
reward mode, answer mode), the multi-seed variance, and the single frozen
test-split run, and `docs/reproducibility.md` for the environment, seeds,
checkpoint configurations, and the command behind every headline number. Selected
ablation highlights, all measured:

- Retriever: recall@1 0.589 (whole-question) to 0.914 (per-requirement) to 0.989
  (entity-aware).
- Decomposer snap-to-vocab: exact-match 0.496 (off) to 0.892 (on), attribute-only.
- Byte-penalty lambda: a sharp cliff, stable at 0.5 and below, collapses at 1.0.
- Reward mode and multi-seed both confirm the SGD bandit is high-variance, so we
  report LinUCB as the recommended learned policy.

## Deployed default policy: LinUCB

Given the multi-seed stability finding, the deployed path (`run_pipeline.py`) now
defaults to LinUCB (`--policy linucb`), with `--policy bandit` and `--policy
lints` still available. LinUCB is stable across seeds (val 0.913 +/- 0.007) and on
the frozen test set matches the bandit and heuristic at 0.895 success / 108 bytes.
Its metrics live in `docs/phase_g_consolidation.md` (Sections 2 and 3).

```bash
python scripts/run_pipeline.py --n 20 --policy linucb \
    --answerer models/answer-base --answer_mode per_requirement
```

## What is left

- The paper writeup itself: related work, the figures, and the limitations
  section.
