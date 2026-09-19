# Reproducibility appendix (Phase G)

This appendix lets a reader reproduce every headline number in the project. It
lists the environment, the exact commands, the seeds, the model checkpoints and
their configurations, and where each reported number comes from. No em dashes are
used. All values were produced by our team on one shared Apple Silicon laptop
using the PyTorch MPS backend.

## Environment

- Python 3.13.7 in a virtual environment (`pip install -e ".[dev,train]"`).
- Key library versions (from `pip freeze`):
  - torch 2.14.0 (MPS backend on Apple Silicon)
  - transformers 5.17.0
  - sentence-transformers 6.0.1
  - peft 0.20.0
  - scikit-learn 1.9.0
  - numpy 2.5.3
  - datasets 5.0.1
  - joblib 1.6.0
  - matplotlib 3.11.2
- A CUDA GPU is much faster for the fine-tuning steps but is not required.
- Repository commit at the time of writing: 2b45a34 (use `git rev-parse HEAD`).

## Determinism and seeds

- Data splits are deterministic and frozen (`scripts/make_splits.py`, seed-level
  grouping, no randomness). See `data/splits.json.FROZEN`.
- Reported stopping policy (`models/bandit_policy.joblib`): trained with seed 0.
- Multi-seed variance runs use seeds 0, 1, 2, 3, 4 (`scripts/multiseed.py`).
- Ablation runs (`scripts/ablate_reward.py`) average over seeds 0, 1, 2.
- The frozen test split is evaluated exactly once for the headline; re-running the
  same frozen models is deterministic and is a reporting operation, not tuning.

## Model checkpoints and their configurations

| checkpoint | base | adapter / type | training config | data |
| --- | --- | --- | --- | --- |
| `models/decomposer-base-lora` | google/flan-t5-base | LoRA r=32, alpha=64, targets q,v,k,o | 20 epochs, lr 5e-4 | 1,008 train pairs |
| `models/answer-base` (reported) | google/flan-t5-base | LoRA | 12 epochs | 968 train pairs |
| `models/answer-base-robust` (not reported) | google/flan-t5-base | LoRA | 10 epochs | 2,904 rows (968 clean + 1,936 noise-augmented) |
| `models/bandit_policy.joblib` (reported) | n/a | per-action SGDClassifier(log_loss) | 8 epochs, per-step reward, lambda 0.5, seed 0 | train split |
| `models/linucb_policy.joblib` | n/a | LinUCB (alpha 1.0) | 8 epochs, lambda 0.5 | train split |
| `models/lints_policy.joblib` | n/a | linear Thompson (v 0.25) | 8 epochs, lambda 0.5 | train split |
| all-MiniLM-L6-v2 | frozen | bi-encoder | not trained | used to embed corpus and queries |
| roberta-base-squad2 | frozen | QA confidence | not trained | evidence sufficiency signal |

Checkpoints are gitignored (large). Regenerate them with the training commands
below. The decomposer took about 15 hours on our laptop; the answerer about 24
minutes; the noise-augmented answerer about 3.5 hours; the policies a few minutes
each.

## Data provenance

- `data/corpus.jsonl`: 186 passages, each with a stable (entity, attribute) label
  and a 384-dimension MiniLM embedding. Original passages carry real source URLs;
  the Phase C expansion passages use a synthetic provenance URL because their
  values come from a curated specifications table.
- `data/tasks.json`: 1,376 tasks (344 hand-authored seeds plus templated
  paraphrases), 512 of which are comparisons.
- `data/splits.json`: train 1,008 / val 240 / test 128, frozen, seed-grouped.
- `data/measured_payloads.json`: measured byte sizes of 19 real pages the
  `backend/` prototype fetched with Playwright (median rendered HTML 506,179 B).

## Commands to rebuild data and models

```bash
source .venv/bin/activate
# data
python scripts/build_seed_corpus.py
python scripts/build_seed_tasks.py
python scripts/expand_dataset_phase_c.py
python scripts/generate_synthetic_tasks.py
python scripts/curate_gold.py
python scripts/make_splits.py
python scripts/build_decomposer_training_data.py
python scripts/build_answer_training_data.py            # add --noise_aug 2 for the robust answerer
python scripts/embed_corpus.py
python scripts/measure_real_pages.py
# models
python scripts/train_decomposer.py --model google/flan-t5-base \
    --output_dir models/decomposer-base-lora --epochs 20 --lora \
    --lora_r 32 --lora_alpha 64 --lr 5e-4 --lora_target_modules q,v,k,o
python scripts/train_answer_generator.py --model google/flan-t5-base \
    --output_dir models/answer-base --epochs 12 --lora
python scripts/train_bandit.py --epochs 8 --lam 0.5 --reward_mode per_step
python scripts/train_baselines.py --epochs 8 --lam 0.5
```

## Where each headline number comes from

| number | value | command / source |
| --- | --- | --- |
| Decomposer exact-match (val, snapped) | 0.892 | `scripts/eval_decomposer.py --model_path models/decomposer-base-lora --adapter_path models/decomposer-base-lora` |
| Retriever recall@1 / recall@5 (entity-aware) | 0.989 / 1.000 | `scripts/eval_retriever.py --k 1` and `--k 5` |
| Frozen-test headline success / bytes | 0.895 / 106 B | `scripts/phase7_experiment.py --n 200 --split test` |
| Bandit vs full at measured page scale | -66.3 J/query | same test run, bandit-vs-baseline block |
| Energy vs full at 506 KB (val) | -109 J/query | `scripts/phase7_experiment.py --n 100 --split val` |
| Radio robustness | adaptive < full in 18/18 cells | `scripts/radio_sensitivity.py` |
| Multi-seed policy variance | bandit 0.790 +/- 0.283; LinUCB 0.913 +/- 0.007 | `scripts/multiseed.py --seeds 0 1 2 3 4` |
| Real-page byte reduction | 92.8% and 97.8% | `scripts/end_to_end_web.py` |
| Figures | figures/fig1..fig4.png | `scripts/make_figures.py` |

## Notes and honest caveats

- The reported learned policy is the seed-0 SGD bandit, but multi-seed shows it is
  high-variance; we recommend LinUCB as the stable learned policy.
- The frozen-test energy uses the measured 506 KB page payload (Tier A); earlier
  drafts used an assumed 60 KB and are superseded.
- `models/answer-base-robust` is retained as an experiment, not the reported
  model; see `eval_notes.md` for the domain-gap result.
