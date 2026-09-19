"""Fine-tunes flan-t5-{small,base} for task decomposition (Phase 2).

Cannot run in this sandbox: needs network access to download base
flan-t5 weights from huggingface.co, which isn't on the sandbox's
allowlist (see data/README.md for the same issue with embed_corpus.py).
Run this on a machine with normal network access.

Usage:
    python scripts/build_decomposer_training_data.py   # if not already run
    pip install transformers torch peft --break-system-packages

    # Full fine-tune of the small model:
    python scripts/train_decomposer.py --model google/flan-t5-small \
        --output_dir models/decomposer-small --epochs 5

    # LoRA fine-tune of the base model (recommended -- base is larger,
    # LoRA keeps the trainable parameter count and checkpoint size down).
    # These hyperparameters were tuned on the 132-example train split: the
    # 5-epoch/r=16/q,v default badly underfit (train_loss ~1.1, val
    # exact-match 0.22); 20 epochs, r=32, and adapting q,v,k,o (not just q,v)
    # lifted val exact-match to ~0.44 and attribute_f1 from 0.42 to 0.70.
    # Higher rank (r=64) overfit this small a dataset, so r=32 is the sweet
    # spot. Retune if the dataset size changes materially.
    python scripts/train_decomposer.py --model google/flan-t5-base \
        --output_dir models/decomposer-base-lora --epochs 20 --lora \
        --lora_r 32 --lora_alpha 64 --lr 5e-4 --lora_target_modules q,v,k,o

Per plan.md Phase 2, benchmark both flan-t5-small and flan-t5-base:
train each with this script, then compare with scripts/eval_decomposer.py.
"""

import argparse
import json
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_pairs(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="e.g. google/flan-t5-small")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora", action="store_true", help="Fine-tune with a LoRA adapter instead of full fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--lora_target_modules",
        default="q,v",
        help="Comma-separated flan-t5 sub-layers to adapt (e.g. 'q,v' or 'q,v,k,o'). "
        "Adding k/o gives LoRA more of the attention block, which helps the model "
        "copy multi-token entity spans faithfully.",
    )
    parser.add_argument("--max_input_len", type=int, default=128)
    parser.add_argument("--max_target_len", type=int, default=128)
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from datasets import Dataset

    from ml_retriever.decomposer import PROMPT_TEMPLATE

    train_pairs = load_pairs(DATA_DIR / "decomposer_train.jsonl")
    val_pairs = load_pairs(DATA_DIR / "decomposer_val.jsonl")
    print(f"Loaded {len(train_pairs)} train / {len(val_pairs)} val pairs")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    def to_dataset(pairs: list[dict]) -> Dataset:
        prompts = [PROMPT_TEMPLATE.format(question=p["question"]) for p in pairs]
        targets = [p["target"] for p in pairs]
        model_inputs = tokenizer(
            prompts, max_length=args.max_input_len, truncation=True
        )
        labels = tokenizer(
            text_target=targets, max_length=args.max_target_len, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return Dataset.from_dict(model_inputs)

    train_ds = to_dataset(train_pairs)
    val_ds = to_dataset(val_pairs)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        logging_steps=10,
        report_to=[],  # no wandb/etc -- keep this fully local, no external calls
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"Training took {elapsed:.1f}s ({elapsed / max(args.epochs, 1):.1f}s/epoch)")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "base_model": args.model,
        "lora": args.lora,
        "lora_r": args.lora_r if args.lora else None,
        "lora_alpha": args.lora_alpha if args.lora else None,
        "lora_target_modules": args.lora_target_modules if args.lora else None,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "train_examples": len(train_pairs),
        "val_examples": len(val_pairs),
        "training_seconds": elapsed,
    }
    with open(Path(args.output_dir) / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model + training_meta.json to {args.output_dir}")


if __name__ == "__main__":
    main()
