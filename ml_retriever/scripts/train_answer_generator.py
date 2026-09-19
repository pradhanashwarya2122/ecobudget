"""Phase 6: fine-tune the generative answerer (flan-t5-small + LoRA).

Reads data/answer_{train,val}.jsonl (build_answer_training_data.py) and trains a
seq2seq model to map the evidence prompt to the synthesized answer. Small + LoRA
per plan.md review-2 (don't default to base; keep memory/time modest). Frozen
after training -- must NOT be retrained against the bandit reward.

Usage:
    python scripts/build_answer_training_data.py
    python scripts/train_answer_generator.py --model google/flan-t5-small \
        --output_dir models/answer-small --epochs 12 --lora
"""
import argparse
import json
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def load(path):
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/flan-t5-small")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_input_len", type=int, default=512)
    ap.add_argument("--max_target_len", type=int, default=64)
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)

    train, val = load(DATA / "answer_train.jsonl"), load(DATA / "answer_val.jsonl")
    print(f"Loaded {len(train)} train / {len(val)} val pairs")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=0.05, target_modules=["q", "v", "k", "o"]))
        model.print_trainable_parameters()

    def to_ds(rows):
        mi = tok([r["input"] for r in rows], max_length=args.max_input_len, truncation=True)
        mi["labels"] = tok(text_target=[r["target"] for r in rows],
                           max_length=args.max_target_len, truncation=True)["input_ids"]
        return Dataset.from_dict(mi)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, predict_with_generate=True, logging_steps=20, report_to=[])
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=to_ds(train),
                             eval_dataset=to_ds(val), data_collator=DataCollatorForSeq2Seq(tok, model=model),
                             processing_class=tok)
    t0 = time.time()
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    meta = {"base_model": args.model, "lora": args.lora, "epochs": args.epochs,
            "lr": args.lr, "lora_target_modules": "q,v,k,o" if args.lora else None,
            "train_examples": len(train), "training_seconds": time.time() - t0}
    json.dump(meta, open(Path(args.output_dir) / "answer_meta.json", "w"), indent=2)
    print(f"Saved answerer to {args.output_dir} ({meta['training_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
