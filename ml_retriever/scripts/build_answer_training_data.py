"""Phase 6: build (prompt -> answer) pairs to fine-tune the generative answerer.

Input = the ANSWER_PROMPT filled with the question and the task's GOLD evidence
(the corpus passages matching each requirement's (entity, attribute)); target =
the display answer the model should synthesize (expected_answer when present --
e.g. a comparison verdict -- else the string/joined-required_facts gold).

Train/val only (test stays frozen). narrative/list included so the generator
learns those forms too. Writes data/answer_{train,val}.jsonl.
"""
import argparse
import json
import random
from pathlib import Path

from ml_retriever.answer import ANSWER_PROMPT
from ml_retriever.types import Requirement

DATA = Path(__file__).resolve().parent.parent / "data"


def target_answer(task) -> str:
    """The training target MUST match what the judge scores against (fork A),
    so train and eval use the same gold. That is `required_facts` for the
    structured/realigned types, the plain string for narrative, and only then
    `expected_answer` (single_fact's value). `expected_answer` verdict strings
    are display-only and are NOT used as targets -- training on verdicts for
    some comparisons and values for others was the inconsistency bug."""
    gt = task.get("ground_truth")
    if isinstance(gt, dict) and gt.get("required_facts"):
        return ", ".join(gt["required_facts"])
    if isinstance(gt, str):
        return gt
    return str(task.get("expected_answer") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise_aug", type=int, default=0,
                    help="per clean example, add this many NOISE-AUGMENTED copies whose "
                         "evidence embeds the gold passage in distractor spec-dump text "
                         "(closes the corpus-to-web domain gap; target unchanged).")
    ap.add_argument("--distractors", type=int, default=6,
                    help="how many distractor passages to surround the gold with per aug copy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    corpus = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    by_key = {}
    for r in corpus:
        m = r.get("metadata", {})
        if "entity" in m and "attribute" in m:
            by_key.setdefault((m["entity"].lower(), m["attribute"].lower()), []).append(r["text"])
    all_texts = [r["text"] for r in corpus]

    tasks = {t["id"]: t for t in json.loads((DATA / "tasks.json").read_text())}
    splits = json.loads((DATA / "splits.json").read_text())

    def noisy_evidence(gold_texts):
        """Embed the gold passages among distractor passages, shuffled -- mimics a
        real web page's messy multi-fact spec dump so the answerer learns to
        extract the wanted value from noise rather than echo a clean snippet."""
        goldset = set(gold_texts)
        pool = [t for t in all_texts if t not in goldset]
        distractors = rng.sample(pool, min(args.distractors, len(pool)))
        mixed = list(gold_texts) + distractors
        rng.shuffle(mixed)
        return " ".join(mixed)

    for split in ("train", "val"):
        rows = []
        for tid in splits[split]:
            t = tasks[tid]
            target = target_answer(t)
            if not target:
                continue
            evidence_texts = []
            for r in t["decomposed_requirements"]:
                key = (r["entity"].lower(), r["attribute"].lower())
                evidence_texts.extend(by_key.get(key, []))
            if not evidence_texts:
                continue
            # clean example (unchanged)
            rows.append({"task_id": tid,
                         "input": ANSWER_PROMPT.format(question=t["question"],
                                                       evidence=" ".join(evidence_texts)[:2000]),
                         "target": target})
            # noise-augmented copies (train only -- keep val a clean measurement)
            if split == "train":
                for k in range(args.noise_aug):
                    rows.append({"task_id": f"{tid}-noise{k}",
                                 "input": ANSWER_PROMPT.format(question=t["question"],
                                                               evidence=noisy_evidence(evidence_texts)[:2000]),
                                 "target": target})
        out = DATA / f"answer_{split}.jsonl"
        with out.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_noise = sum(1 for r in rows if "-noise" in r["task_id"])
        print(f"Wrote {len(rows)} {split} pairs to {out} ({n_noise} noise-augmented)")


if __name__ == "__main__":
    main()
