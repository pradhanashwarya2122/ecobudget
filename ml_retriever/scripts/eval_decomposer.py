"""Evaluates a decomposer against a held-out split (Phase 2).

Metrics (per plan.md):
    - decomposition accuracy: exact match on the full (entity, attribute)
      set per question -- both entity AND attribute must be correct for
      every requirement, order-independent. Target: > 90% on val.
    - micro F1: precision/recall over individual (entity, attribute)
      pairs across the whole eval set -- gives partial credit for
      questions the exact-match metric scores as a total miss (e.g. 3/4
      correct requirements on a 4-requirement comparison question).
    - per-example latency and (if on GPU) peak memory, for the
      flan-t5-small vs flan-t5-base benchmark plan.md asks for.

Defaults to the val split. The test split is intentionally not an
option here -- see build_decomposer_training_data.py's docstring and
plan.md: it stays untouched until Phase 7.

Usage:
    # Heuristic baseline, no model needed:
    python scripts/eval_decomposer.py --heuristic

    # A trained checkpoint:
    python scripts/eval_decomposer.py --model_path models/decomposer-small

    # A LoRA adapter on top of a base model:
    python scripts/eval_decomposer.py --model_path google/flan-t5-base \
        --adapter_path models/decomposer-base-lora

    # Compare two checkpoints side by side:
    python scripts/eval_decomposer.py --model_path models/decomposer-small --tag small
    python scripts/eval_decomposer.py --model_path models/decomposer-base-lora --adapter_path models/decomposer-base-lora --tag base+lora
    # (each run appends a row to eval_results.jsonl for later comparison)
"""

import argparse
import json
import statistics
from pathlib import Path

from ml_retriever.decomposer import (
    HeuristicDecomposer,
    load_attribute_vocab,
    normalize_attribute,
    parse_requirements,
)
from ml_retriever.types import Requirement

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_pairs(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def known_entities_from(tasks_path: Path) -> list[str]:
    with open(tasks_path) as f:
        tasks = json.load(f)
    entities = set()
    for t in tasks:
        for r in t["decomposed_requirements"]:
            entities.add(r["entity"])
    # Longest first so gazetteer matching in HeuristicDecomposer prefers
    # e.g. "Empire State Building" over any shorter accidental substring.
    return sorted(entities, key=len, reverse=True)


def _strict_key(r: Requirement) -> tuple[str, str]:
    return r.key()


def _norm_key(r: Requirement) -> tuple[str, str]:
    entity, attribute = r.key()
    return (entity, normalize_attribute(attribute))


def exact_match(pred: list[Requirement], gold: list[Requirement]) -> bool:
    return {r.key() for r in pred} == {r.key() for r in gold}


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate(decompose_fn, pairs: list[dict]) -> dict:
    # Strict = entity AND exact attribute string. Normalized = attribute
    # canonicalized first, so "year_of_first_ascent" == "first_ascent_year".
    # Entity-only and attribute-only F1 separate "found the right things"
    # from "spelled the attribute the way the key did" -- the crux of why a
    # single strict number is misleading.
    exact_matches = 0
    exact_matches_norm = 0
    entity_exact = 0
    strict_tp = strict_fp = strict_fn = 0
    ent_tp = ent_fp = ent_fn = 0
    attr_tp = attr_fp = attr_fn = 0
    latencies = []

    for pair in pairs:
        gold = parse_requirements(pair["target"])
        result = decompose_fn(pair["question"])
        if isinstance(result, tuple):
            pred, latency = result
            latencies.append(latency)
        else:
            pred = result

        if exact_match(pred, gold):
            exact_matches += 1

        pred_norm = {_norm_key(r) for r in pred}
        gold_norm = {_norm_key(r) for r in gold}
        if pred_norm == gold_norm:
            exact_matches_norm += 1

        pred_ents = {r.key()[0] for r in pred}
        gold_ents = {r.key()[0] for r in gold}
        if pred_ents == gold_ents:
            entity_exact += 1

        pred_keys = {r.key() for r in pred}
        gold_keys = {r.key() for r in gold}
        strict_tp += len(pred_keys & gold_keys)
        strict_fp += len(pred_keys - gold_keys)
        strict_fn += len(gold_keys - pred_keys)

        ent_tp += len(pred_ents & gold_ents)
        ent_fp += len(pred_ents - gold_ents)
        ent_fn += len(gold_ents - pred_ents)

        pred_attrs = {normalize_attribute(r.key()[1]) for r in pred}
        gold_attrs = {normalize_attribute(r.key()[1]) for r in gold}
        attr_tp += len(pred_attrs & gold_attrs)
        attr_fp += len(pred_attrs - gold_attrs)
        attr_fn += len(gold_attrs - pred_attrs)

    n = len(pairs)
    precision, recall, f1 = _f1(strict_tp, strict_fp, strict_fn)
    _, _, entity_f1 = _f1(ent_tp, ent_fp, ent_fn)
    _, _, attribute_f1 = _f1(attr_tp, attr_fp, attr_fn)

    result = {
        "n_examples": n,
        "exact_match_accuracy": exact_matches / n if n else 0.0,
        "exact_match_normalized": exact_matches_norm / n if n else 0.0,
        "entity_exact_match": entity_exact / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "entity_f1": entity_f1,
        "attribute_f1": attribute_f1,
    }
    if latencies:
        result["mean_latency_seconds"] = statistics.mean(latencies)
        result["p95_latency_seconds"] = (
            statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val"], help="test is intentionally not selectable here -- see module docstring")
    parser.add_argument("--heuristic", action="store_true")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--tag", default=None, help="Label for this run in eval_results.jsonl")
    parser.add_argument(
        "--no_attr_snap",
        action="store_true",
        help="Disable snapping model-generated attributes to the known corpus "
        "vocabulary (for A/B comparison; snapping is on by default for model runs).",
    )
    args = parser.parse_args()

    if not args.heuristic and not args.model_path:
        parser.error("pass --heuristic or --model_path")

    pairs = load_pairs(DATA_DIR / f"decomposer_{args.split}.jsonl")

    if args.heuristic:
        known_entities = known_entities_from(DATA_DIR / "tasks.json")
        decomposer = HeuristicDecomposer(known_entities)
        decompose_fn = decomposer.decompose
        tag = args.tag or "heuristic"
        version = decomposer.version
    else:
        from ml_retriever.decomposer import TaskDecomposer

        vocab = None if args.no_attr_snap else load_attribute_vocab(DATA_DIR / "corpus.jsonl")
        decomposer = TaskDecomposer(
            args.model_path, adapter_path=args.adapter_path, attribute_vocab=vocab
        )

        # evaluate() understands either a bare prediction or a
        # (prediction, latency) tuple; use the latter so latency stats
        # get reported for model-backed runs.
        def decompose_fn(q):
            reqs, bench = decomposer.decompose_with_benchmark(q)
            return reqs, bench.latency_seconds

        tag = args.tag or args.model_path
        version = decomposer.version

    metrics = evaluate(decompose_fn, pairs)
    metrics["tag"] = tag
    metrics["version"] = version
    metrics["split"] = args.split

    print(json.dumps(metrics, indent=2))

    results_path = DATA_DIR / "decomposer_eval_results.jsonl"
    with open(results_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")
    print(f"Appended to {results_path}")


if __name__ == "__main__":
    main()
