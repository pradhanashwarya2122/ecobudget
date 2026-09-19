"""Phase 2: Task Decomposition.

Turns a free-text question into a list of `Requirement` objects
(entity, attribute pairs). Two implementations are provided:

- `HeuristicDecomposer`: a small rule-based baseline with no model
  dependency. Exists so (a) the training/eval scripts have something to
  run against before any model is trained, and (b) Phase 2's "> 90%"
  target has an honest baseline to beat, not just a number in a vacuum.
  This mirrors the fallback pattern in `backend/usefulness.py`'s
  `UsefulnessModel`.

- `TaskDecomposer`: wraps a local seq2seq model (flan-t5-small or
  flan-t5-base, optionally with a LoRA adapter) fine-tuned by
  `scripts/train_decomposer.py`. This is the actual Phase 2 deliverable.
  Loading it requires `transformers` (+ `peft` for LoRA) and either a
  local checkpoint or network access to download base flan-t5 weights --
  neither is available in this sandbox, so it is not exercised by the
  test suite here. Train and evaluate it on a machine with those.

Both share the same serialization format so training data, model
output, and heuristic output are all interchangeable.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

from ml_retriever.types import Requirement

# --- Serialization -----------------------------------------------------
#
# Requirements are serialized as "entity|attribute" pairs joined by " ## ".
# This is deliberately not JSON: seq2seq models trained on short structured
# strings like this are more sample-efficient and easier to constrain than
# ones trained to emit valid JSON, and parsing is still trivial.
#
# Example: "iPhone 15|price ## Samsung Galaxy S24|price"

REQ_SEP = " ## "
FIELD_SEP = "|"

# Glue words dropped before canonicalizing an attribute, so slugs that differ
# only in connecting words match (e.g. "year of first ascent" vs
# "first_ascent_year"). Kept tiny -- only true glue words -- so distinct
# attributes never collapse together (tests/test_eval_metric.py guards this).
_ATTR_STOPWORDS = {"of", "the", "a", "an"}


def normalize_attribute(attr: str) -> str:
    """Canonical form of an attribute slug: lowercase, split on non-alphanumerics,
    drop glue words, token-sort. Word order and separator style stop mattering."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", attr.lower()) if t]
    tokens = [t for t in tokens if t not in _ATTR_STOPWORDS]
    return "_".join(sorted(tokens))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def snap_attribute(attr: str, norm_vocab: dict[str, str], max_rel_dist: float = 0.4) -> str:
    """Snap a free-form attribute to the nearest known slug.

    The decomposer emits open-vocabulary attributes and often lands one edit
    away from a real slug ("noises_cancelation", "first_ascent" for
    "first_ascent_year"). `norm_vocab` maps each known slug's normalized form to
    its canonical spelling. An exact normalized hit returns the canonical slug;
    otherwise the nearest slug by edit distance wins, but only if it is within
    `max_rel_dist` of the longer string -- far-off predictions are left as-is
    rather than force-snapped to a wrong slug."""
    if not norm_vocab:
        return attr
    na = normalize_attribute(attr)
    if na in norm_vocab:
        return norm_vocab[na]
    best, best_d = None, None
    for nv, canonical in norm_vocab.items():
        d = _levenshtein(na, nv)
        if best_d is None or d < best_d:
            best, best_d, best_nv = canonical, d, nv
    if best is not None and best_d <= max_rel_dist * max(len(na), len(best_nv), 1):
        return best
    return attr


def load_attribute_vocab(corpus_path) -> list[str]:
    """Distinct attribute slugs present in a corpus.jsonl, for snap_attribute."""
    import json

    attrs = set()
    with open(corpus_path) as f:
        for line in f:
            row = json.loads(line)
            attrs.add(row["metadata"]["attribute"])
    return sorted(attrs)


def serialize_requirements(reqs: list[Requirement]) -> str:
    if not reqs:
        return ""
    return REQ_SEP.join(f"{r.entity}{FIELD_SEP}{r.attribute}" for r in reqs)


def parse_requirements(text: str) -> list[Requirement]:
    """Parses model/heuristic output back into Requirement objects.

    Malformed segments (missing the field separator, empty entity or
    attribute) are dropped rather than raising, since generated text is
    never guaranteed well-formed. Duplicate (entity, attribute) pairs
    (by `Requirement.key()`) are dropped, keeping the first occurrence.
    """
    reqs: list[Requirement] = []
    seen: set[tuple[str, str]] = set()
    for segment in text.split(REQ_SEP):
        segment = segment.strip()
        if not segment or FIELD_SEP not in segment:
            continue
        entity, _, attribute = segment.partition(FIELD_SEP)
        entity, attribute = entity.strip(), attribute.strip()
        if not entity or not attribute:
            continue
        req = Requirement(entity=entity, attribute=attribute)
        if req.key() in seen:
            continue
        seen.add(req.key())
        reqs.append(req)
    return reqs


# --- Heuristic baseline --------------------------------------------------

# Mirrors the attribute vocabulary used to build the Phase 1 corpus/tasks
# (see scripts/build_seed_corpus.py, scripts/generate_synthetic_tasks.py).
# A real NL question won't always use these exact words, which is exactly
# why this is a baseline to beat, not the Phase 2 deliverable.
_ATTRIBUTE_KEYWORDS = {
    "price": "price",
    "cost": "price",
    "battery capacity": "battery",
    "battery life": "battery_life",
    "battery": "battery",
    "weight": "weight",
    "chip": "chip",
    "processor": "chip",
    "camera": "camera",
    "display refresh rate": "display_refresh_rate",
    "refresh rate": "display_refresh_rate",
    "display": "display",
    "screen": "display",
    "travel season": "best_time_to_visit",
    "best time to visit": "best_time_to_visit",
    "daily budget": "daily_budget",
    "top attraction": "top_attraction",
    "trip length": "typical_trip_length",
    "price per night": "price_per_night",
    "rating": "rating",
    "amenities": "amenities",
    "room count": "room_count",
    "rooms": "room_count",
    "noise cancellation": "noise_cancellation",
    "codec": "codec_support",
    "height": "height",
    "tall": "height",
    "floor count": "floors",
    "floors": "floors",
    "completion year": "completion_year",
    "completed": "completion_year",
    "architect": "architect",
    "elevation": "elevation",
    "location": "location",
    "first ascent": "first_ascent_year",
    "population": "population",
    "land area": "area",
    "area": "area",
    "capital": "capital",
    "official language": "official_language",
    "language": "official_language",
    "cushioning": "cushioning",
    "recommended use": "best_for",
    "driving range": "range",
    "range": "range",
    "battery capacity": "battery_capacity",
    "change a flat car tire": "procedure",
    "change a tire": "procedure",
    "replace a flat tire": "procedure",
    "summarize": "summary",
    "summary": "summary",
    "premise": "summary",
}

# Longest-keyword-first so "battery life" matches before "battery".
_SORTED_ATTR_PATTERNS = sorted(_ATTRIBUTE_KEYWORDS, key=len, reverse=True)

_ENTITY_SPLIT_RE = re.compile(r"\band\b|,|\bor\b", flags=re.IGNORECASE)


def _guess_attribute(question: str) -> Optional[str]:
    q = question.lower()
    for phrase in _SORTED_ATTR_PATTERNS:
        if phrase in q:
            return _ATTRIBUTE_KEYWORDS[phrase]
    return None


def _guess_entities(question: str, known_entities: list[str]) -> list[str]:
    """Finds known entity names that appear verbatim in the question.

    Requires a `known_entities` gazetteer (e.g. every entity in the
    corpus) because free text alone doesn't reliably delimit multi-word
    product/place names ("iPhone 15", "Empire State Building") the way
    splitting on "and"/commas would. This is the heuristic's main
    limitation and a good chunk of why it's not expected to hit >90%.
    """
    q = question
    found = [e for e in known_entities if e.lower() in q.lower()]
    # Preserve question order rather than gazetteer order.
    found.sort(key=lambda e: q.lower().index(e.lower()))
    return found


class HeuristicDecomposer:
    """Rule-based baseline: keyword-match one attribute, gazetteer-match
    entities, pair every found entity with the found attribute.

    Cannot handle multi-attribute questions ("compare X and Y on A and
    B") correctly -- it only extracts one attribute per question -- and
    depends on an explicit entity gazetteer. Both are intentional
    limitations that motivate Phase 2's ML model.
    """

    version = "heuristic-v1"

    def __init__(self, known_entities: list[str]):
        self.known_entities = known_entities

    def decompose(self, question: str) -> list[Requirement]:
        attribute = _guess_attribute(question)
        entities = _guess_entities(question, self.known_entities)
        if attribute is None or not entities:
            return []
        return [Requirement(entity=e, attribute=attribute) for e in entities]


# --- ML-backed decomposer -------------------------------------------------

PROMPT_TEMPLATE = (
    "Decompose this question into entity|attribute requirements, "
    "separated by ' ## ':\n{question}"
)


@dataclass
class DecomposerBenchmark:
    """Timing/memory numbers from a single decompose() call, for the
    model-size comparison Phase 2 asks for."""

    latency_seconds: float
    peak_cuda_memory_bytes: Optional[int]


class TaskDecomposer:
    """Wraps a fine-tuned flan-t5 (small or base) seq2seq model.

    Requires `transformers`, and `peft` if `adapter_path` is given.
    Loading a model needs either a local `model_name_or_path` checkpoint
    directory (as written by `scripts/train_decomposer.py`) or network
    access to download the base flan-t5 weights on first use -- this
    class does not vendor or cache weights itself.
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 64,
        device: Optional[str] = None,
        attribute_vocab: Optional[list[str]] = None,
        snap_max_rel_dist: float = 0.4,
    ):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "TaskDecomposer requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch --break-system-packages"
            ) from e

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

        self.version = model_name_or_path
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError(
                    "adapter_path was given but 'peft' is not installed. "
                    "Install with: pip install peft --break-system-packages"
                ) from e
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.version = f"{model_name_or_path}+lora:{adapter_path}"

        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        # Closed-vocabulary attribute constraint: snap each generated attribute
        # to the nearest known slug. Entity extraction is near-solved; most
        # remaining errors are attributes a single edit off a real slug.
        self._norm_vocab = (
            {normalize_attribute(v): v for v in attribute_vocab}
            if attribute_vocab
            else {}
        )
        self.snap_max_rel_dist = snap_max_rel_dist

    def decompose(self, question: str) -> list[Requirement]:
        reqs, _ = self.decompose_with_benchmark(question)
        return reqs

    def decompose_with_benchmark(
        self, question: str
    ) -> tuple[list[Requirement], DecomposerBenchmark]:
        torch = self._torch
        prompt = PROMPT_TEMPLATE.format(question=question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            # no_repeat_ngram_size + repetition_penalty stop the degenerate
            # loops greedy decoding falls into on questions whose attribute
            # the model is unsure of (e.g. "recognition recognition ...").
            # Targets are short structured strings, so this never truncates a
            # legitimate answer.
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
            )
        latency = time.perf_counter() - start

        peak_mem = (
            torch.cuda.max_memory_allocated(self.device)
            if torch.cuda.is_available()
            else None
        )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reqs = parse_requirements(text)
        if self._norm_vocab:
            snapped: list[Requirement] = []
            seen: set[tuple[str, str]] = set()
            for r in reqs:
                sr = Requirement(
                    entity=r.entity,
                    attribute=snap_attribute(r.attribute, self._norm_vocab, self.snap_max_rel_dist),
                    value=r.value,
                )
                if sr.key() in seen:
                    continue
                seen.add(sr.key())
                snapped.append(sr)
            reqs = snapped
        return reqs, DecomposerBenchmark(
            latency_seconds=latency, peak_cuda_memory_bytes=peak_mem
        )
