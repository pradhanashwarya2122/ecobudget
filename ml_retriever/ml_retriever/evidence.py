"""Phase 4: Evidence tracking with requirements.

Tracks, for a fixed set of Requirements, whether the Passages retrieved so far
provide sufficient evidence for each one -- using ONLY the passages added (no
ground truth, no corpus metadata). Satisfies the Phase 0 `EvidenceTracker`
protocol, so the controller/bandit can treat it as a black box.

Sufficiency is a CONTENT signal, not a metadata match. A passage is evidence
for a requirement to the degree its text is semantically close to the
requirement ("entity attribute"); the default scorer is MiniLM cosine
similarity (reusing a passage's precomputed embedding when present). This is
why a passage that mentions the right entity but never its attribute -- e.g.
"the iPhone 15 weighs 171 g" for the requirement (iPhone 15, price) -- scores
low and does NOT satisfy it, whereas a naive entity-match would wrongly count
it. The scorer is injectable so the test suite can exercise the tracker's
gating logic with a transparent, model-free lexical score.

Design mirrors the rest of the package: the scoring that needs a model is
lazy/injectable; the tracker's state machine is pure and fully unit-tested.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np

from .retriever import _unit, requirement_to_query
from .types import Passage, Requirement

# A scorer maps (requirement, passage) -> evidence strength, higher = stronger.
ScoreFn = Callable[[Requirement, Passage], float]


class MiniLMScorer:
    """Default evidence scorer: cosine similarity between the requirement's
    query embedding and the passage embedding.

    Reuses `passage.embedding` when present (corpus passages are pre-embedded);
    otherwise embeds the passage text. Requirement query embeddings are cached.
    Lazily imports sentence-transformers so importing this module -- and the
    model-free tests that inject their own scorer -- needs no model download.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._query_cache: dict[tuple[str, str], np.ndarray] = {}

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __call__(self, requirement: Requirement, passage: Passage) -> float:
        key = requirement.key()
        if key not in self._query_cache:
            vec = self._get_model().encode(requirement_to_query(requirement))
            self._query_cache[key] = _unit(np.asarray(vec, dtype=float))
        q = self._query_cache[key]
        emb = passage.embedding
        if emb is None:
            emb = self._get_model().encode(passage.text)
        return float(np.dot(q, _unit(np.asarray(emb, dtype=float))))


class QAScorer:
    """Attribute-sensitive evidence scorer via extractive QA.

    Pure embedding similarity is NOT enough for sufficiency: MiniLM embeddings
    are dominated by the entity name, so a passage about the iPhone 15's WEIGHT
    scores ~0.62 against the requirement (iPhone 15, price) and would wrongly
    satisfy it. QA discriminates the attribute directly: it asks "What is the
    {attribute} of {entity}?" of the passage and returns the model's answer
    confidence, which stays low when the passage does not actually contain that
    attribute's value. This is the QA-confidence signal plan.md Phase 4 calls
    for, reusing the same local extractive-QA model family as backend/scorer.py
    (no hosted API).

    Lazily loads transformers' QA pipeline; `handle_impossible_answer=True` lets
    it report low confidence when the answer is absent.
    """

    def __init__(self, model_name: str = "deepset/roberta-base-squad2", max_length: int = 384):
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    def _load(self):
        # Driven directly via AutoModelForQuestionAnswering rather than
        # pipeline("question-answering"): some transformers builds don't
        # register that pipeline task, but the model class is always present.
        if self._model is None:
            import torch  # noqa: F401
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            self._model.eval()
        return self._tokenizer, self._model

    def _question(self, requirement: Requirement) -> str:
        attribute = requirement.attribute.replace("_", " ").strip()
        return f"What is the {attribute} of {requirement.entity}?"

    def _answer(self, question: str, context: str) -> tuple[str, float]:
        """Best answer span + its confidence (net of the impossible score)."""
        import torch

        tokenizer, model = self._load()
        inputs = tokenizer(
            question, context, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        with torch.no_grad():
            out = model(**inputs)
        start = torch.softmax(out.start_logits[0], dim=-1)
        end = torch.softmax(out.end_logits[0], dim=-1)
        null_prob = float(start[0] * end[0])  # CLS = "no answer"
        s = int(torch.argmax(start[1:]).item()) + 1
        e = int(torch.argmax(end[s:]).item()) + s
        conf = max(0.0, float(start[s] * end[e]) - null_prob)
        span = tokenizer.decode(inputs["input_ids"][0][s : e + 1], skip_special_tokens=True).strip()
        return span, conf

    def __call__(self, requirement: Requirement, passage: Passage) -> float:
        text = passage.text
        if not text or len(text.strip()) < 10:
            return 0.0
        span, conf = self._answer(self._question(requirement), text)
        return conf if span else 0.0

    def extract_answer(self, requirement: Requirement, passage: Passage) -> str:
        """The answer span the model extracts for this requirement from the
        passage -- used to derive grounded `required_facts` for the judge."""
        text = passage.text
        if not text or len(text.strip()) < 10:
            return ""
        span, _ = self._answer(self._question(requirement), text)
        return span


class CachedScorer:
    """Memoizes a scorer by (requirement key, passage_id).

    Retrieval is deterministic, so the same (requirement, passage) pair recurs
    across bandit epochs and conditions; caching the (expensive) QA call keeps
    training and the multi-condition comparison from re-running the model on
    pairs already seen."""

    def __init__(self, score_fn: ScoreFn):
        self._score_fn = score_fn
        self._cache: dict[tuple[tuple[str, str], str], float] = {}

    def __call__(self, requirement: Requirement, passage: Passage) -> float:
        key = (requirement.key(), passage.passage_id)
        if key not in self._cache:
            self._cache[key] = self._score_fn(requirement, passage)
        return self._cache[key]


class EvidenceCoverageTracker:
    """Per-requirement evidence tracker (Phase 0 `EvidenceTracker` protocol).

    A requirement is SATISFIED once the best evidence score seen for it reaches
    `threshold`. `coverage()` is a continuous partial-credit view (each
    requirement contributes min(best_score/threshold, 1)), distinct from the
    hard-count `frac_satisfied()`.
    """

    def __init__(
        self,
        requirements: Iterable[Requirement],
        threshold: float = 0.3,
        score_fn: Optional[ScoreFn] = None,
    ):
        # threshold default (0.3) is calibrated for the QA-confidence default
        # scorer. NOTE: QA confidence is attribute-dependent -- factoid
        # attributes with a crisp value span (price, battery) score ~0.5-0.9,
        # while descriptive ones (noise_cancellation, summary) sit near the
        # floor. So a single global threshold trades wrong-attribute precision
        # against descriptive-attribute recall; pass an explicit threshold, or
        # a better-calibrated score_fn, when that tradeoff matters.
        # dedup by (entity, attribute) key, preserving insertion order
        self.requirements: list[Requirement] = []
        seen_keys: set[tuple[str, str]] = set()
        for r in requirements:
            if r.key() in seen_keys:
                continue
            seen_keys.add(r.key())
            self.requirements.append(r)

        self.threshold = threshold
        # Default to QA confidence: it is attribute-sensitive, unlike raw
        # MiniLM similarity which the entity name dominates. MiniLMScorer stays
        # available as a cheaper, rougher option for callers who want it.
        self._score_fn: ScoreFn = score_fn or QAScorer()
        self._best_score: dict[tuple[str, str], float] = {r.key(): 0.0 for r in self.requirements}
        self._best_passage_id: dict[tuple[str, str], Optional[str]] = {
            r.key(): None for r in self.requirements
        }
        self._seen_passage_ids: set[str] = set()

    def add_passage(self, passage: Passage) -> None:
        # idempotent per passage_id
        if passage.passage_id in self._seen_passage_ids:
            return
        self._seen_passage_ids.add(passage.passage_id)
        for r in self.requirements:
            score = self._score_fn(r, passage)
            if score > self._best_score[r.key()]:
                self._best_score[r.key()] = score
                self._best_passage_id[r.key()] = passage.passage_id

    def _is_satisfied(self, key: tuple[str, str]) -> bool:
        return self._best_score[key] >= self.threshold

    def is_sufficient(self) -> bool:
        return all(self._is_satisfied(r.key()) for r in self.requirements)

    def frac_satisfied(self) -> float:
        if not self.requirements:
            return 1.0
        return sum(self._is_satisfied(r.key()) for r in self.requirements) / len(self.requirements)

    def frac_remaining(self) -> float:
        return 1.0 - self.frac_satisfied()

    def coverage(self) -> float:
        if not self.requirements:
            return 1.0
        total = sum(
            min(max(self._best_score[r.key()], 0.0) / self.threshold, 1.0)
            for r in self.requirements
        )
        return total / len(self.requirements)

    def unanswered_requirements(self) -> list[Requirement]:
        return [r for r in self.requirements if not self._is_satisfied(r.key())]

    # --- introspection helpers (not part of the protocol) ---
    def best_score(self, requirement: Requirement) -> float:
        return self._best_score.get(requirement.key(), 0.0)

    def evidence_passage_id(self, requirement: Requirement) -> Optional[str]:
        return self._best_passage_id.get(requirement.key())


def gather_evidence(
    requirements: Iterable[Requirement],
    retriever,
    k: int = 5,
    threshold: float = 0.5,
    score_fn: Optional[ScoreFn] = None,
) -> EvidenceCoverageTracker:
    """Wire Phase 3 -> Phase 4: retrieve per requirement and feed the passages
    into a tracker. This is the per-requirement replacement for the old
    whole-question evidence-adding path. Needs a real retriever (and thus a
    model); the tracker's logic itself is covered model-free in the tests.
    """
    requirements = list(requirements)
    tracker = EvidenceCoverageTracker(requirements, threshold=threshold, score_fn=score_fn)
    for req in requirements:
        for scored in retriever.retrieve(req, k=k):
            tracker.add_passage(scored.passage)
    return tracker
