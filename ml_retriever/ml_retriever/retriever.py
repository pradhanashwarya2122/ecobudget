"""Phase 3: Requirement-aware retrieval.

Retrieves passages PER REQUIREMENT, not per whole question. A whole-question
query ("Compare the iPhone 15 and Galaxy S24 on price and battery") blurs
several information needs into one embedding and tends to surface passages for
only the dominant one; decomposing into Requirements first and retrieving for
each ("iPhone 15 price", "Galaxy S24 battery", ...) gives every need its own
ranked list. This is the Phase 3 deliverable that `scripts/eval_retriever.py`
benchmarks against the whole-question baseline via recall@5.

Design, mirroring decomposer.py:
  - The ranking math (`rank_passages`) is pure and model-free, so the test
    suite exercises it with dummy embeddings and no model download.
  - `RequirementRetriever` lazily loads sentence-transformers (and, optionally,
    a cross-encoder re-ranker) only when it actually has to embed a query.
    Passages are assumed pre-embedded (see scripts/embed_corpus.py), so we
    never re-encode the corpus here.

Reuse note: the similarity primitive is the same MiniLM cosine used by
backend/scorer.py's `compute_utility`. The value-per-byte score here is
`similarity / log(byte_size)` -- deliberately NOT backend's `utility / bytes`
(see plan.md Phase 3): dividing by raw bytes collapses to near-zero for any
real passage and over-rewards tiny fragments; dividing by log(bytes) keeps
similarity dominant while gently preferring the more compact of two
comparably-relevant passages.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .types import Passage, Requirement

# Floor for log(byte_size): a passage shorter than this many bytes shouldn't
# get an outsized value-per-byte boost from a near-zero/negative denominator.
_MIN_BYTES_FOR_LOG = 10


def requirement_to_query(requirement: Requirement) -> str:
    """The query string for a requirement: entity plus a de-slugged attribute.

    "display_refresh_rate" -> "display refresh rate", so the MiniLM query
    reads like natural text rather than a snake_case token."""
    attribute = requirement.attribute.replace("_", " ").strip()
    return f"{requirement.entity} {attribute}".strip()


def attribute_to_query(requirement: Requirement) -> str:
    """The attribute phrase alone, de-slugged. Used by the entity-aware
    retriever, which has already gated to one entity's passages and so wants
    the attribute -- not the (dominant) entity name -- to do the ranking."""
    return requirement.attribute.replace("_", " ").strip()


def _normalize_entity(name: str) -> str:
    """Case/space-insensitive entity key. Deliberately NOT stemming or dropping
    tokens: 'Samsung Galaxy S24' and 'Samsung Galaxy S24 Ultra' must stay
    distinct (the exact collision that caused the entity-dominance misses)."""
    return re.sub(r"\s+", " ", name.strip().lower())


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


@dataclass
class ScoredPassage:
    passage: Passage
    similarity: float  # raw cosine similarity in [-1, 1]
    score: float       # ranking score actually sorted on (similarity, or value-per-byte)


def rank_passages(
    query_embedding,
    passages: list[Passage],
    k: int = 5,
    value_per_byte: bool = False,
) -> list[ScoredPassage]:
    """Rank pre-embedded passages against a query embedding by cosine
    similarity (or similarity / log(byte_size) when `value_per_byte`).

    Pure and model-free: both arguments are plain data, so this is the part
    the test suite covers directly. Passages without an embedding are skipped.
    """
    q = _unit(np.asarray(query_embedding, dtype=float))
    scored: list[ScoredPassage] = []
    for p in passages:
        if p.embedding is None:
            continue
        sim = float(np.dot(q, _unit(np.asarray(p.embedding, dtype=float))))
        if value_per_byte:
            score = sim / math.log(max(p.byte_size, _MIN_BYTES_FOR_LOG))
        else:
            score = sim
        scored.append(ScoredPassage(passage=p, similarity=sim, score=score))
    # Stable tie-break on passage_id keeps ordering deterministic.
    scored.sort(key=lambda s: (s.score, s.passage.passage_id), reverse=True)
    return scored[:k]


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, requirement: Requirement, k: int = 5) -> list[ScoredPassage]:
        ...


class RequirementRetriever:
    """Bi-encoder retrieval over a pre-embedded corpus, with an optional
    cross-encoder re-rank.

    Requires `sentence-transformers` to embed queries; it is imported lazily
    on first use (like TaskDecomposer) so importing this module, and the
    model-free unit tests, need no model download.
    """

    def __init__(
        self,
        corpus: list[Passage],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        value_per_byte: bool = False,
        use_cross_encoder: bool = False,
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_pool: int = 20,
    ):
        self.corpus = corpus
        self.model_name = model_name
        self.value_per_byte = value_per_byte
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_name = cross_encoder_name
        self.rerank_pool = rerank_pool
        self._encoder = None
        self._cross_encoder = None

    def _encode(self, text: str) -> np.ndarray:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "RequirementRetriever needs 'sentence-transformers' to embed "
                    "queries. Install it, or call rank_passages() directly with "
                    "precomputed embeddings."
                ) from e
            self._encoder = SentenceTransformer(self.model_name)
        return np.asarray(self._encoder.encode(text), dtype=float)

    def _rerank(self, query: str, scored: list[ScoredPassage], k: int) -> list[ScoredPassage]:
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.cross_encoder_name)
        pairs = [(query, s.passage.text) for s in scored]
        ce_scores = self._cross_encoder.predict(pairs)
        for s, ce in zip(scored, ce_scores):
            s.score = float(ce)
        scored.sort(key=lambda s: (s.score, s.passage.passage_id), reverse=True)
        return scored[:k]

    def retrieve(self, requirement: Requirement, k: int = 5) -> list[ScoredPassage]:
        query = requirement_to_query(requirement)
        query_embedding = self._encode(query)
        if self.use_cross_encoder:
            pool = rank_passages(
                query_embedding, self.corpus, k=max(self.rerank_pool, k),
                value_per_byte=self.value_per_byte,
            )
            return self._rerank(query, pool, k)
        return rank_passages(query_embedding, self.corpus, k=k, value_per_byte=self.value_per_byte)


def _entity_tokens_present(entity_norm: str, text: str) -> bool:
    """Live-fetch fallback entity match: is the (normalized) entity string
    present in the passage text on word boundaries? Word boundaries stop
    'S24' from matching inside 'S240' and keep the match order-sensitive
    enough to be a real signal, while still tolerating surrounding prose."""
    if not entity_norm:
        return False
    pattern = r"\b" + re.escape(entity_norm) + r"\b"
    return re.search(pattern, text.lower()) is not None


def passage_matches_entity(passage: Passage, entity: str) -> bool:
    """Does this passage belong to the requirement's entity?

    Offline (our corpus) every passage carries a gold `metadata['entity']`, so
    we use an EXACT normalized match -- this is what cleanly separates near-twin
    entities ('S24' vs 'S24 Ultra', 'Pixel 8' vs 'Pixel 9') that pure embedding
    similarity conflates. When metadata is absent (a live-fetched web passage),
    fall back to a word-boundary text mention. Pure and model-free, so tests
    cover it directly."""
    want = _normalize_entity(entity)
    meta_entity = passage.metadata.get("entity")
    if meta_entity:
        return _normalize_entity(meta_entity) == want
    return _entity_tokens_present(want, passage.text)


class EntityAwareRetriever:
    """Phase D: two-stage retrieval that fixes the entity-dominance and
    attribute-confusion failures of the plain bi-encoder.

    Stage 1 (entity gate): keep only passages that belong to the requirement's
    entity (`passage_matches_entity`). This removes near-twin entities that the
    embedding conflates -- retrieving 'Samsung Galaxy S24 Ultra' for a 'Samsung
    Galaxy S24' requirement, or an 'Oberoi ... Jaipur' hotel passage for a
    'Jaipur' requirement.

    Stage 2 (attribute rank): rank the gated passages by similarity to the
    ATTRIBUTE phrase alone (`attribute_to_query`). Within one entity's passages
    the entity name is a constant, so it can no longer dominate the cosine and
    the attribute becomes the discriminator -- fixing cases like Empire State
    Building height/floors/architect all collapsing onto 'floors'.

    Graceful degradation: if the gate is empty (an entity with no matching
    passage -- e.g. a live, never-seen entity) it falls back to the plain
    bi-encoder over the whole corpus, so recall is never worse than the
    baseline for unknown entities. Same `Retriever` protocol, so it is a
    drop-in for `RequirementRetriever` in the pipeline.
    """

    def __init__(
        self,
        corpus: list[Passage],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        value_per_byte: bool = False,
        rank_on: str = "attribute",  # "attribute" (gated) or "requirement" (full)
    ):
        self.corpus = corpus
        self.model_name = model_name
        self.value_per_byte = value_per_byte
        self.rank_on = rank_on
        self._encoder = None

    def _encode(self, text: str) -> np.ndarray:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "EntityAwareRetriever needs 'sentence-transformers' to embed "
                    "queries. Install it, or call rank_passages() directly with "
                    "precomputed embeddings."
                ) from e
            self._encoder = SentenceTransformer(self.model_name)
        return np.asarray(self._encoder.encode(text), dtype=float)

    def gate(self, requirement: Requirement) -> list[Passage]:
        """The entity-matched candidate pool (Stage 1), exposed for eval/tests."""
        return [p for p in self.corpus if passage_matches_entity(p, requirement.entity)]

    def retrieve(self, requirement: Requirement, k: int = 5) -> list[ScoredPassage]:
        pool = self.gate(requirement)
        if not pool:
            # Unknown entity: degrade to plain bi-encoder over the full corpus.
            pool = self.corpus
            query = requirement_to_query(requirement)
        elif self.rank_on == "requirement":
            query = requirement_to_query(requirement)
        else:
            query = attribute_to_query(requirement)
        query_embedding = self._encode(query)
        return rank_passages(query_embedding, pool, k=k, value_per_byte=self.value_per_byte)
