"""Model-free tests for Phase 3 retrieval ranking math.

Like the rest of this suite, these use dummy embeddings and never load a
model -- they cover rank_passages / requirement_to_query, the parts that
don't need sentence-transformers. RequirementRetriever's query encoding is
exercised by scripts/eval_retriever.py on a machine with the model.
"""

import math

from ml_retriever.retriever import (
    EntityAwareRetriever,
    attribute_to_query,
    passage_matches_entity,
    rank_passages,
    requirement_to_query,
)
from ml_retriever.types import Passage, Requirement


def _p(pid, emb, byte_size=100, text="x", metadata=None):
    return Passage(passage_id=pid, text=text, byte_size=byte_size, embedding=emb,
                   metadata=metadata or {})


class TestRequirementToQuery:
    def test_deslugs_attribute(self):
        q = requirement_to_query(Requirement(entity="iPhone 15", attribute="display_refresh_rate"))
        assert q == "iPhone 15 display refresh rate"


class TestRankPassages:
    def test_orders_by_cosine_similarity(self):
        # query points along x; p_near is closer in direction than p_far
        passages = [
            _p("far", [0.0, 1.0]),
            _p("near", [1.0, 0.1]),
            _p("mid", [1.0, 1.0]),
        ]
        ranked = rank_passages([1.0, 0.0], passages, k=3)
        assert [r.passage.passage_id for r in ranked] == ["near", "mid", "far"]

    def test_top_k_truncates(self):
        passages = [_p(f"p{i}", [1.0, float(i)]) for i in range(10)]
        ranked = rank_passages([1.0, 0.0], passages, k=3)
        assert len(ranked) == 3

    def test_passages_without_embedding_are_skipped(self):
        passages = [_p("has", [1.0, 0.0]), _p("none", None)]
        ranked = rank_passages([1.0, 0.0], passages, k=5)
        assert [r.passage.passage_id for r in ranked] == ["has"]

    def test_similarity_is_cosine_not_dot(self):
        # same direction, different magnitude -> cosine 1.0 for both
        ranked = rank_passages([1.0, 0.0], [_p("a", [5.0, 0.0])], k=1)
        assert math.isclose(ranked[0].similarity, 1.0, abs_tol=1e-6)

    def test_value_per_byte_prefers_smaller_when_similarity_ties(self):
        # identical direction (sim ties); smaller byte_size should win on vpb
        passages = [_p("big", [1.0, 0.0], byte_size=1000), _p("small", [1.0, 0.0], byte_size=50)]
        ranked = rank_passages([1.0, 0.0], passages, k=2, value_per_byte=True)
        assert ranked[0].passage.passage_id == "small"

    def test_plain_similarity_does_not_depend_on_byte_size(self):
        passages = [_p("big", [1.0, 0.0], byte_size=1000), _p("small", [1.0, 0.0], byte_size=50)]
        ranked = rank_passages([1.0, 0.0], passages, k=2, value_per_byte=False)
        # tie on similarity -> deterministic tie-break by passage_id desc
        assert {r.passage.passage_id for r in ranked} == {"big", "small"}
        assert all(math.isclose(r.similarity, 1.0, abs_tol=1e-6) for r in ranked)


class TestAttributeToQuery:
    def test_deslugs_attribute_only(self):
        # Entity name is dropped -- the entity-aware retriever has already gated
        # to one entity, so only the attribute should drive ranking.
        q = attribute_to_query(Requirement(entity="iPhone 15", attribute="display_refresh_rate"))
        assert q == "display refresh rate"


class TestPassageMatchesEntity:
    def test_exact_metadata_match(self):
        p = _p("a", [1.0, 0.0], metadata={"entity": "Samsung Galaxy S24"})
        assert passage_matches_entity(p, "samsung galaxy s24")

    def test_near_twin_entities_do_not_collide(self):
        # The exact collision that caused entity-dominance misses: a bare 'S24'
        # requirement must NOT match an 'S24 Ultra' passage and vice versa.
        base = _p("a", [1.0, 0.0], metadata={"entity": "Samsung Galaxy S24"})
        ultra = _p("b", [1.0, 0.0], metadata={"entity": "Samsung Galaxy S24 Ultra"})
        assert passage_matches_entity(base, "Samsung Galaxy S24")
        assert not passage_matches_entity(ultra, "Samsung Galaxy S24")
        assert passage_matches_entity(ultra, "Samsung Galaxy S24 Ultra")
        assert not passage_matches_entity(base, "Samsung Galaxy S24 Ultra")

    def test_text_fallback_when_no_metadata_entity(self):
        # Live-fetched passage: no metadata entity, match on word-boundary text.
        p = _p("a", [1.0, 0.0], text="The Eiffel Tower is 330 metres tall.")
        assert passage_matches_entity(p, "Eiffel Tower")
        assert not passage_matches_entity(p, "Empire State Building")

    def test_text_fallback_respects_word_boundaries(self):
        p = _p("a", [1.0, 0.0], text="The S240 router ships next year.")
        assert not passage_matches_entity(p, "S24")


class TestEntityAwareRetriever:
    def _corpus(self):
        # Two entities x two attributes. Embeddings are chosen so that WITHIN an
        # entity the attribute axis discriminates, and so that a plain similarity
        # search would be tempted by the wrong entity (both share the x axis).
        return [
            _p("s24_price",  [1.0, 1.0, 0.0], metadata={"entity": "S24", "attribute": "price"}),
            _p("s24_batt",   [1.0, 0.0, 1.0], metadata={"entity": "S24", "attribute": "battery"}),
            _p("ultra_price",[1.0, 1.0, 0.0], metadata={"entity": "S24 Ultra", "attribute": "price"}),
            _p("ultra_batt", [1.0, 0.0, 1.0], metadata={"entity": "S24 Ultra", "attribute": "battery"}),
        ]

    def test_gate_restricts_to_entity(self):
        r = EntityAwareRetriever(self._corpus())
        pool = r.gate(Requirement(entity="S24", attribute="price"))
        assert {p.passage_id for p in pool} == {"s24_price", "s24_batt"}

    def test_unknown_entity_falls_back_to_full_corpus(self, monkeypatch):
        r = EntityAwareRetriever(self._corpus())
        r._encoder = object()
        monkeypatch.setattr(r, "_encode", lambda text: [1.0, 0.0, 0.0])
        # No passage matches this entity -> gate empty -> full-corpus fallback,
        # so retrieval still returns something rather than nothing.
        out = r.retrieve(Requirement(entity="Nokia 3310", attribute="price"), k=4)
        assert len(out) == 4

    def test_gate_then_attribute_rank(self, monkeypatch):
        # Stub the encoder so the test is model-free: "price" -> price axis.
        r = EntityAwareRetriever(self._corpus())
        r._encoder = object()
        monkeypatch.setattr(r, "_encode", lambda text: [0.0, 1.0, 0.0])  # price axis
        out = r.retrieve(Requirement(entity="S24", attribute="price"), k=2)
        ids = [s.passage.passage_id for s in out]
        # top-1 must be S24's price passage: not the Ultra's (entity gate) and
        # not S24's battery (attribute rank).
        assert ids[0] == "s24_price"
        assert "ultra_price" not in ids and "ultra_batt" not in ids
