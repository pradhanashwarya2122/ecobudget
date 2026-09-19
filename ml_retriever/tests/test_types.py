import pytest

from ml_retriever.types import AnswerResult, Passage, Requirement


class TestRequirement:
    def test_basic_construction(self):
        r = Requirement(entity="iPhone 15", attribute="battery capacity")
        assert r.entity == "iPhone 15"
        assert r.attribute == "battery capacity"
        assert r.value is None

    def test_key_is_case_and_whitespace_insensitive(self):
        r1 = Requirement(entity=" iPhone 15 ", attribute="Battery Capacity")
        r2 = Requirement(entity="iphone 15", attribute="battery capacity")
        assert r1.key() == r2.key()

    def test_value_does_not_affect_key(self):
        r1 = Requirement(entity="iPhone 15", attribute="price", value="$799")
        r2 = Requirement(entity="iPhone 15", attribute="price", value=None)
        assert r1.key() == r2.key()

    def test_frozen_is_hashable(self):
        r = Requirement(entity="Galaxy S24", attribute="weight")
        # Must be hashable so it can be used as a dict key / set member by
        # trackers in later phases.
        {r: "ok"}


class TestPassage:
    def test_basic_construction(self):
        p = Passage(passage_id="p1", text="hello world", byte_size=11, source_url="https://example.com")
        assert p.passage_id == "p1"
        assert p.embedding is None
        assert p.metadata == {}

    def test_rejects_empty_id(self):
        with pytest.raises(ValueError):
            Passage(passage_id="", text="x", byte_size=1)

    def test_rejects_negative_byte_size(self):
        with pytest.raises(ValueError):
            Passage(passage_id="p1", text="x", byte_size=-5)

    def test_default_metadata_not_shared_between_instances(self):
        p1 = Passage(passage_id="p1", text="a", byte_size=1)
        p2 = Passage(passage_id="p2", text="b", byte_size=1)
        p1.metadata["k"] = "v"
        assert p2.metadata == {}


class TestAnswerResult:
    def test_normal_answer(self):
        r = AnswerResult(
            answer="42",
            confidence=0.9,
            used_evidence_ids=["p1", "p2"],
            abstained=False,
            generator_version="v0",
        )
        assert r.answer == "42"

    def test_abstained_result(self):
        r = AnswerResult(
            answer=None,
            confidence=0.1,
            used_evidence_ids=[],
            abstained=True,
            generator_version="v0",
        )
        assert r.abstained is True

    def test_abstained_with_answer_is_invalid(self):
        with pytest.raises(ValueError):
            AnswerResult(
                answer="should not be here",
                confidence=0.5,
                used_evidence_ids=[],
                abstained=True,
                generator_version="v0",
            )

    @pytest.mark.parametrize("bad_confidence", [-0.01, 1.01, 2.0, -5.0])
    def test_confidence_out_of_range_is_invalid(self, bad_confidence):
        with pytest.raises(ValueError):
            AnswerResult(
                answer="x",
                confidence=bad_confidence,
                used_evidence_ids=[],
                abstained=False,
                generator_version="v0",
            )
