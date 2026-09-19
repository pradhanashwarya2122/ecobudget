"""Phase 6 answer-generator tests: abstain logic and result shape, model-free."""

from ml_retriever.answer import (
    EvidenceAnswerGenerator,
    ExtractiveQAAnswerGenerator,
    GenerativeAnswerGenerator,
)
from ml_retriever.types import Passage, Requirement


def _p(pid, text, b=100):
    return Passage(passage_id=pid, text=text, byte_size=b)


class TestGenerative:
    def test_abstains_with_no_evidence(self):
        g = GenerativeAnswerGenerator(generate_fn=lambda prompt: "should not be called")
        r = g.generate("Q?", [])
        assert r.abstained and r.answer is None and r.used_evidence_ids == []

    def test_answers_from_evidence(self):
        g = GenerativeAnswerGenerator(generate_fn=lambda prompt: "Samsung S24 (120Hz vs 60Hz)")
        r = g.generate("Which has a higher refresh rate?", [_p("p1", "S24 120Hz"), _p("p2", "iPhone 60Hz")])
        assert not r.abstained
        assert r.answer == "Samsung S24 (120Hz vs 60Hz)"
        assert r.used_evidence_ids == ["p1", "p2"]

    def test_empty_generation_abstains(self):
        g = GenerativeAnswerGenerator(generate_fn=lambda prompt: "   ")
        r = g.generate("Q?", [_p("p1", "evidence")])
        assert r.abstained and r.answer is None

    def test_prompt_includes_question_and_evidence(self):
        seen = {}
        def capture(prompt):
            seen["prompt"] = prompt
            return "ans"
        GenerativeAnswerGenerator(generate_fn=capture).generate("How much?", [_p("p1", "it costs 799")])
        assert "How much?" in seen["prompt"] and "799" in seen["prompt"]

    def test_version_records_checkpoint(self):
        g = GenerativeAnswerGenerator(model_name_or_path="models/answer-small",
                                      adapter_path="models/answer-small", generate_fn=lambda p: "x")
        assert "models/answer-small" in g.version and "lora" in g.version


class TestExtractiveQA:
    class _FakeScorer:
        def __init__(self, span, conf):
            self._span, self._conf = span, conf
        def extract_answer(self, req, passage):
            return self._span
        def __call__(self, req, passage):
            return self._conf

    def test_abstains_below_confidence(self):
        g = ExtractiveQAAnswerGenerator(Requirement("iPhone 15", "price"),
                                        scorer=self._FakeScorer("$799", 0.1), min_confidence=0.2)
        r = g.generate("price?", [_p("p1", "iPhone 15 $799")])
        assert r.abstained and r.answer is None

    def test_answers_above_confidence(self):
        g = ExtractiveQAAnswerGenerator(Requirement("iPhone 15", "price"),
                                        scorer=self._FakeScorer("$799", 0.8), min_confidence=0.2)
        r = g.generate("price?", [_p("p1", "iPhone 15 $799")])
        assert not r.abstained and r.answer == "$799" and r.used_evidence_ids == ["p1"]


class TestEvidenceConcat:
    def test_abstains_with_no_evidence(self):
        r = EvidenceAnswerGenerator().generate("Q?", [])
        assert r.abstained and r.answer is None
