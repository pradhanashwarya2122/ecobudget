import json
from pathlib import Path

from ml_retriever.decomposer import (
    HeuristicDecomposer,
    normalize_attribute,
    parse_requirements,
    serialize_requirements,
    snap_attribute,
)
from ml_retriever.types import Requirement

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestSerializationRoundTrip:
    def test_single_requirement(self):
        reqs = [Requirement(entity="iPhone 15", attribute="price")]
        text = serialize_requirements(reqs)
        assert parse_requirements(text) == reqs

    def test_multiple_requirements(self):
        reqs = [
            Requirement(entity="iPhone 15", attribute="price"),
            Requirement(entity="Samsung Galaxy S24", attribute="price"),
        ]
        text = serialize_requirements(reqs)
        assert parse_requirements(text) == reqs

    def test_empty_list(self):
        assert serialize_requirements([]) == ""
        assert parse_requirements("") == []

    def test_entity_names_with_spaces_and_numbers_survive(self):
        reqs = [Requirement(entity="Empire State Building", attribute="completion_year")]
        assert parse_requirements(serialize_requirements(reqs)) == reqs


class TestParseRobustness:
    """The model won't always emit well-formed output -- these lock in
    graceful-degradation behavior rather than exceptions."""

    def test_missing_field_separator_is_dropped(self):
        assert parse_requirements("this has no pipe") == []

    def test_empty_entity_or_attribute_is_dropped(self):
        assert parse_requirements("|price") == []
        assert parse_requirements("iPhone 15|") == []

    def test_extra_whitespace_is_tolerated(self):
        text = "  iPhone 15 | price   ##   Samsung Galaxy S24 | price  "
        result = parse_requirements(text)
        assert result == [
            Requirement(entity="iPhone 15", attribute="price"),
            Requirement(entity="Samsung Galaxy S24", attribute="price"),
        ]

    def test_duplicate_requirements_are_deduped(self):
        text = "iPhone 15|price ## iPhone 15|price"
        assert parse_requirements(text) == [Requirement(entity="iPhone 15", attribute="price")]

    def test_mixed_valid_and_malformed_segments(self):
        text = "iPhone 15|price ## garbage ## Samsung Galaxy S24|price"
        result = parse_requirements(text)
        assert result == [
            Requirement(entity="iPhone 15", attribute="price"),
            Requirement(entity="Samsung Galaxy S24", attribute="price"),
        ]


class TestHeuristicDecomposer:
    """Sanity checks on the rule-based baseline. Not expected to hit the
    Phase 2 >90% target -- see its docstring -- but should handle the
    simple template shapes it was built against."""

    def test_single_fact_question(self):
        d = HeuristicDecomposer(known_entities=["iPhone 15", "Samsung Galaxy S24"])
        result = d.decompose("What is the price of iPhone 15?")
        assert result == [Requirement(entity="iPhone 15", attribute="price")]

    def test_comparison_question_pairs_attribute_with_every_found_entity(self):
        d = HeuristicDecomposer(known_entities=["iPhone 15", "Samsung Galaxy S24"])
        result = d.decompose("Compare iPhone 15 and Samsung Galaxy S24 on price.")
        assert {r.key() for r in result} == {
            ("iphone 15", "price"),
            ("samsung galaxy s24", "price"),
        }

    def test_unknown_entity_yields_no_requirements(self):
        d = HeuristicDecomposer(known_entities=["iPhone 15"])
        assert d.decompose("What is the price of a Google Pixel 9?") == []

    def test_no_recognizable_attribute_yields_no_requirements(self):
        d = HeuristicDecomposer(known_entities=["iPhone 15"])
        assert d.decompose("Tell me something interesting about iPhone 15.") == []

    def test_longer_entity_name_preferred_over_substring(self):
        # Gazetteer order shouldn't matter; "Empire State Building" must
        # be matched whole, not accidentally missed.
        d = HeuristicDecomposer(known_entities=["Empire State Building", "Burj Khalifa"])
        result = d.decompose("How tall is the Empire State Building?")
        assert result == [Requirement(entity="Empire State Building", attribute="height")]

    def test_procedure_convention(self):
        d = HeuristicDecomposer(known_entities=["tire"])
        assert d.decompose("How do I change a flat car tire?") == [
            Requirement(entity="tire", attribute="procedure")
        ]

    def test_narrative_convention(self):
        d = HeuristicDecomposer(known_entities=["Mamma Mia"])
        assert d.decompose("Summarize the premise of Mamma Mia.") == [
            Requirement(entity="Mamma Mia", attribute="summary")
        ]


class TestHeuristicBaselineOnValSplit:
    """Runs the heuristic against the real val split as a smoke test for
    the eval pipeline (mirrors what scripts/eval_decomposer.py does).
    Exact numbers aren't asserted -- only that it runs and produces a
    plausible score -- since the vocabulary/gazetteer heuristic is
    expected to be imperfect (that's the point: Phase 2 replaces it)."""

    def test_runs_and_scores_reasonably(self):
        with open(DATA_DIR / "decomposer_val.jsonl") as f:
            pairs = [json.loads(line) for line in f]
        with open(DATA_DIR / "tasks.json") as f:
            tasks = json.load(f)

        known_entities = sorted(
            {r["entity"] for t in tasks for r in t["decomposed_requirements"]},
            key=len,
            reverse=True,
        )
        d = HeuristicDecomposer(known_entities)

        exact_matches = 0
        for pair in pairs:
            gold = parse_requirements(pair["target"])
            pred = d.decompose(pair["question"])
            if {r.key() for r in pred} == {r.key() for r in gold}:
                exact_matches += 1

        accuracy = exact_matches / len(pairs)
        # Not a strict regression bound -- just confirms the heuristic
        # baseline is doing real work (not 0%) and leaves clear room for
        # the ML model to beat it (not near 100%).
        assert 0.3 < accuracy < 0.95


class TestAttributeSnapping:
    VOCAB = {
        normalize_attribute(v): v
        for v in ["noise_cancellation", "first_ascent_year", "price", "battery_life", "display_refresh_rate"]
    }

    def test_exact_normalized_match_returns_canonical(self):
        # word order / separators differ but it's the same attribute
        assert snap_attribute("noise cancellation", self.VOCAB) == "noise_cancellation"

    def test_typo_snaps_to_nearest_slug(self):
        assert snap_attribute("noises_cancelation", self.VOCAB) == "noise_cancellation"

    def test_truncation_snaps_to_full_slug(self):
        assert snap_attribute("first_ascent", self.VOCAB) == "first_ascent_year"

    def test_far_off_attribute_is_left_unchanged(self):
        # nothing in vocab is close -> don't force-snap to a wrong slug
        assert snap_attribute("elevation", self.VOCAB) == "elevation"

    def test_empty_vocab_is_identity(self):
        assert snap_attribute("whatever", {}) == "whatever"

    def test_distinct_slugs_do_not_collapse(self):
        assert snap_attribute("battery_life", self.VOCAB) == "battery_life"
