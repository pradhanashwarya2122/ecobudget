"""Guards for the lenient eval metric in scripts/eval_decomposer.py.

The normalized-attribute match gives partial credit when a prediction means
the same thing as the gold slug but spells it differently. That is only safe
if normalization never maps two *different* gold slugs onto one canonical
form -- otherwise attribute_f1 / exact_match_normalized would be silently
inflated. These tests assert that invariant on the real attribute vocabulary.
"""

import importlib.util
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

# eval_decomposer lives in scripts/ (not an installed package); load directly.
_spec = importlib.util.spec_from_file_location(
    "eval_decomposer", SCRIPTS / "eval_decomposer.py"
)
eval_decomposer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_decomposer)
normalize_attribute = eval_decomposer.normalize_attribute


def _seed_attributes() -> set[str]:
    seeds = json.loads((DATA / "tasks_seed.json").read_text(encoding="utf-8"))
    return {
        r["attribute"]
        for t in seeds
        for r in t["decomposed_requirements"]
    }


def test_normalization_has_no_collisions_on_real_vocab():
    """Distinct raw slugs must stay distinct after normalization.

    If this fails, normalize_attribute is merging two different attributes
    and the metric is inflating F1 -- fix the normalizer (shrink the
    stopword set), do not weaken this test.
    """
    raw = _seed_attributes()
    canon = {normalize_attribute(a) for a in raw}
    assert len(canon) == len(raw), (
        "normalize_attribute collision: "
        f"{len(raw)} raw slugs -> {len(canon)} canonical forms"
    )


def test_semantically_equal_slugs_match():
    # The concrete case that motivated normalization.
    assert normalize_attribute("year_of_first_ascent") == normalize_attribute(
        "first_ascent_year"
    )


def test_different_slugs_do_not_match():
    assert normalize_attribute("battery") != normalize_attribute("battery_life")
    assert normalize_attribute("price") != normalize_attribute("price_per_night")
