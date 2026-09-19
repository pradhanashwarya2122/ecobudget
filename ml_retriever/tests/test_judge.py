import pytest

from ml_retriever.judge import (
    JUDGES,
    NARRATIVE_F1_THRESHOLD,
    PROCEDURE_STEP_F1_THRESHOLD,
    judge_task,
    token_f1,
)


@pytest.mark.parametrize(
    "answer_type",
    ["single_fact", "yes_no", "list", "comparison", "multi_part"],
)
def test_each_structured_judge_has_clear_pass_and_fail(answer_type):
    task = {"answer_type": answer_type, "expected_answer": "New Delhi"}
    assert JUDGES[answer_type]("The answer is New Delhi.", task) == (True, 1.0)
    assert JUDGES[answer_type]("The answer is Tokyo.", task) == (False, 0.0)


def test_procedure_judge_has_clear_pass_and_fail():
    task = {
        "answer_type": "procedure",
        "ground_truth": {
            "required_facts": ["loosen lug nuts", "raise vehicle"],
            "match_threshold": 1.0,
        },
    }
    assert judge_task("Loosen lug nuts then raise vehicle", task) == (True, 1.0)
    assert judge_task("Raise vehicle", task) == (False, 0.5)


def test_narrative_judge_has_clear_pass_and_fail():
    task = {
        "answer_type": "narrative",
        "ground_truth": "Sophie invites three possible fathers to her wedding.",
    }
    success, score = judge_task("Sophie invites three possible fathers to her wedding.", task)
    assert success is True
    assert score == 1.0
    assert judge_task("A phone comparison.", task) == (False, 0.0)


def test_narrative_f1_boundary_is_pinned():
    gold = "one two three four five six seven eight nine ten"
    just_above = "one two three extra"
    just_below = "one two three extra another final"

    assert token_f1(just_above, gold) > NARRATIVE_F1_THRESHOLD
    assert judge_task(
        just_above, {"answer_type": "narrative", "ground_truth": gold}
    )[0] is True
    assert token_f1(just_below, gold) < NARRATIVE_F1_THRESHOLD
    assert judge_task(
        just_below, {"answer_type": "narrative", "ground_truth": gold}
    )[0] is False


def test_procedure_step_f1_boundary_is_pinned():
    step = "one two three four five six seven eight nine ten"
    just_above = "one two three extra"
    just_below = "one two three extra another final"
    task = {
        "answer_type": "procedure",
        "ground_truth": {"required_facts": [step], "match_threshold": 1.0},
    }

    assert token_f1(just_above, step) > PROCEDURE_STEP_F1_THRESHOLD
    assert judge_task(just_above, task) == (True, 1.0)
    assert token_f1(just_below, step) < PROCEDURE_STEP_F1_THRESHOLD
    assert judge_task(just_below, task) == (False, 0.0)


class TestFormattingNormalizer:
    """Judge normalizer covers ONLY formatting variations (symmetric), not synonyms."""

    def _task(self, facts):
        return {"answer_type": "comparison",
                "ground_truth": {"required_facts": facts, "match_threshold": 1.0}}

    def test_unit_spacing_and_case(self):
        ok, frac = judge_task("Samsung 120 Hz, iPhone 60 hz", self._task(["120Hz", "60Hz"]))
        assert ok and frac == 1.0

    def test_currency_and_commas(self):
        ok, _ = judge_task("it costs 1099 dollars", self._task(["$1,099"]))
        assert ok

    def test_trailing_decimal(self):
        ok, _ = judge_task("weighs 10 ounces", self._task(["10.0 ounces"]))
        assert ok

    def test_article_dropping(self):
        ok, _ = judge_task("iPhone 15", self._task(["the iPhone 15"]))
        assert ok

    def test_fact_fraction_is_continuous(self):
        # one of two facts present -> 0.5, and (threshold 1.0) not a success
        ok, frac = judge_task("only 120Hz here", self._task(["120Hz", "60Hz"]))
        assert not ok and frac == 0.5

    def test_no_synonym_matching(self):
        # "oz" is a synonym of "ounces", NOT a formatting variant -> must NOT match
        ok, _ = judge_task("10 oz", self._task(["10 ounces"]))
        assert not ok
