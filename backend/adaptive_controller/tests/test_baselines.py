from backend.adaptive_controller.baselines import FixedBudgetController, FixedThresholdController
from backend.adaptive_controller.simulator import SimulatedEpisode
from backend.adaptive_controller.sample_tasks import easy_task, hard_task


def test_fixed_budget_stops_at_budget():
    result = SimulatedEpisode(hard_task()).run(FixedBudgetController(budget_bytes=5000))
    assert result["final_bytes_used"] >= 5000
    assert result["final_bytes_used"] < 5000 + 2600  # doesn't overshoot by more than one step


def test_fixed_budget_zero_stops_immediately():
    result = SimulatedEpisode(easy_task()).run(FixedBudgetController(budget_bytes=0))
    assert result["stopped_at_step"] == 0


def test_fixed_threshold_stops_on_easy_task_quickly():
    result = SimulatedEpisode(easy_task()).run(FixedThresholdController(confidence_threshold=0.9))
    assert result["stopped_at_step"] <= 2


def test_fixed_threshold_requires_more_steps_on_hard_task():
    easy = SimulatedEpisode(easy_task()).run(FixedThresholdController(confidence_threshold=0.75))
    hard = SimulatedEpisode(hard_task()).run(FixedThresholdController(confidence_threshold=0.75))
    assert hard["stopped_at_step"] >= easy["stopped_at_step"]