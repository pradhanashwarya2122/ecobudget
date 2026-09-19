from backend.adaptive_controller import AdaptiveController
from backend.adaptive_controller.simulator import SimulatedEpisode
from backend.adaptive_controller.sample_tasks import easy_task, medium_task, hard_task


def test_easy_task_stops_quickly():
    result = SimulatedEpisode(easy_task()).run(AdaptiveController())
    assert result["stopped_at_step"] <= 2


def test_hard_task_requires_more_steps():
    result = SimulatedEpisode(hard_task()).run(AdaptiveController())
    assert result["stopped_at_step"] >= 4


def test_difficulty_ordering_holds():
    controller = AdaptiveController()
    easy = SimulatedEpisode(easy_task()).run(controller)
    medium = SimulatedEpisode(medium_task()).run(controller)
    hard = SimulatedEpisode(hard_task()).run(controller)
    assert easy["stopped_at_step"] <= medium["stopped_at_step"] <= hard["stopped_at_step"]