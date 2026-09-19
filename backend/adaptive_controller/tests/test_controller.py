from backend.adaptive_controller import AdaptiveController, Context, Action


def make_context(**overrides):
    base = dict(
        question="What is the capital of France?",
        data_used=0,
        resources_seen=0,
        relevance_score=0.0,
        answer_confidence=0.0,
        evidence_coverage=0.0,
        steps_taken=0,
    )
    base.update(overrides)
    return Context(**base)


def test_stops_when_confident_and_covered():
    controller = AdaptiveController()
    ctx = make_context(answer_confidence=0.9, evidence_coverage=0.85, steps_taken=2)
    assert controller.decide(ctx).action == Action.STOP


def test_retrieves_when_uncertain():
    controller = AdaptiveController()
    ctx = make_context(answer_confidence=0.2, evidence_coverage=0.1, steps_taken=1)
    assert controller.decide(ctx).action == Action.RETRIEVE


def test_stops_at_max_steps_even_if_uncertain():
    controller = AdaptiveController(max_steps=3)
    ctx = make_context(answer_confidence=0.1, evidence_coverage=0.1, steps_taken=3)
    assert controller.decide(ctx).action == Action.STOP