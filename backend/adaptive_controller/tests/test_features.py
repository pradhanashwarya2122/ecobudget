from backend.adaptive_controller.features import ContextVectorBuilder, FEATURE_NAMES
from backend.adaptive_controller.context import Context


def make_context(**overrides):
    base = dict(
        question="q", data_used=1000, resources_seen=1, relevance_score=0.5,
        answer_confidence=0.5, evidence_coverage=0.5, steps_taken=1,
        estimated_remaining_information=3.0,
    )
    base.update(overrides)
    return Context(**base)


def test_vector_length_matches_feature_names():
    builder = ContextVectorBuilder()
    vec = builder.build(make_context())
    assert len(vec) == len(FEATURE_NAMES)


def test_first_call_has_zero_deltas():
    builder = ContextVectorBuilder()
    vec = builder.build(make_context(relevance_score=0.7, answer_confidence=0.6))
    relevance_delta = vec[FEATURE_NAMES.index("relevance_delta")]
    confidence_delta = vec[FEATURE_NAMES.index("confidence_delta")]
    assert relevance_delta == 0.0
    assert confidence_delta == 0.0


def test_second_call_computes_delta():
    builder = ContextVectorBuilder()
    builder.build(make_context(relevance_score=0.5, answer_confidence=0.4))
    vec = builder.build(make_context(relevance_score=0.8, answer_confidence=0.3))
    relevance_delta = vec[FEATURE_NAMES.index("relevance_delta")]
    confidence_delta = vec[FEATURE_NAMES.index("confidence_delta")]
    assert abs(relevance_delta - 0.3) < 1e-9
    assert abs(confidence_delta - (-0.1)) < 1e-9


def test_reset_clears_deltas():
    builder = ContextVectorBuilder()
    builder.build(make_context(relevance_score=0.9))
    builder.reset()
    vec = builder.build(make_context(relevance_score=0.2))
    relevance_delta = vec[FEATURE_NAMES.index("relevance_delta")]
    assert relevance_delta == 0.0


def test_no_ground_truth_field_in_output():
    # Context has no ground-truth field at all, but this test guards
    # against someone adding one later and wiring it in by accident.
    builder = ContextVectorBuilder()
    context = make_context()
    assert not hasattr(context, "ground_truth_correct_at_step")
    builder.build(context)