import numpy as np

from backend.adaptive_controller.bandit import LinUCBBandit
from backend.adaptive_controller.actions import Action


def test_untrained_bandit_returns_valid_action():
    bandit = LinUCBBandit(n_features=9, alpha=1.0)
    decision = bandit.choose_action([0.0] * 9)
    assert decision.action in (Action.STOP, Action.RETRIEVE)


def test_update_changes_future_scores():
    bandit = LinUCBBandit(n_features=9, alpha=0.0)  # alpha=0 removes exploration noise for this test
    x = [1.0] * 9
    before = bandit._score(Action.STOP, np.array(x))
    bandit.update(x, Action.STOP, reward=10.0)
    after = bandit._score(Action.STOP, np.array(x))
    assert after > before


def test_bandit_learns_to_prefer_rewarded_arm():
    bandit = LinUCBBandit(n_features=9, alpha=0.1)
    x = [0.5] * 9
    # Repeatedly reward STOP, punish RETRIEVE, for the same context
    for _ in range(20):
        bandit.update(x, Action.STOP, reward=5.0)
        bandit.update(x, Action.RETRIEVE, reward=-5.0)
    decision = bandit.choose_action(x)
    assert decision.action == Action.STOP


def test_alpha_zero_is_deterministic_given_same_history():
    bandit = LinUCBBandit(n_features=9, alpha=0.0)
    x = [0.3] * 9
    bandit.update(x, Action.RETRIEVE, reward=2.0)
    d1 = bandit.choose_action(x)
    d2 = bandit.choose_action(x)
    assert d1.action == d2.action


def test_confidence_is_bounded():
    bandit = LinUCBBandit(n_features=9, alpha=1.0)
    decision = bandit.choose_action([2.0] * 9)
    assert 0.0 <= decision.confidence <= 1.0