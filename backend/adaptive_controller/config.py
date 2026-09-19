"""
Reward coefficients for the adaptive controller.

Calibration: rewards must make correct retrieval-then-stop the
dominant strategy across all difficulty levels.

For a hard task (6 retrieval steps to gt, then STOP):
  correct path: 6 × (-0.5) + CORRECT_STOP_REWARD = -3 + 10 = +7
  stop at step 0: -WRONG_STOP_PENALTY = -15
  → correct path wins by 22 points

WRONG_STOP_PENALTY must exceed the maximum possible retrieval cost
so that no amount of step-cost accumulation makes early stopping
attractive. Max retrieval cost = 7 steps × 0.5 + byte costs ≈ 4.
Penalty of 15 safely exceeds this.
"""

CORRECT_STOP_REWARD       = 10.0   # reward for stopping once evidence is sufficient
WRONG_STOP_PENALTY        = 15.0   # penalty for stopping before evidence is sufficient
                                   # must exceed max possible retrieval cost so early
                                   # stopping is never the locally optimal choice
BYTE_COST_PER_KB          =  0.01  # cost per KB fetched
RETRIEVAL_STEP_COST       =  0.5   # fixed cost per retrieval step
UNNECESSARY_RETRIEVAL_PENALTY = 8.0  # extra cost for retrieving past sufficiency