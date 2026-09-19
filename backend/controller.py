from usefulness import UsefulnessModel, build_features


class EcoBudgetController:
    """
    Repeatedly selects the passage with the highest predicted-usefulness-per-byte
    and stops when requirements are covered, no useful candidate remains, or a
    hard cap is hit. Ground truth never enters selection or stopping.
    """

    def __init__(self, tracker, usefulness_model=None,
                 max_loads=15, min_useful=0.05, byte_budget=None):
        self.tracker = tracker
        self.model = usefulness_model or UsefulnessModel()
        self.max_loads = max_loads
        self.min_useful = min_useful
        self.byte_budget = byte_budget
        self.num_loads = 0
        self.cumulative_bytes = 0

    def score(self, candidates):
        scored = []
        for c in candidates:
            sim = c.get('embed_utility', c.get('utility', 0.0))
            feats = build_features(sim, c['bytes'], self.tracker,
                                   self.num_loads, self.cumulative_bytes)
            usefulness = float(self.model.predict(feats)[0])
            scored.append({**c, "usefulness": usefulness,
                           "sel_score": usefulness / max(c['bytes'], 1)})
        scored.sort(key=lambda x: x["sel_score"], reverse=True)
        return scored

    def select_next(self, candidates):
        if not candidates:
            return None
        return self.score(candidates)[0]

    def should_stop(self, best):
        if self.tracker.is_sufficient(threshold=1.0):
            return True, "all_requirements_covered"
        if self.num_loads >= self.max_loads:
            return True, "max_loads_reached"
        if self.byte_budget and self.cumulative_bytes >= self.byte_budget:
            return True, "byte_budget_exhausted"
        if best is None:
            return True, "no_candidates"
        if best["usefulness"] < self.min_useful:
            return True, "no_useful_candidate"
        return False, None

    def register_load(self, passage):
        self.num_loads += 1
        self.cumulative_bytes += passage['bytes']