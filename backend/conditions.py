"""
EcoBudget v2 conditions.
Adaptive EcoBudget uses the ML controller (usefulness-per-byte selection +
learned stop) over the Flan-T5 EvidenceCoverageTracker.
Ground truth is never used inside any condition.
"""

from scorer import rank_by_vpb, find_best_answer_full_page, check_answerability
from evidence import EvidenceCoverageTracker
from controller import EcoBudgetController
from carbon import estimate_co2e


def select_within_budget(ranked_resources, budget_bytes):
    selected, used = [], 0
    for r in ranked_resources:
        if used + r['bytes'] <= budget_bytes:
            selected.append(r)
            used += r['bytes']
    return selected, used


def best_from_selected(selected):
    if not selected:
        return 0.0, None
    best = max(selected, key=lambda r: r.get('qa_score', r.get('utility', 0)))
    return best.get('qa_score', best.get('utility', 0)), best.get('extracted_answer')


def run_normal(task_text, resources):
    total_bytes = sum(r['bytes'] for r in resources)
    text_blob = " ".join(r['content'] for r in resources if r['type'] == 'text')
    score, answer = find_best_answer_full_page(task_text, resources, use_entity_filter=False)
    return {
        "condition": "Normal", "bytes_used": total_bytes,
        "resources_loaded": len(resources), "resources_total": len(resources),
        "answer": answer, "answer_score": score, "selected_text": text_blob,
        "budget_final": None, "iterations": 1, "gate_reason": "full page",
        "carbon": estimate_co2e(total_bytes),
    }


def run_fixed_eco(task_text, resources, fixed_budget_bytes=20_000):
    text_resources = [r for r in resources if r['type'] == 'text']
    ranked = rank_by_vpb(task_text, resources, use_entity_filter=False)
    selected, used_bytes = select_within_budget(ranked, fixed_budget_bytes)
    score, answer = best_from_selected(selected)
    return {
        "condition": f"Fixed-{fixed_budget_bytes // 1000}KB", "bytes_used": used_bytes,
        "resources_loaded": len(selected), "resources_total": len(text_resources),
        "answer": answer, "answer_score": score,
        "selected_text": " ".join(r['content'] for r in selected),
        "budget_final": fixed_budget_bytes, "iterations": 1,
        "gate_reason": "fixed budget",
        "carbon": estimate_co2e(used_bytes),
    }


def run_ecobudget(task_text, resources, max_passages=15):
    """
    Adaptive controller: score all not-yet-loaded candidates by predicted
    usefulness / byte, load the best, update evidence, stop via the
    learned/heuristic rule. Candidates tracked by identity, not content.
    """
    ranked = rank_by_vpb(task_text, resources, use_entity_filter=True)
    text_ranked = [r for r in ranked if r['type'] == 'text']
    for i, r in enumerate(text_ranked):
        r['_cand_id'] = i

    tracker = EvidenceCoverageTracker(task_text)
    controller = EcoBudgetController(tracker, max_loads=max_passages)

    def qa_fn(question, passage):
        return check_answerability(question, passage)

    loaded_ids = set()
    passages_loaded = 0
    stop_reason = "exhausted_candidates"

    while True:
        candidates = [r for r in text_ranked if r['_cand_id'] not in loaded_ids]
        best = controller.select_next(candidates)
        stop, reason = controller.should_stop(best)
        if stop:
            stop_reason = reason
            break

        loaded_ids.add(best['_cand_id'])
        controller.register_load(best)
        tracker.check_passage(best['content'], best['bytes'], qa_fn)
        passages_loaded += 1

    best_answer = tracker.best_answer()
    if best_answer is None:
        loaded = [r for r in text_ranked if r['_cand_id'] in loaded_ids]
        best_score, best_answer = best_from_selected(loaded)
    else:
        satisfied = [r for r in tracker.requirements if r["satisfied"]]
        best_score = max((r.get("confidence", 0) for r in satisfied), default=0.0)

    return {
        "condition": "EcoBudget",
        "bytes_used": controller.cumulative_bytes,
        "resources_loaded": passages_loaded,
        "resources_total": len(text_ranked),
        "answer": best_answer,
        "answer_score": best_score,
        "selected_text": " ".join(r['content'] for r in text_ranked
                                  if r['_cand_id'] in loaded_ids),
        "budget_final": controller.cumulative_bytes,
        "iterations": passages_loaded,
        "gate_reason": stop_reason,
        "coverage": tracker.coverage(),
        "requirements_met": tracker.n_satisfied(),
        "requirements_total": tracker.n_total(),
        "coverage_detail": tracker.status_report(),
        "carbon": estimate_co2e(controller.cumulative_bytes),
    }