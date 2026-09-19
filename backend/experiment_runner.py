import json
import csv
import time
import uuid
import os
from scorer import parse_resources, check_task_success
from conditions import run_normal, run_fixed_eco, run_ecobudget

BENCHMARK = "benchmark_tasks_v1_FROZEN.json"
FIXED_BUDGETS = [10_000, 25_000, 50_000, 100_000]


def load_benchmark():
    with open(BENCHMARK) as f:
        return json.load(f)


def all_conditions(task_text, resources):
    yield run_normal(task_text, resources)
    for b in FIXED_BUDGETS:
        yield run_fixed_eco(task_text, resources, fixed_budget_bytes=b)
    yield run_ecobudget(task_text, resources)


def main():
    tasks = load_benchmark()
    run_id = time.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    os.makedirs("results", exist_ok=True)
    out_path = f"results/run_{run_id}.csv"

    fields = ["task_id", "category", "difficulty", "condition", "success",
              "bytes_used", "co2e_grams", "resources_loaded", "coverage",
              "stop_reason", "answer", "ground_truth"]
    rows = []

    for t in tasks:
        html = open(t["page_file"]).read()
        resources = parse_resources(html, base_url=t.get("url"))
        for out in all_conditions(t["question"], resources):
            success = check_task_success(out.get("answer"), t["ground_truth"])
            rows.append({
                "task_id": t["id"],
                "category": t.get("category", ""),
                "difficulty": t.get("difficulty", ""),
                "condition": out["condition"],
                "success": int(success),
                "bytes_used": out["bytes_used"],
                "co2e_grams": out["carbon"]["estimated_co2e_grams"],
                "resources_loaded": out["resources_loaded"],
                "coverage": out.get("coverage", ""),
                "stop_reason": out.get("gate_reason", ""),
                "answer": (out.get("answer") or "")[:80],
                "ground_truth": t["ground_truth"],
            })
        print(f"  done {t['id']}: {t['question'][:50]}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {out_path}\n")

    from collections import defaultdict
    agg = defaultdict(lambda: {"success": 0, "bytes": 0, "co2e": 0.0, "n": 0})
    for r in rows:
        a = agg[r["condition"]]
        a["success"] += r["success"]
        a["bytes"] += r["bytes_used"]
        a["co2e"] += r["co2e_grams"]
        a["n"] += 1

    print(f"{'Condition':<14}{'Success':>9}{'AvgBytes':>12}{'AvgCO2e_g':>12}")
    print("-" * 47)
    for cond, a in agg.items():
        print(f"{cond:<14}{a['success']}/{a['n']:>7}"
              f"{a['bytes'] // a['n']:>12}{a['co2e'] / a['n']:>12.5f}")


if __name__ == "__main__":
    main()
