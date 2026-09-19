"""Generate the 4 headline paper figures from the measured results.

Reads data/phase7_results_test.json (frozen test split, measured payloads) if
present, else data/phase7_results.json, plus data/measured_payloads.json and
data/e2e_results.json. Writes 300-dpi PNGs to figures/.

  fig1_pareto.png            success vs energy -- the money plot
  fig2_crossover.png         compute vs 5G transfer across payload realism
  fig3_radio_tail.png        5G RRC radio/tail energy by condition (the novelty)
  fig4_realpage_savings.png  measured byte savings on real live web pages

Usage: python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# palette
GREEN = "#2a9d8f"    # our adaptive methods
BLUE = "#264653"     # learned externals
GREY = "#9aa0a6"     # fixed budgets
RED = "#e76f51"      # full load / worst
INK = "#1d1d1f"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14,
    "axes.titleweight": "bold", "axes.edgecolor": "#cccccc", "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.8,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ADAPTIVE = {"bandit", "heuristic", "adaptive_rag", "linucb", "lints", "one_per_req"}


# The reported headline set (matches phase7_experiment.py HEADLINE_TYPES):
# comparison, single_fact, multi_part, yes_no; procedure excluded (n<3, always fails).
HEADLINE_TYPES = {"comparison", "single_fact", "multi_part", "yes_no"}


def load():
    f = DATA / "phase7_results_test.json"
    split = "test"
    if not f.exists():
        f = DATA / "phase7_results.json"; split = "val"
    res = json.loads(f.read_text())
    # Replace each condition's top-level (all-types) aggregate with the HEADLINE
    # aggregate, so the figures show the SAME numbers as the reported tables
    # (the top-level includes the failing procedure tasks and would understate
    # success). Weighted by per-type n.
    pt = res.get("per_type", {})
    for cond in list(res.keys()):
        if cond == "per_type" or cond not in pt:
            continue
        types = pt[cond]
        keys = set()
        for ty in HEADLINE_TYPES:
            if ty in types:
                keys |= {k for k, v in types[ty].items() if isinstance(v, (int, float))}
        agg = {}
        den = sum(types[ty]["n"] for ty in HEADLINE_TYPES if ty in types)
        for k in keys:
            if k == "n":
                continue
            num = sum(types[ty][k] * types[ty]["n"] for ty in HEADLINE_TYPES
                      if ty in types and k in types[ty])
            agg[k] = num / den if den else res[cond].get(k, 0.0)
        agg["n"] = den
        res[cond] = agg
    return res, split


def kfmt(x, _=None):
    return f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"


def fig1_pareto(res, split):
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    # baselines get individual labels with hand-tuned offsets to avoid collisions
    baseline_labels = {
        "full": (8, -4), "fixed-1500B": (8, 12), "fixed-1000B": (-8, -16),
        "fixed-500B": (8, 8), "fixed-250B": (8, 8),
    }
    adaptive_pts = []
    for c in res:
        if c == "per_type" or "total_j_html_page" not in res[c]:
            continue
        x = res[c]["total_j_html_page"]; y = res[c]["success"]
        if c in ADAPTIVE:
            adaptive_pts.append((x, y))
            ax.scatter(x, y, s=170, color=GREEN, edgecolor="white", linewidth=1.5, zorder=5)
        elif c == "full":
            ax.scatter(x, y, s=150, color=RED, edgecolor="white", linewidth=1.5, zorder=4)
        else:
            ax.scatter(x, y, s=120, color=GREY, edgecolor="white", linewidth=1.5, zorder=4)
        if c in baseline_labels:
            dx, dy = baseline_labels[c]
            ha = "left" if dx >= 0 else "right"
            ax.annotate(c, (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=9, color=INK, ha=ha)
    # one callout for the whole converged adaptive cluster
    if adaptive_pts:
        cx = sum(p[0] for p in adaptive_pts) / len(adaptive_pts)
        cy = sum(p[1] for p in adaptive_pts) / len(adaptive_pts)
        ax.annotate("adaptive methods converge here\n(bandit, heuristic, LinUCB, LinTS,\n"
                    "Adaptive-RAG, one-per-req)",
                    (cx, cy), xytext=(48, -70), textcoords="offset points", fontsize=9.5,
                    color=GREEN, fontweight="bold", ha="left",
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    ax.set_xlabel("Energy per query at real page scale  (J, html_page)")
    ax.set_ylabel("Answer success")
    ax.set_title("Task-sufficient loading is Pareto-optimal")
    ax.scatter([], [], s=150, color=GREEN, label="adaptive (ours + heuristic + externals)")
    ax.scatter([], [], s=120, color=GREY, label="fixed budgets")
    ax.scatter([], [], s=150, color=RED, label="full-page load")
    ax.legend(loc="center right", frameon=True, framealpha=0.95)
    ax.set_title("Task-sufficient loading is Pareto-optimal")
    ax.text(0.02, 0.02, f"{split} set, measured payloads (median HTML 506 KB)",
            transform=ax.transAxes, fontsize=8, color="#888")
    fig.savefig(FIGS / "fig1_pareto.png"); plt.close(fig)


def fig2_crossover(res, split):
    scenarios = ["text", "resource", "html_page", "full_page"]
    labels = ["text\n(~130 B)", "resource\n(~KB)", "html_page\n(506 KB)", "full_page\n(2 MB)"]
    b = res["bandit"]
    compute = [b["compute_j"]] * 4
    transfer = [b[f"transfer_j_{s}"] for s in scenarios]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = range(4)
    ax.plot(x, compute, "-o", color=BLUE, lw=2.5, label="compute (running the models)")
    ax.plot(x, transfer, "-o", color=GREEN, lw=2.5, label="5G transfer (moving bytes)")
    ax.set_yscale("log")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("Energy per query  (J, log scale)")
    ax.set_title("Compute vs 5G transfer: the crossover at real page scale")
    # shade where transfer dominates
    cross = next((i for i in range(4) if transfer[i] > compute[i]), None)
    if cross is not None:
        ax.axvspan(cross - 0.5, 3.5, color=GREEN, alpha=0.07)
        ax.text(3.35, max(transfer) * 0.6, "transfer\ndominates", ha="right",
                color=GREEN, fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.text(0.02, 0.02, f"{split} set, bandit condition", transform=ax.transAxes,
            fontsize=8, color="#888")
    fig.savefig(FIGS / "fig2_crossover.png"); plt.close(fig)


def fig3_radio(res, split):
    conds = ["bandit", "heuristic", "fixed-1000B", "full"]
    present = [c for c in conds if c in res]
    tight = [res[c]["radio_j"] for c in present]
    fastd = [res[c]["radio_j_fastdormancy"] for c in present]
    import numpy as np
    x = np.arange(len(present)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = [GREEN if c in ADAPTIVE else (RED if c == "full" else GREY) for c in present]
    b1 = ax.bar(x - w/2, tight, w, color=colors, label="tight loop (single tail)")
    b2 = ax.bar(x + w/2, fastd, w, color=colors, alpha=0.55, hatch="//",
                label="fast dormancy (per-fetch tail)")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.0f}", (r.get_x() + r.get_width()/2, r.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(present)
    ax.set_ylabel("Radio energy per query  (J)")
    ax.set_title("5G RRC radio / tail energy: fewer fetches, less radio")
    ax.legend(loc="upper left", frameon=True)
    ax.text(0.98, 0.02, f"{split} set", transform=ax.transAxes, ha="right",
            fontsize=8, color="#888")
    fig.savefig(FIGS / "fig3_radio_tail.png"); plt.close(fig)


def fig4_realpage():
    e2e = json.loads((DATA / "e2e_results.json").read_text())
    import numpy as np
    labels = [r["question"].split("?")[0][:26] + "..." for r in e2e]
    full = [r["full_bytes"] / 1000 for r in e2e]
    task = [r["task_bytes"] / 1000 for r in e2e]
    red = [r["byte_reduction_pct"] for r in e2e]
    x = np.arange(len(e2e)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.bar(x - w/2, full, w, color=RED, label="full page load")
    ax.bar(x + w/2, task, w, color=GREEN, label="task-sufficient load")
    for i in range(len(e2e)):
        ax.annotate(f"{full[i]:.0f} KB", (x[i]-w/2, full[i]), xytext=(0,3),
                    textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"{task[i]:.1f} KB\n(-{red[i]:.0f}%)", (x[i]+w/2, task[i]), xytext=(0,3),
                    textcoords="offset points", ha="center", fontsize=8, color=GREEN, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Bytes transferred  (KB)")
    ax.set_title("Measured on real live web pages: >90% fewer bytes")
    ax.legend(loc="upper right", frameon=True)
    ax.text(0.02, 0.92, "live gsmarena.com pages", transform=ax.transAxes,
            fontsize=8, color="#888")
    fig.savefig(FIGS / "fig4_realpage_savings.png"); plt.close(fig)


def main():
    res, split = load()
    fig1_pareto(res, split)
    fig2_crossover(res, split)
    fig3_radio(res, split)
    fig4_realpage()
    print(f"Wrote 4 figures to {FIGS}/ (phase7 split: {split})")
    for p in sorted(FIGS.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
