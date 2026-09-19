"""Tier B: is the radio-energy finding robust to the (cited) coefficient ranges?

The RadioStateModel coefficients are grounded in 5G/LTE measurement literature
(Narayanan SIGCOMM 2021; Huang MobiSys 2012; 3GPP TS 38.331), each with a
plausible RANGE, not a single true value. This sweeps the two coefficients the
result is most sensitive to (tail_seconds and active_power_w) plus the
fast-dormancy toggle, using the per-condition fetch counts from the last Phase 7
run (data/phase7_results.json), and checks that the qualitative claim -- the
adaptive policy uses less radio energy than full-load in EVERY cell -- holds
across the whole grid.

Each fetch is charged one html_page (measured 506 KB) of transported bytes.

Usage: python scripts/radio_sensitivity.py
"""
import json
from pathlib import Path

from ml_retriever.energy import PayloadModel, RadioStateModel

DATA = Path(__file__).resolve().parent.parent / "data"


def main():
    res = json.loads((DATA / "phase7_results.json").read_text())
    html_bytes = PayloadModel().html_page_bytes  # measured 506 KB
    fetches = {c: res[c]["actions"] for c in ("bandit", "heuristic", "full", "fixed-1000B")}

    tail_grid = [2.0, 5.0, 10.0]        # 5G RRC_INACTIVE short ... LTE Ttail [H12/38.331]
    active_grid = [1.5, 2.5, 3.5]       # 5G sub-6 active RX range [N21]
    print("Radio-energy sensitivity (per query). Each fetch = one 506 KB page.")
    print("Checking: bandit radio_J < full radio_J in every cell.\n")
    print(f"{'tail_s':>6}{'act_W':>7}{'dorm':>6}{'bandit_J':>10}{'full_J':>9}"
          f"{'fixed1k_J':>11}{'bandit<full?':>13}")
    all_hold = True
    for demote in (False, True):
        for tail in tail_grid:
            for act in active_grid:
                m = RadioStateModel(tail_seconds=tail, active_power_w=act,
                                    demote_between_fetches=demote)

                def rj(cond):
                    n = fetches[cond]
                    return m.account(int(round(n)) * html_bytes, int(round(n)))["radio_j"]

                b, f, fx = rj("bandit"), rj("full"), rj("fixed-1000B")
                hold = b < f
                all_hold &= hold
                print(f"{tail:>6.0f}{act:>7.1f}{('fastD' if demote else 'tight'):>6}"
                      f"{b:>10.2f}{f:>9.2f}{fx:>11.2f}{('yes' if hold else 'NO'):>13}")
    print(f"\nRobustness: bandit uses less radio energy than full-load in EVERY cell: "
          f"{'CONFIRMED' if all_hold else 'FAILS'}")


if __name__ == "__main__":
    main()
