# Energy model (Phase A) — 5G transfer vs compute

First-order energy accounting for the pipeline. Every coefficient is an
explicit, cited *assumption* and a constructor argument (`ml_retriever/energy.py`)
so it can be swapped. The aim is a defensible order-of-magnitude COMPUTE-vs-
TRANSFER comparison and a 5G-grounded transfer term — not a hardware measurement
(that is later-phase work).

## Compute energy (running the models)
`inference FLOPs ≈ 2 · params · (input_tokens + output_tokens)` per forward pass
(standard 2×MAC approximation), summed over every model call in a query, divided
by a device efficiency `flops_per_joule`.

| model | params | role |
|---|---|---|
| flan-t5-base | 248 M | decomposer, answer generator |
| roberta-base-squad2 | 125 M | evidence QA scorer |
| MiniLM-L6 | 22.7 M | query embedding |

- **Device efficiency (dominant assumption):** default `5e10 FLOPS/J`
  (≈50 GFLOPS/W, edge-CPU class). Mobile-SoC NPUs reach 1–10 TOPS/W; a server
  GPU differs — swap per target device.
- Token counts per op are documented averages in `energy.py::OP_TOKENS`.
- Ops per query (caching-independent): 1 decompose, 1 query-embed per
  requirement, **1 QA score per (requirement × added passage)**, and 1 answer
  generation per requirement (per-requirement mode). QA scoring is the dominant
  compute term because it scales with passages retrieved — which is exactly what
  the stopping policy controls.

## Transfer energy (5G access path)
`energy/bit = network_j_per_bit + device_rx_j_per_bit`, with the network term
split RAN + transport (RAN dominates 5G energy, ~70–80%).

- `network_j_per_bit = 2.25e-5` J/bit — from a mid-range mobile-access figure of
  ~0.05 kWh/GB for the 5G era (literature spans ~0.01–0.1 kWh/GB);
  0.05 kWh/GB = 1.8e5 J / 8e9 bit.
- `ran_fraction = 0.75` (RAN share); `device_rx_j_per_bit = 4.0e-7` (smartphone
  modem receive, assumption).
- **Payload note:** Phase A applies this to the passage *text* bytes
  (~25–259 B). Real transfers are KB–MB pages; **Phase B** adds realistic
  `page_bytes`, which will raise the transfer term and re-test the finding below.

## Carbon
`gCO2e = (total_J / 3.6e6 J/kWh) · grid_g_per_kwh`, default grid intensity
`475 gCO2e/kWh` (IEA global-average electricity intensity; green-host variants
lower). Consistency note: the sibling `backend/carbon.py` uses 494 gCO2e/kWh (the
Sustainable Web Design default). The two are different published conventions of
the same quantity, within ~4% and the same order of magnitude; the paper reports
the IEA value and states the coefficient explicitly so it is swappable. This is
deliberate, not a discrepancy.

## Phase A finding (val, headline tasks, n=28 at the time; current runs use n=92 val / n=124 test)
- **Compute dominates transfer by ~100×.** Per query: compute ≈ 5–11 J,
  transfer ≈ 0.03–0.15 J → compute is **98.6–99.4%** of total energy. At
  short-snippet payloads, *saving bytes alone saves almost nothing.* The naive
  "fewer bytes ⇒ greener" story is false at this scale — quantified here.
- **But the policy still wins on TOTAL energy**, because compute is dominated by
  QA-scoring inferences that scale with passages retrieved. Fewer retrievals ⇒
  fewer QA/answer inferences ⇒ less compute. The bandit's total energy is
  **significantly lower** than every non-degenerate baseline:
  - vs heuristic: **−0.82 J/query [95% CI −1.58, −0.23]** (CI excludes 0)
  - vs fixed-1000B/1500B and full: **−4.9 to −5.1 J/query** (large, significant)
- **Reframed claim (stronger and honest):** the energy benefit of adaptive
  stopping comes from *doing less computation* (fewer evidence-scoring
  inferences), not from moving fewer bytes. Compute-aware adaptive retrieval is
  the lever; the transfer term is negligible until payloads are realistic
  (Phase B).

## Threats to validity
- Compute energy is FLOPs-based with assumed device efficiency and token counts
  (first-order). The compute≫transfer *ratio* (~100×) is robust to reasonable
  coefficient choices; absolute joules are not precise.
- Transfer uses text bytes; the "compute dominates" conclusion is
  payload-scale-dependent and is re-tested at realistic page sizes in Phase B.
- Per-query gCO2e is tiny in isolation; it matters at population scale.

---

# Phase B — realistic payloads & the compute↔transfer crossover

Phase A charged transfer at extracted-text bytes (~25–259 B), an optimistic
lower bound. On a real 5G link you fetch a web resource/page (KB–MB) to obtain a
passage. `PayloadModel` (`energy.py`) charges transfer across four scenarios,
de-duplicating page fetches by `source_url`:

| scenario | per-passage/page bytes | source (assumption) |
|---|---|---|
| text | extracted text (25–259 B) | Phase A lower bound |
| resource | text ×4, floor 800 B | HTML markup + HTTP/TLS overhead |
| html_page | 60 KB / unique page (SUPERSEDED: now measured 506 KB, see Tier A addendum) | assumed, later measured |
| full_page | 2 MB / unique page | HTTP-Archive-class median full page weight |

## The crossover finding (val, headline, n=28 at the time; superseded by the Phase G / Tier A tables below at n=92) — total J/query
| condition | text | resource | html_page | full_page |
|---|---|---|---|---|
| bandit | 6.09 | 6.35 | 25.3 | 647 |
| heuristic | 6.91 | 7.26 | 31.2 | 818 |
| fixed-1500B / full | 11.2 | 12.1 | 69.6 | 1961 |

**Bandit transfer share:** text 0.7% → resource 4.7% → **html_page 76%** →
full_page 99%.

**The compute-vs-transfer balance flips with payload realism.** At snippet scale
compute dominates (Phase A). At realistic web-page scale (**html_page onward,
the real RAG-over-web regime**) **transfer dominates** — so loading fewer
pages/bytes is the primary energy lever after all. The Phase A "compute
dominates" result was an artifact of unrealistically small payloads; it holds
only in the text/resource regime.

## The bandit wins in BOTH regimes (paired, 95% CI)
> **SUPERSEDED for the bandit-vs-heuristic row — see the Phase G note below.**
> These numbers were computed under the OLD per-requirement bi-encoder
> retriever. The vs-full and vs-fixed rows still hold directionally; the
> vs-heuristic delta changed after Phase D. Use the Phase G table for headline
> claims.

| bandit vs | net J (text) | net J (html_page) |
|---|---|---|
| heuristic | −0.82 [−1.58, −0.23] | **−5.9 [−10.5, −2.2]** |
| full | −5.12 [−5.75, −4.49] | **−44.3 [−51.3, −37.7]** |
| fixed-1000B | −4.86 [−5.42, −4.29] | **−42.0 [−47.6, −36.4]** |

All CIs exclude 0 (old retriever). At snippet scale compute dominates; at
html_page scale transfer dominates — the crossover finding is unchanged.

---

# Phase G addendum — energy under the entity-aware retriever (current)

After Phase D, `phase7_experiment.py` uses `EntityAwareRetriever` and the bandit
was recalibrated to **λ=0.5** (the old λ=4.0 collapsed under the new byte scale;
see `eval_notes.md`). The crossover story is intact (bandit transfer share:
text 0.4% → html_page 69% → full_page 99%), but the **bandit now matches the
heuristic** rather than beating it — with a near-perfect retriever, both
converge on the same "one retrieve per requirement, then stop" policy.

**Current net energy, bandit vs baselines (val, n=92, paired 95% CI):**
| bandit vs | net J (text) | net J (html_page) |
|---|---|---|
| heuristic | **0.00 [0.00, 0.00]** (identical policy) | **0.0 [0.0, 0.0]** |
| full | −7.28 [−7.67, −6.83] | **−19.3 [−23.9, −15.3]** |
| fixed-1000B | −7.14 [−7.52, −6.69] | **−17.7 [−21.4, −14.5]** |

Honest claim for the paper: the learned policy is **Pareto-optimal** — it ties
the heuristic (the efficient frontier) and dominates fixed/full budgets by
~18–19 J/query at html_page scale (CIs exclude 0), at statistically equal
success. It does NOT beat the heuristic under a strong retriever; its advantage
over hand-tuned heuristics is a function of retrieval uncertainty.

## Threats to validity (Phase B)
- Page counts assume ~1 unique page per passage (our corpus rarely shares a
  `source_url`); if multiple passages share a page, page-scenario differences
  shrink — a conservative assumption for the adaptive policy.
- `html_page`/`full_page` sizes (60 KB, 2 MB) are HTTP-Archive-class medians,
  documented and swappable; the crossover *location* (between resource and
  html_page) is robust, the exact joules are not.
- No page is actually fetched; a measured deployment is later-phase work.

---

# Tier A addendum — payloads grounded in REAL measured pages

Phases B/F above charged the `html_page` scenario at an ASSUMED 60 KB. That was
wrong. `scripts/measure_real_pages.py` measures the 19 real pages the sibling
`backend/` prototype fetched with Playwright (backend/pages/) and writes
`data/measured_payloads.json`:

| quantity | measured |
|---|---|
| real rendered-HTML page (median) | 506,179 B |
| real rendered-HTML page (mean) | 922,112 B |
| extracted visible text (median) | 91,568 B |
| text-to-HTML ratio (median) | 0.167 |

`PayloadModel.html_page_bytes` now defaults to the measured median (506 KB), not
60 KB. Real pages are ~8x heavier than the old assumption, so the transfer term
and the adaptive policy's advantage are correspondingly larger.

## Corrected Phase 7 energy at measured page scale (val, n=92, total J/query)
| condition | html_page (measured) | full_page |
|---|---|---|
| bandit / heuristic | 118.9 | 452.0 |
| linucb / lints | 113.8 | 432.1 |
| fixed-1000B | 215.7 | 813.6 |
| full / fixed-1500B | 227.9 | 861.5 |

Bandit transfer share at html_page is now **95.0%** (was 76% under the 60 KB
assumption): at real page weights transfer overwhelmingly dominates, so loading
fewer pages is decisively the energy lever.

## Bandit vs baselines at measured html_page (paired 95% CI)
- vs full: net **-109.0 J/query [-148.3, -74.7]** (was -19.3 under the assumption)
- vs fixed-1000B: **-96.8 J [-128.1, -67.9]**
- vs heuristic: 0.0 (identical policy); vs linucb: +5.1 J (linucb a hair leaner)

All large CIs exclude 0. Grounding the payloads in real pages multiplies the
measured energy advantage of task-sufficient loading ~5-6x.

## Real-page byte/CO2 savings (measured, on the 19 real pages)
A conventional load transfers the whole document: measured median 506 KB (mean
922 KB) of HTML, ~2 MB with assets. Task-sufficient extraction transfers only the
passages the task needs (~130 B median in our eval) -- a >99.9% byte reduction on
real pages. Over the 5G per-bit model, fetching one 506 KB page costs ~93 J of
transfer; the whole task-sufficient query costs ~6 J. Threats to validity: the
506 KB median is over 19 pages in our corpus domains (products/travel/specs), and
per-passage live fetching of every corpus source_url is future work -- the
payload SCALE is measured, the per-passage mapping is not.

---

# Tier B addendum — RRC radio coefficients grounded in cited literature

The Phase F RadioStateModel coefficients are no longer free assumptions; each is
grounded in 5G/LTE measurement literature or a 3GPP standard, with a plausible
range. Sources:

| coefficient | value (default) | range | source |
|---|---|---|---|
| throughput_bps | 150 Mbps | 100-250 | Narayanan et al., SIGCOMM 2021 (5G sub-6 median downlink) |
| active_power_w | 2.5 W | 1.5-3.5 | Narayanan et al., SIGCOMM 2021 (5G sub-6 active RX, incremental) |
| tail_power_w | 1.2 W | 1.0-1.5 | Huang et al., MobiSys 2012 (LTE RRC_CONNECTED tail) |
| promotion_j | 2.0 J | 1.2-2.5 | Huang et al., MobiSys 2012 (RRC promotion energy) |
| tail_seconds | 10 s | 2-11.6 | Huang et al. (LTE Ttail ~11.6 s); 3GPP TS 38.331 (5G RRC_INACTIVE, shorter/configurable) |
| idle_power_w | 0.02 W | - | RRC_IDLE baseline |

5G introduces RRC_INACTIVE (3GPP TS 38.331) specifically to cut the LTE tail, so
5G tails are typically shorter than LTE's; we sweep 2-10 s rather than fix one.

## Sensitivity (scripts/radio_sensitivity.py) -- is the finding robust?
Sweeping tail_seconds in {2, 5, 10} s x active_power in {1.5, 2.5, 3.5} W x
{tight-loop, fast-dormancy}, using the measured per-condition fetch counts and
charging each fetch one measured 506 KB page:

- The adaptive policy uses **less radio energy than full-load in EVERY one of the
  18 cells** (CONFIRMED). The qualitative claim does not depend on any single
  coefficient choice.
- Tight loop: bandit 4.8-14.8 J vs full 6.5-18.9 J (the constant 8-10 s tail is
  paid by both; the gap is the shorter active span).
- Fast dormancy (5G RRC_INACTIVE releasing between fetches): the gap explodes --
  bandit 9-28 J vs full 41-128 J -- because full-load's ~9 fetches each pay their
  own promotion + tail. This is where the 5G radio-state contribution is
  strongest: adaptive stopping avoids repeated promotions.

Honest read: at realistic 5G coefficients the per-query radio saving vs full-load
ranges from ~2 J (conservative tight loop) to ~100 J (aggressive fast dormancy).
The direction is robust; the magnitude depends on the RRC release policy, which
we report as a range rather than a point estimate.
