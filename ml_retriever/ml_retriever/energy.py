"""Phase A: end-to-end energy accounting -- COMPUTE (running the models) vs
TRANSFER (moving bytes over a 5G access path) -- so we can report NET energy
per query and answer the existential question: do the bytes saved by stopping
early outweigh the joules spent running flan-t5 + RoBERTa-QA + MiniLM?

Everything here is a first-order engineering estimate with explicitly cited,
configurable coefficients. Nothing is measured on real hardware yet (that is a
later phase); the goal is a defensible order-of-magnitude comparison and a
model that is 5G-grounded rather than a single generic web-CO2 coefficient.

Two independent models:
  * ComputeModel: transformer inference energy = FLOPs / (device FLOPs-per-joule).
    FLOPs ~= 2 * params * (input_tokens + output_tokens) per forward pass (the
    standard 2xMAC approximation), summed over every model call in a query.
  * FiveGTransferModel: energy per transported bit split into RAN (access) +
    transport (core) + device modem receive -- the RAN dominates 5G energy.

Carbon: joules -> kWh -> gCO2e via grid carbon intensity.

Coefficient sources are documented in docs/energy_model.md; the defaults below
are mid-range literature values, labelled as assumptions, and every one is a
constructor argument so a reviewer can swap them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Model sizes (trainable+frozen params, approximate) -------------------
MODEL_PARAMS = {
    "flan-t5-small": 77_000_000,     # decomposer-small / answer-small base
    "flan-t5-base": 248_000_000,     # decomposer-base-lora / answer-base base
    "minilm": 22_700_000,            # all-MiniLM-L6-v2 query embedding
    "roberta-base-squad2": 125_000_000,  # QA evidence scorer
}

# Average token counts per op type (documented approximations; see energy_model.md).
# input+output tokens processed by one call of that op.
OP_TOKENS = {
    "decompose": (24, 24),       # flan-t5-base: question in, entity|attribute out
    "query_embed": (10, 0),      # MiniLM: one requirement query encoded
    "qa_score": (96, 8),         # roberta: question+passage in, short span out
    "answer_gen": (160, 24),     # flan-t5-base: prompt(question+evidence) in, answer out
}

OP_MODEL = {
    "decompose": "flan-t5-base",
    "query_embed": "minilm",
    "qa_score": "roberta-base-squad2",
    "answer_gen": "flan-t5-base",
}


@dataclass
class OpCounts:
    """How many times each model op runs for a single query (logical count,
    independent of any caching used to speed up the experiment)."""
    decompose: int = 0
    query_embed: int = 0
    qa_score: int = 0
    answer_gen: int = 0

    def as_dict(self) -> dict:
        return {"decompose": self.decompose, "query_embed": self.query_embed,
                "qa_score": self.qa_score, "answer_gen": self.answer_gen}


@dataclass
class ComputeModel:
    """Inference energy from FLOPs and a device-class efficiency.

    `flops_per_joule` is the dominant assumption. Defaults to an edge-CPU class
    (~50 GFLOPS/W = 5e10 FLOPS/J). Mobile-SoC NPUs are far more efficient
    (1-10 TOPS/W); a server GPU is different again -- swap per target device."""
    flops_per_joule: float = 5.0e10  # edge-CPU class assumption
    # Compute THROUGHPUT (for latency, Phase F), distinct from efficiency above.
    # ~0.2 TFLOP/s edge-CPU class; mobile NPUs reach 1-10 TFLOP/s -- swap per
    # device. Only used for end-to-end latency, never for energy.
    flops_per_second: float = 2.0e11

    def _flops(self, op: str, count: int) -> float:
        if count == 0:
            return 0.0
        params = MODEL_PARAMS[OP_MODEL[op]]
        tin, tout = OP_TOKENS[op]
        # 2 * params * tokens per forward pass (2xMAC); generation counts
        # output tokens as additional autoregressive steps.
        return count * 2.0 * params * (tin + tout)

    def _total_flops(self, ops: OpCounts) -> float:
        return sum(self._flops(op, getattr(ops, op)) for op in OP_TOKENS)

    def energy_joules(self, ops: OpCounts) -> float:
        return self._total_flops(ops) / self.flops_per_joule

    def time_seconds(self, ops: OpCounts) -> float:
        """Wall-clock compute time (Phase F latency): FLOPs / device throughput."""
        return self._total_flops(ops) / self.flops_per_second

    def breakdown_joules(self, ops: OpCounts) -> dict:
        return {op: self._flops(op, getattr(ops, op)) / self.flops_per_joule
                for op in OP_TOKENS}


@dataclass
class FiveGTransferModel:
    """Energy to move one bit over a 5G access path, split RAN + transport +
    device-modem receive. RAN dominates mobile-network energy (~70-80%).

    Defaults derive from a mid-range mobile-access figure of ~0.05 kWh/GB for
    the 5G era (studies span ~0.01-0.1 kWh/GB); that is 0.05*3.6e6/8e9 =
    2.25e-8... note: 0.05 kWh/GB = 1.8e5 J / 8e9 bit = 2.25e-5 J/bit network,
    split 75% RAN / 25% transport, plus a separate device-modem receive term."""
    network_j_per_bit: float = 2.25e-5   # RAN + transport (mobile access, 5G-era assumption)
    ran_fraction: float = 0.75           # RAN share of network energy
    device_rx_j_per_bit: float = 4.0e-7  # smartphone modem receive energy per bit (assumption)

    @property
    def ran_j_per_bit(self) -> float:
        return self.network_j_per_bit * self.ran_fraction

    @property
    def transport_j_per_bit(self) -> float:
        return self.network_j_per_bit * (1.0 - self.ran_fraction)

    def energy_joules(self, num_bytes: int) -> float:
        bits = num_bytes * 8
        return bits * (self.network_j_per_bit + self.device_rx_j_per_bit)

    def breakdown_joules(self, num_bytes: int) -> dict:
        bits = num_bytes * 8
        return {"ran": bits * self.ran_j_per_bit,
                "transport": bits * self.transport_j_per_bit,
                "device_rx": bits * self.device_rx_j_per_bit}


@dataclass
class PayloadModel:
    """Maps a set of retrieved passages to the bytes actually transported over
    the link, under a chosen realism scenario (Phase B).

    Phase A used raw extracted-text bytes (~25-259 B) -- an under-count, because
    on a real link you fetch a web resource/page, not a clean snippet. Scenarios
    span the realistic range so the compute-vs-transfer finding is reported
    across assumptions rather than pinned to one:
      - "text":      extracted text bytes (Phase A; optimistic lower bound).
      - "resource":  the passage as a real web resource -- text inflated for
                     HTML markup + HTTP/TLS overhead, with a floor.
      - "html_page": fetch the source HTML document once per unique page
                     (HTTP-Archive-class median HTML weight).
      - "full_page": fetch the full page with assets once per unique page
                     (HTTP-Archive-class median total page weight).
    Page scenarios de-duplicate by source_url (one fetch serves all its
    passages), which is what rewards a policy that touches fewer pages.
    """
    resource_inflation: float = 4.0        # HTML markup + overhead vs extracted text
    resource_floor_bytes: int = 800        # min realistic resource-on-wire size
    # MEASURED (Tier A): median rendered-HTML byte size of the 19 real pages the
    # backend fetched with Playwright (backend/pages/), computed by
    # scripts/measure_real_pages.py -> data/measured_payloads.json. Replaces the
    # earlier assumed 60 KB, which underestimated real page weight ~8x.
    html_page_bytes: int = 506_179         # measured median real HTML document
    full_page_bytes: int = 2_000_000       # full page weight with assets (HTTP-Archive-class assumption)

    def transfer_bytes(self, passages, scenario: str = "text") -> int:
        if scenario == "text":
            return sum(p.byte_size for p in passages)
        if scenario == "resource":
            return sum(max(int(p.byte_size * self.resource_inflation), self.resource_floor_bytes)
                       for p in passages)
        # page-level: one fetch per unique source
        sources = {getattr(p, "source_url", "") or p.passage_id for p in passages}
        n_pages = len(sources)
        if scenario == "html_page":
            return n_pages * self.html_page_bytes
        if scenario == "full_page":
            return n_pages * self.full_page_bytes
        raise ValueError(f"unknown scenario: {scenario!r}")


SCENARIOS = ("text", "resource", "html_page", "full_page")


@dataclass
class RadioStateModel:
    """Phase F: 5G RRC radio-state / tail energy -- the networking-specific
    lever that a byte count alone misses.

    5G energy is dominated not just by bytes moved but by how long the modem is
    held in the high-power RRC_CONNECTED state. Fetching keeps the radio awake;
    after the last fetch an inactivity timer (`tail_seconds`) holds it CONNECTED
    before it releases to RRC_IDLE -- the classic *tail energy* waste. A policy
    that issues fewer, earlier retrievals keeps the radio active for a shorter
    span and lets the tail start sooner.

    First-order per-query model (all coefficients cited/assumed, swappable):
      - One IDLE->CONNECTED promotion per query (radio wakes once).
      - The retrieval loop serializes n_fetches: each costs a scheduling/RTT
        setup + its transfer time, with per-step compute (QA-score + decide)
        between fetches. Across a tight loop the inactivity timer keeps resetting,
        so the radio stays CONNECTED for the whole span (active power).
      - After the last fetch: `tail_seconds` at `tail_power_w`, then release.

    Fast-dormancy sensitivity (`demote_between_fetches=True`): if the network
    releases aggressively, each fetch pays its own promotion + tail -- which
    magnifies the advantage of issuing fewer fetches. Reported as a sensitivity;
    the default (single tail) is the CONSERVATIVE choice for our claim.
    """
    # Tier B: coefficients grounded in 5G/LTE measurement literature + 3GPP, not
    # free assumptions. Sources (see docs/energy_model.md Tier B table):
    #  [N21] Narayanan et al., "A Variegated Look at 5G in the Wild", SIGCOMM 2021
    #  [H12] Huang et al., "A Close Examination of ... 4G LTE Networks", MobiSys 2012
    #  [38.331] 3GPP TS 38.331 (NR RRC: IDLE / INACTIVE / CONNECTED, timers)
    throughput_bps: float = 150e6      # 5G sub-6 median downlink ~100-250 Mbps [N21]
    active_power_w: float = 2.5        # 5G sub-6 RRC_CONNECTED active RX, incremental [N21] (range 1.5-3.5)
    tail_power_w: float = 1.2          # RRC_CONNECTED inactivity tail / DRX [H12] (LTE tail ~1.06-1.5 W)
    idle_power_w: float = 0.02         # RRC_IDLE baseline (near-negligible)
    promotion_j: float = 2.0           # IDLE->CONNECTED ramp energy [H12] (LTE ~1.2-2.5 J)
    tail_seconds: float = 10.0         # RRC inactivity timer; LTE Ttail ~11.6 s [H12], 5G configurable/shorter via RRC_INACTIVE [38.331]
    per_step_compute_s: float = 0.10   # QA-score + decide between fetches (measured-order)
    setup_s_per_fetch: float = 0.04    # scheduling + RTT per fetch [N21]-class RTT
    demote_between_fetches: bool = False  # fast-dormancy sensitivity toggle [38.331] RRC_INACTIVE

    def transfer_time_s(self, num_bytes: int) -> float:
        return (num_bytes * 8) / self.throughput_bps

    def account(self, num_bytes: int, n_fetches: int) -> dict:
        """Radio energy + active-time for a query that issued `n_fetches`
        retrievals moving `num_bytes` total. n_fetches=0 => no radio session."""
        if n_fetches <= 0:
            return {"radio_active_time_s": 0.0, "promotion_j": 0.0,
                    "active_j": 0.0, "tail_j": 0.0, "radio_j": 0.0, "n_promotions": 0}
        tx = self.transfer_time_s(num_bytes)
        if self.demote_between_fetches:
            # Each fetch is its own session: promotion + its active slice + tail.
            per_fetch_active = tx / n_fetches + self.setup_s_per_fetch
            active_span = per_fetch_active * n_fetches
            n_promotions = n_fetches
            tail_total = self.tail_seconds * n_fetches
        else:
            # Tight loop: one promotion, radio held across the whole span, one tail.
            active_span = tx + n_fetches * self.setup_s_per_fetch \
                + max(n_fetches - 1, 0) * self.per_step_compute_s
            n_promotions = 1
            tail_total = self.tail_seconds
        promotion_j = self.promotion_j * n_promotions
        active_j = active_span * self.active_power_w
        tail_j = tail_total * self.tail_power_w
        radio_j = promotion_j + active_j + tail_j
        return {
            "radio_active_time_s": active_span + tail_total,
            "promotion_j": promotion_j,
            "active_j": active_j,
            "tail_j": tail_j,
            "radio_j": radio_j,
            "n_promotions": n_promotions,
        }


@dataclass
class EnergyAccountant:
    """Combines compute + 5G transfer into total joules and gCO2e."""
    compute: ComputeModel = field(default_factory=ComputeModel)
    transfer: FiveGTransferModel = field(default_factory=FiveGTransferModel)
    payload: PayloadModel = field(default_factory=PayloadModel)
    # Grid carbon intensity. 475 gCO2e/kWh = IEA global-average electricity
    # intensity (well cited). The sibling backend/carbon.py uses 494 (the
    # Sustainable Web Design / SWD default); both are documented, same order of
    # magnitude, and swappable -- see docs/energy_model.md Carbon note.
    grid_g_per_kwh: float = 475.0

    def account_passages(self, ops: OpCounts, passages, scenario: str = "text") -> dict:
        """Account energy with transfer bytes derived from the retrieved
        passages under a payload realism scenario (Phase B)."""
        return self.account(ops, self.payload.transfer_bytes(passages, scenario))

    def account(self, ops: OpCounts, num_bytes: int) -> dict:
        compute_j = self.compute.energy_joules(ops)
        transfer_j = self.transfer.energy_joules(num_bytes)
        total_j = compute_j + transfer_j
        gco2e = (total_j / 3.6e6) * self.grid_g_per_kwh  # J -> kWh -> gCO2e
        return {
            "compute_j": compute_j,
            "transfer_j": transfer_j,
            "total_j": total_j,
            "compute_fraction": compute_j / total_j if total_j else 0.0,
            "gco2e": gco2e,
        }


# --- Logical op-counting for the pipeline (caching-independent) ------------

def query_op_counts(n_requirements: int, n_passages_added: int,
                    answer_mode: str = "per_requirement") -> OpCounts:
    """The model ops one query performs, independent of any experiment cache:
      - 1 decompose (question -> requirements)
      - 1 query embedding per requirement (retrieval)
      - 1 QA score per (requirement, added passage): the tracker scores every
        added passage against every requirement
      - answer generations: 1 per requirement (per_requirement) or 1 (joint)
    """
    n_gen = n_requirements if answer_mode == "per_requirement" else 1
    return OpCounts(
        decompose=1,
        query_embed=n_requirements,
        qa_score=n_requirements * n_passages_added,
        answer_gen=n_gen,
    )
