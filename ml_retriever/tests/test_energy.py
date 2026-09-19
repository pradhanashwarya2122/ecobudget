"""Phase A energy-accounting tests (pure math, deterministic)."""

import math

from ml_retriever.energy import (
    ComputeModel, EnergyAccountant, FiveGTransferModel, OpCounts, query_op_counts,
)


class TestTransfer:
    def test_scales_with_bytes(self):
        m = FiveGTransferModel()
        assert m.energy_joules(2000) == 2 * m.energy_joules(1000)

    def test_ran_dominates(self):
        m = FiveGTransferModel()
        b = m.breakdown_joules(10000)
        assert b["ran"] > b["transport"]  # RAN is the majority share

    def test_zero_bytes_zero_energy(self):
        assert FiveGTransferModel().energy_joules(0) == 0.0


class TestCompute:
    def test_scales_with_op_count(self):
        c = ComputeModel()
        one = c.energy_joules(OpCounts(qa_score=1))
        ten = c.energy_joules(OpCounts(qa_score=10))
        assert math.isclose(ten, 10 * one)

    def test_bigger_model_costs_more(self):
        c = ComputeModel()
        # answer_gen (flan-t5-base) vs query_embed (minilm), one call each
        assert c.energy_joules(OpCounts(answer_gen=1)) > c.energy_joules(OpCounts(query_embed=1))

    def test_more_efficient_device_uses_less(self):
        ops = OpCounts(qa_score=5, answer_gen=2)
        edge = ComputeModel(flops_per_joule=5e10).energy_joules(ops)
        npu = ComputeModel(flops_per_joule=5e12).energy_joules(ops)
        assert npu < edge


class TestAccountant:
    def test_total_is_compute_plus_transfer(self):
        a = EnergyAccountant()
        r = a.account(OpCounts(qa_score=3, answer_gen=2), num_bytes=1000)
        assert math.isclose(r["total_j"], r["compute_j"] + r["transfer_j"])
        assert 0.0 <= r["compute_fraction"] <= 1.0

    def test_gco2e_conversion(self):
        a = EnergyAccountant(grid_g_per_kwh=475.0)
        r = a.account(OpCounts(), num_bytes=0)  # all zero -> zero everything
        assert r["gco2e"] == 0.0


class TestOpCounts:
    def test_per_requirement_counts(self):
        # 2 requirements, 3 passages added, per-requirement answering
        ops = query_op_counts(n_requirements=2, n_passages_added=3, answer_mode="per_requirement")
        assert ops.decompose == 1
        assert ops.query_embed == 2
        assert ops.qa_score == 2 * 3
        assert ops.answer_gen == 2

    def test_joint_mode_one_generation(self):
        ops = query_op_counts(n_requirements=2, n_passages_added=3, answer_mode="joint")
        assert ops.answer_gen == 1


from ml_retriever.energy import PayloadModel
from ml_retriever.types import Passage


def _pg(pid, text_bytes, src):
    return Passage(passage_id=pid, text="x", byte_size=text_bytes, source_url=src)


class TestPayloadModel:
    P = [_pg("a", 200, "http://p1"), _pg("b", 100, "http://p1"), _pg("c", 150, "http://p2")]

    def test_text_is_sum_of_text_bytes(self):
        assert PayloadModel().transfer_bytes(self.P, "text") == 450

    def test_resource_inflates_with_floor(self):
        m = PayloadModel(resource_inflation=4.0, resource_floor_bytes=800)
        # 200*4=800, 100*4=400->floor 800, 150*4=600->floor 800
        assert m.transfer_bytes(self.P, "resource") == 800 + 800 + 800

    def test_html_page_dedups_by_source(self):
        m = PayloadModel(html_page_bytes=60_000)
        # two unique sources (p1, p2) -> 2 pages
        assert m.transfer_bytes(self.P, "html_page") == 2 * 60_000

    def test_full_page_larger_than_html(self):
        # With the MEASURED html_page median (~506 KB) the realistic full-page
        # (with assets, ~2 MB) is ~4x the HTML document, not 10x+.
        m = PayloadModel()
        assert m.transfer_bytes(self.P, "full_page") > 2 * m.transfer_bytes(self.P, "html_page")

    def test_html_page_default_is_measured_median(self):
        # Tier A: the default is grounded in real fetched pages, not the old
        # assumed 60 KB. Kept in sync with data/measured_payloads.json.
        import json
        from pathlib import Path
        assert PayloadModel().html_page_bytes == 506_179
        mp = Path(__file__).resolve().parent.parent / "data" / "measured_payloads.json"
        if mp.exists():
            assert json.loads(mp.read_text())["html_page_bytes_measured"] == 506_179

    def test_crossover_transfer_can_dominate_at_page_scale(self):
        from ml_retriever.energy import EnergyAccountant
        a = EnergyAccountant()
        ops = OpCounts(qa_score=4, answer_gen=2)  # a few inferences
        text = a.account_passages(ops, self.P, "text")
        page = a.account_passages(ops, self.P, "full_page")
        assert text["compute_fraction"] > 0.9      # compute dominates at snippet scale
        assert page["compute_fraction"] < 0.1       # transfer dominates at full-page scale


from ml_retriever.energy import RadioStateModel, ComputeModel


class TestRadioStateModel:
    def test_no_fetch_no_radio_session(self):
        r = RadioStateModel().account(num_bytes=0, n_fetches=0)
        assert r["radio_j"] == 0.0 and r["radio_active_time_s"] == 0.0
        assert r["n_promotions"] == 0

    def test_more_fetches_cost_more_energy_and_time(self):
        m = RadioStateModel()
        few = m.account(num_bytes=134, n_fetches=2)
        many = m.account(num_bytes=552, n_fetches=9)
        # Phase F point: fewer/earlier retrievals => less radio active energy
        # and shorter active time.
        assert many["radio_j"] > few["radio_j"]
        assert many["radio_active_time_s"] > few["radio_active_time_s"]

    def test_single_promotion_in_tight_loop(self):
        r = RadioStateModel(demote_between_fetches=False).account(500, 5)
        assert r["n_promotions"] == 1

    def test_fast_dormancy_magnifies_fetch_count(self):
        tight = RadioStateModel(demote_between_fetches=False)
        fastd = RadioStateModel(demote_between_fetches=True)
        f_few, f_many = fastd.account(134, 2), fastd.account(552, 9)
        t_few, t_many = tight.account(134, 2), tight.account(552, 9)
        assert fastd.account(552, 9)["n_promotions"] == 9
        assert (f_many["radio_j"] - f_few["radio_j"]) > (t_many["radio_j"] - t_few["radio_j"])

    def test_tail_energy_present_even_for_one_fetch(self):
        r = RadioStateModel(tail_seconds=8.0, tail_power_w=1.0).account(50, 1)
        assert math.isclose(r["tail_j"], 8.0, rel_tol=1e-9)

    def test_defaults_are_literature_grounded_ranges(self):
        # Tier B: defaults sit inside the cited 5G/LTE ranges (not free values).
        m = RadioStateModel()
        assert 1.5 <= m.active_power_w <= 3.5     # 5G sub-6 active RX [N21]
        assert 1.0 <= m.tail_power_w <= 1.5       # RRC tail [H12]
        assert 1.2 <= m.promotion_j <= 2.5        # RRC promotion [H12]
        assert 2.0 <= m.tail_seconds <= 11.6      # LTE Ttail / 5G RRC_INACTIVE
        assert 100e6 <= m.throughput_bps <= 250e6 # 5G sub-6 downlink [N21]

    def test_adaptive_beats_full_load_radio_across_coefficients(self):
        # The robustness claim: fewer fetches => less radio energy, for any
        # coefficient choice in the cited ranges.
        for tail in (2.0, 10.0):
            for act in (1.5, 3.5):
                for demote in (False, True):
                    m = RadioStateModel(tail_seconds=tail, active_power_w=act,
                                        demote_between_fetches=demote)
                    adaptive = m.account(2 * 506_179, 2)["radio_j"]
                    full = m.account(9 * 506_179, 9)["radio_j"]
                    assert adaptive < full


class TestComputeTime:
    def test_time_scales_with_ops(self):
        c = ComputeModel()
        one = c.time_seconds(OpCounts(qa_score=1))
        ten = c.time_seconds(OpCounts(qa_score=10))
        assert ten > one > 0
        assert math.isclose(ten, one * 10, rel_tol=1e-9)
