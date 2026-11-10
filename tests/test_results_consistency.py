"""Consistency tests for the committed result tables.

These do not re-run the experiments (that needs the raw data); they check that
the tables the manuscript cites are internally coherent, so a stale or
half-written JSON cannot silently reach the paper.
"""
import json
from pathlib import Path

import numpy as np
import pytest

RES = Path(__file__).resolve().parents[1] / "results"
pytestmark = pytest.mark.skipif(not RES.exists(), reason="results/ not built")

INV16 = ["br_freq_hz", "br_period_s", "br_period_cv", "ie_ratio", "br_skew", "br_kurt",
         "hr_freq_hz", "hr_period_cv", "hrv_rmssd_s", "hr_skew", "hr_kurt",
         "hr_br_ratio", "br_power_frac", "hr_power_frac", "br_hr_power_ratio",
         "cardioresp_coupling"]


def _j(name):
    with open(RES / name) as f:
        return json.load(f)


def test_vitals_summary_numbers():
    S = _j("vitals_summary.json")
    dog = S["dog_recording"]["full"]
    assert dog["hr"]["n"] == 10 and abs(dog["hr"]["mae"] - 7.11) < 0.2
    assert abs(dog["br"]["mae"] - 1.27) < 0.1
    vs = S["vs120_recording"]["hr"]
    assert vs["n"] == 48 and abs(vs["mae"] - 1.54) < 0.1
    # the median absolute error can never exceed the maximum
    for k in ("dog_recording", "vs120_recording"):
        pass
    for block in S["dog_recording"].values():
        for m in ("hr", "br"):
            assert block[m]["median_ae"] <= block[m]["max_ae"] + 1e-9


def test_ablation_ordering_is_plausible():
    S = _j("vitals_summary.json")["dog_recording"]
    assert S["full"]["hr"]["mae"] <= S["plain_median_fusion"]["hr"]["mae"]
    assert S["full"]["hr"]["mae"] <= S["max_snr_channel"]["hr"]["mae"]


def test_identification_protocols_differ():
    S = _j("identification_summary.json")["closed_set"]
    p0 = S["P0"]["resp+cardiac"]["RF"]["mean"]
    p1 = S["P1"]["resp+cardiac"]["RF"]
    p2 = S["P2"]["resp+cardiac"]["RF"]
    assert 0.50 < p0 < 0.60
    assert 0.40 < p1 < 0.52
    assert p2 < p1, "long-horizon accuracy should be lower than short-horizon"
    assert S["P1"]["n_train"] == 1010 and S["P1"]["n_test"] == 510


def test_open_set_repeated_partitions_are_near_chance():
    S = _j("identification_summary.json")["open_set"]
    r = S["random_partitions"]
    assert abs(r["Mahalanobis"]["auroc"]["mean"] - 0.5) < 0.12
    assert abs(r["MaxProb"]["auroc"]["mean"] - 0.5) < 0.15
    for k, v in r.items():
        if isinstance(v, dict) and "accept_known" in v:
            assert 0 < v["accept_known"]["mean"] < 1
            assert 0 < v["reject_unknown"]["mean"] < 1


def test_species_separation_does_not_exceed_the_hardware_control():
    """The paper's central negative result: dog-vs-human separability must not
    be larger than human-vs-human separability across two different radars.
    If this ever fails, the claim must be rewritten rather than the test."""
    S = _j("cross_species_summary.json")
    species = S["B_all110_grouped"]["RF"]["balanced_acc"]
    control = S["NC1_human60_vs_human120"]["inv16"]["balanced_acc"]
    assert species <= control, "species separation exceeds the same-species control"
    assert control - species < 0.15, "the two must stay in the same range"
    # and a second same-species control (same radar, different session state)
    assert S["NC2_dog_scen1_vs_scen2"]["inv16"]["balanced_acc"] > 0.75
    # the permutation test guards against the split being uninformative
    assert S["F_permutation_dog_vs_p60"]["p"] < 0.01


def test_rate_matched_control_collapses_grid_features():
    S = _j("rate_matched_age_summary.json")["rate_matched"]
    gap = S["dog_native_50Hz"]["grid5_only"]["balanced_acc"] - S["dog_decimated_10Hz"]["grid5_only"]["balanced_acc"]
    assert gap > 0.05, "grid-dependent descriptors must degrade once grids are matched"
    inv_gap = abs(S["dog_native_50Hz"]["inv16"]["balanced_acc"]
                  - S["dog_decimated_10Hz"]["inv16"]["balanced_acc"])
    assert inv_gap < 0.05, "invariant descriptors must be unaffected"


def test_age_is_not_better_than_the_mean_predictor():
    S = _j("rate_matched_age_summary.json")["age"]
    assert S["RF"]["r2"] < 0.05
    assert S["Ridge"]["mae"] >= S["Mean"]["mae"] - 0.5
    assert S["ridge_perm_p"] > 0.05


def test_feature_tables_have_the_documented_columns():
    import pandas as pd
    inv = pd.read_csv(RES / "features" / "inv16_dogs.csv")
    assert set(INV16) <= set(inv.columns)
    assert len(inv) == 560
    p60 = pd.read_csv(RES / "features" / "inv16_p60.csv")
    assert len(p60) == 1408
    vs = pd.read_csv(RES / "features" / "inv16_vs120.csv")
    assert len(vs) == 480
