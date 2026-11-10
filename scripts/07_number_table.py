"""Emit ``docs/NUMBERS.md``: every headline number in the manuscript mapped to
the committed result file and field it comes from.

Run after the experiment scripts.  The table is generated rather than typed so
that a re-run cannot leave the documentation behind the results; the test
``tests/test_results_consistency.py`` guards the result files themselves.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "NUMBERS.md"

# (section, claim, file, dotted path, formatter)
ROWS = [
    # --- vital signs -----------------------------------------------------
    ("Vital signs", "Dog HR MAE (recording level, n = 10)", "vitals_summary.json",
     "dog_recording.full.hr.mae", "bpm2"),
    ("Vital signs", "Dog HR bias", "vitals_summary.json",
     "dog_recording.full.hr.bias", "signed2"),
    ("Vital signs", "Dog HR 95 % limits of agreement", "vitals_summary.json",
     "dog_recording.full.hr", "loa"),
    ("Vital signs", "Dog HR Pearson r", "vitals_summary.json",
     "dog_recording.full.hr.pearson_r", "r2"),
    ("Vital signs", "Dog BR MAE (recording level)", "vitals_summary.json",
     "dog_recording.full.br.mae", "bpm2"),
    ("Vital signs", "Dog HR MAE (30 s windows, n = 110)", "vitals_summary.json",
     "dog_window30s.hr.mae", "bpm2"),
    ("Vital signs", "Dog BR MAE (30 s windows)", "vitals_summary.json",
     "dog_window30s.br.mae", "bpm2"),
    ("Vital signs", "120 GHz HR MAE (recording level, n = 48)", "vitals_summary.json",
     "vs120_recording.hr.mae", "bpm2"),
    ("Vital signs", "120 GHz HR MAE (30 s windows, n = 336)", "vitals_summary.json",
     "vs120_window30s.hr.mae", "bpm2"),
    ("Vital signs", "120 GHz BR MAE (recording level)", "vitals_summary.json",
     "vs120_recording.br.mae", "bpm2"),
    ("Vital signs", "60 GHz resting HR MAE (n = 220)", "vitals_summary.json",
     "p60.full.rest.mae", "bpm2"),
    ("Vital signs", "60 GHz post-exercise HR MAE (n = 220)", "vitals_summary.json",
     "p60.full.post_exercise.mae", "bpm2"),
    ("Vital signs", "60 GHz HR MAE below 70 bpm", "vitals_summary.json",
     "p60_rest_by_refHR.0-70.mae", "bpm2"),
    ("Vital signs", "60 GHz HR MAE 70-90 bpm", "vitals_summary.json",
     "p60_rest_by_refHR.70-90.mae", "bpm2"),
    ("Vital signs", "60 GHz HR MAE above 90 bpm", "vitals_summary.json",
     "p60_rest_by_refHR.90-300.mae", "bpm2"),
    # --- ablation --------------------------------------------------------
    ("Ablation", "Dog HR MAE, no harmonic notch", "vitals_summary.json",
     "dog_recording.no_harmonic_notch.hr.mae", "bpm2"),
    ("Ablation", "Dog HR MAE, single strongest range bin", "vitals_summary.json",
     "dog_recording.single_bin_peak.hr.mae", "bpm2"),
    ("Ablation", "Dog BR MAE, single strongest range bin", "vitals_summary.json",
     "dog_recording.single_bin_peak.br.mae", "bpm2"),
    ("Ablation", "Dog HR MAE, plain median fusion", "vitals_summary.json",
     "dog_recording.plain_median_fusion.hr.mae", "bpm2"),
    ("Ablation", "Dog HR MAE, maximum-SNR channel only", "vitals_summary.json",
     "dog_recording.max_snr_channel.hr.mae", "bpm2"),
    ("Ablation", "Dog HR MAE, no cardiac/respiratory ratio gate", "vitals_summary.json",
     "dog_recording.no_ratio_gate.hr.mae", "bpm2"),
    ("Ablation", "60 GHz resting HR MAE, no harmonic notch", "vitals_summary.json",
     "p60.no_harmonic_notch.rest.mae", "bpm2"),
    # --- closed set ------------------------------------------------------
    ("Closed set", "P0 random-window accuracy (RF, 19-D)", "identification_summary.json",
     "closed_set.P0.resp+cardiac.RF.mean", "pct1"),
    ("Closed set", "P0 fold standard deviation", "identification_summary.json",
     "closed_set.P0.resp+cardiac.RF.sd", "pct1"),
    ("Closed set", "P1 temporal-block accuracy (RF)", "identification_summary.json",
     "closed_set.P1.resp+cardiac.RF", "pct1"),
    ("Closed set", "P1 training windows", "identification_summary.json",
     "closed_set.P1.n_train", "int"),
    ("Closed set", "P1 test windows", "identification_summary.json",
     "closed_set.P1.n_test", "int"),
    ("Closed set", "P2 25 min-horizon accuracy (RF)", "identification_summary.json",
     "closed_set.P2.resp+cardiac.RF", "pct1"),
    ("Closed set", "Best classifier at P2 (all 19 descriptors)", "identification_summary.json",
     "closed_set.P2.all19.RF", "pct1"),
    # --- open set --------------------------------------------------------
    ("Open set", "Fixed split, Mahalanobis accept-known", "identification_summary.json",
     "open_set.fixed_split_anaesth_known.Mahalanobis.accept_known", "pct1"),
    ("Open set", "Fixed split, Mahalanobis reject-unknown", "identification_summary.json",
     "open_set.fixed_split_anaesth_known.Mahalanobis.reject_unknown", "pct1"),
    ("Open set", "Fixed split, Mahalanobis AUROC", "identification_summary.json",
     "open_set.fixed_split_anaesth_known.Mahalanobis.auroc", "r3"),
    ("Open set", "20 random partitions, Mahalanobis AUROC", "identification_summary.json",
     "open_set.random_partitions.Mahalanobis.auroc.mean", "r3"),
    ("Open set", "20 random partitions, Mahalanobis AUROC s.d.", "identification_summary.json",
     "open_set.random_partitions.Mahalanobis.auroc.sd", "r3"),
    ("Open set", "20 random partitions, reject-unknown", "identification_summary.json",
     "open_set.random_partitions.Mahalanobis.reject_unknown.mean", "pct1"),
    ("Open set", "20 random partitions, MaxProb AUROC", "identification_summary.json",
     "open_set.random_partitions.MaxProb.auroc.mean", "r3"),
    ("Open set", "20 random partitions, top-two-margin AUROC", "identification_summary.json",
     "open_set.random_partitions.Top2Margin.auroc.mean", "r3"),
    # --- cross species and controls --------------------------------------
    ("Species", "Dogs vs 110 adults, accuracy (RF, grouped CV)", "cross_species_summary.json",
     "B_all110_grouped.RF.acc_mean", "pct1"),
    ("Species", "Dogs vs 110 adults, balanced accuracy", "cross_species_summary.json",
     "B_all110_grouped.RF.balanced_acc", "pct1"),
    ("Species", "Dogs vs 110 adults, AUROC", "cross_species_summary.json",
     "B_all110_grouped.RF.auroc", "r3"),
    ("Species", "Windows: dogs / humans", "cross_species_summary.json",
     "B_all110_grouped.RF", "windows"),
    ("Species", "NC1 cross-platform control, balanced accuracy", "cross_species_summary.json",
     "NC1_human60_vs_human120.inv16.balanced_acc", "pct1"),
    ("Species", "NC1 with grid-dependent descriptors", "cross_species_summary.json",
     "NC1_human60_vs_human120.inv16+grid5.balanced_acc", "pct1"),
    ("Species", "NC2 anaesthetised vs awake dogs", "cross_species_summary.json",
     "NC2_dog_scen1_vs_scen2.inv16.balanced_acc", "pct1"),
    ("Species", "NC3 human rest vs post-exercise", "cross_species_summary.json",
     "NC3_p60_rest_vs_post.inv16.balanced_acc", "pct1"),
    ("Species", "Transfer to 120 GHz, fraction called human", "cross_species_summary.json",
     "D_transfer_to_120GHz.RF.frac_predicted_human", "pct1"),
    ("Species", "Transfer to 60 GHz, fraction called human", "cross_species_summary.json",
     "D_transfer_to_60GHz.RF.frac_predicted_human", "pct1"),
    ("Species", "Three-domain balanced accuracy", "cross_species_summary.json",
     "E_three_domain_grouped.balanced_acc", "pct1"),
    ("Species", "Permutation test p (dog vs 60 GHz)", "cross_species_summary.json",
     "F_permutation_dog_vs_p60.p", "p3"),
    ("Species", "Effect size d, hr_power_frac, species", "cross_species_summary.json",
     "G_effect_size_dog_minus_human60.hr_power_frac", "signed2"),
    ("Species", "Effect size d, br_power_frac, hardware", "cross_species_summary.json",
     "G_effect_size_human60_minus_human120.br_power_frac", "signed2"),
    ("Species", "Effect size d, cardiorespiratory coupling, species", "cross_species_summary.json",
     "G_effect_size_dog_minus_human60.cardioresp_coupling", "signed2"),
    # --- rate matching and age -------------------------------------------
    ("Rate match", "Grid-dependent descriptors, native 50 Hz", "rate_matched_age_summary.json",
     "rate_matched.dog_native_50Hz.grid5_only.balanced_acc", "pct1"),
    ("Rate match", "Grid-dependent descriptors, decimated 10 Hz", "rate_matched_age_summary.json",
     "rate_matched.dog_decimated_10Hz.grid5_only.balanced_acc", "pct1"),
    ("Rate match", "Invariant descriptors, native 50 Hz", "rate_matched_age_summary.json",
     "rate_matched.dog_native_50Hz.inv16.balanced_acc", "pct1"),
    ("Rate match", "Invariant descriptors, decimated 10 Hz", "rate_matched_age_summary.json",
     "rate_matched.dog_decimated_10Hz.inv16.balanced_acc", "pct1"),
    ("Rate match", "Combined set, native 50 Hz", "rate_matched_age_summary.json",
     "rate_matched.dog_native_50Hz.inv16+grid5.balanced_acc", "pct1"),
    ("Rate match", "Combined set, decimated 10 Hz", "rate_matched_age_summary.json",
     "rate_matched.dog_decimated_10Hz.inv16+grid5.balanced_acc", "pct1"),
    ("Rate match", "Median drift, spectral_centroid", "rate_matched_age_summary.json",
     "drift_50_to_10Hz.spectral_centroid.median_rel_change", "pct1"),
    ("Rate match", "Median drift, hrv_rmssd_s", "rate_matched_age_summary.json",
     "drift_50_to_10Hz.hrv_rmssd_s.median_rel_change", "pct1"),
    ("Age", "Random forest MAE (years)", "rate_matched_age_summary.json",
     "age.RF.mae", "y1"),
    ("Age", "Random forest R-squared", "rate_matched_age_summary.json",
     "age.RF.r2", "signed2"),
    ("Age", "Ridge MAE (years)", "rate_matched_age_summary.json",
     "age.Ridge.mae", "y1"),
    ("Age", "Mean-predictor MAE (years)", "rate_matched_age_summary.json",
     "age.Mean.mae", "y1"),
    ("Age", "Ridge permutation p", "rate_matched_age_summary.json",
     "age.ridge_perm_p", "p2"),
    ("Age", "Subjects / age range", "rate_matched_age_summary.json",
     "age", "ageinfo"),
]


def get(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:  # pragma: no cover - paths are fixed above
            raise KeyError(dotted)
    return cur


def fmt(kind, value):
    if kind == "bpm1":
        return f"{value:.1f} bpm"
    if kind == "bpm2":
        return f"{value:.2f} bpm"
    if kind == "signed1":
        return f"{value:+.1f} bpm"
    if kind == "loa":
        return f"{value['loa_low']:.1f} to {value['loa_high']:.1f} bpm"
    if kind == "r2":
        return f"{value:.2f}"
    if kind == "r3":
        return f"{value:.3f}"
    if kind == "pct1":
        return f"{100 * value:.1f} %"
    if kind == "p3":
        return f"p = {value:.3f}"
    if kind == "p2":
        return f"p = {value:.2f}"
    if kind == "signed2":
        return f"{value:+.2f}"
    if kind == "int":
        return f"{int(value)}"
    if kind == "y1":
        return f"{value:.1f} years"
    if kind == "windows":
        return f"{value['n0']} dog / {value['n1']} human windows"
    if kind == "ageinfo":
        lo, hi = value["age_range"]
        return f"{value['n_subjects']} subjects, {lo:.0f}-{hi:.0f} years"
    raise ValueError(kind)


def main() -> None:
    cache: dict[str, dict] = {}
    sections: dict[str, list[str]] = {}
    for section, claim, fname, path, kind in ROWS:
        if fname not in cache:
            cache[fname] = json.loads((RES / fname).read_text(encoding="utf-8"))
        value = fmt(kind, get(cache[fname], path))
        sections.setdefault(section, []).append(
            f"| {claim} | {value} | `results/{fname}` → `{path}` |"
        )

    lines = [
        "# Number table",
        "",
        "Every number quoted in the manuscript, with the committed result file and",
        "field it comes from.  Generated by `python scripts/07_number_table.py`; do",
        "not edit by hand.  The values are the ones the test suite checks, so a",
        "number that changes in the code cannot silently stay in the paper.",
        "",
    ]
    for section, rows in sections.items():
        lines += [f"## {section}", "", "| Quantity | Value | Source |", "|---|---|---|", *rows, ""]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({sum(len(v) for v in sections.values())} rows)")


if __name__ == "__main__":
    main()
