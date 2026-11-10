"""Experiment 1 - vital-sign estimation and estimator ablation.

Recording level (as in the original analysis) and window level (30 s windows,
15 s hop, compared with the reference averaged over the same window).
Writes results/vitals_*.csv and results/vitals_summary.json.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcb import config as C  # noqa: E402
from rcb import io  # noqa: E402
from rcb.stats import agreement  # noqa: E402
from rcb.vitals import ecg_hr, multibin_estimate, single_channel_estimate  # noqa: E402

RES = ROOT / "results"

DOG_CFG = dict(br_band=(0.15, 0.45), hr_band=(1.6, 2.6), br_nperseg=4096, hr_nperseg=2048,
               ratio=(4.0, 12.0), hw=0.025)
DOG_VARIANTS = {
    "full": dict(),
    "no_harmonic_notch": dict(notch=False),
    "no_ratio_gate": dict(ratio=None),
    "plain_median_fusion": dict(fusion="median"),
    "max_snr_channel": dict(fusion="max_snr"),
}
P_CFG = dict(br_band=(0.1, 0.5), hr_band=(0.8, 3.0), br_nperseg=256, hr_nperseg=256,
             ratio=(2.5, 12.0), hw=0.05, prefilter=True, hr_bpm_range=(45, 140),
             br_bpm_range=(5, 30))


# ----------------------------------------------------------------------------
def dog_subject(subj):
    M = io.load_dog_radar(subj)
    ref = io.load_dog_reference(subj)
    rows = []
    for name, v in DOG_VARIANTS.items():
        e = multibin_estimate(M, C.DOG_FS, **{**DOG_CFG, **v})
        rows.append({"subject": subj, "variant": name, "level": "recording",
                     "hr_est": e["hr"], "br_est": e["br"],
                     "hr_ref": ref["hr"].mean(), "br_ref": ref["br"].mean()})
    # baseline: single highest-variance bin, plain spectral peak
    b = single_channel_estimate(M[:, int(np.argmax(M.std(axis=0)))], C.DOG_FS,
                                DOG_CFG["br_band"], DOG_CFG["hr_band"], 4096)
    rows.append({"subject": subj, "variant": "single_bin_peak", "level": "recording",
                 "hr_est": b["hr"], "br_est": b["br"],
                 "hr_ref": ref["hr"].mean(), "br_ref": ref["br"].mean()})
    # window level (full estimator)
    fs, win, hop = C.DOG_FS, int(30 * C.DOG_FS), int(15 * C.DOG_FS)
    for k, s in enumerate(range(0, len(M) - win + 1, hop)):
        e = multibin_estimate(M[s:s + win], fs, **DOG_CFG)
        t0, t1 = int(s / fs), int((s + win) / fs)
        if e is None:
            continue
        rows.append({"subject": subj, "variant": "full", "level": "window30s", "win": k,
                     "hr_est": e["hr"], "br_est": e["br"],
                     "hr_ref": ref["hr"][t0:t1].mean(), "br_ref": ref["br"][t0:t1].mean()})
    return rows


# ----------------------------------------------------------------------------
def vs_record(args):
    subj, scen = args
    d = io.load_vitalsense(subj, scen)
    if d is None or "hr_series" not in d:
        return []
    x, fs = d["vital_sig"], C.VS_FS
    rows = []
    e = single_channel_estimate(x, fs, (0.15, 0.45), (0.8, 2.6), 4096)
    rows.append({"subject": subj, "scenario": scen, "level": "recording",
                 "hr_est": e["hr"], "br_est": e["br"],
                 "hr_ref": d["hr_series"].mean(), "br_ref": d["rr_series"].mean()})
    win, hop = int(30 * fs), int(15 * fs)
    for k, s in enumerate(range(0, len(x) - win + 1, hop)):
        e = single_channel_estimate(x[s:s + win], fs, (0.15, 0.45), (0.8, 2.6), 4096)
        t0, t1 = int(s / fs), int((s + win) / fs)
        rows.append({"subject": subj, "scenario": scen, "level": "window30s", "win": k,
                     "hr_est": e["hr"], "br_est": e["br"],
                     "hr_ref": d["hr_series"][t0:t1].mean(), "br_ref": d["rr_series"][t0:t1].mean()})
    return rows


# ----------------------------------------------------------------------------
def p_record(args):
    subj, posture, state = args
    cube, ranges = io.load_p_radar(subj, posture, state)
    ecg = io.load_p_ecg(subj, posture, state)
    if cube is None or ecg is None:
        return []
    ref = ecg_hr(ecg, C.P_ECG_FS)
    if ref is None:
        return []
    ph = io.p_phase_matrix(cube)
    rw = np.exp(-((ranges - 3.0) ** 2) / (2 * 2.0 ** 2))
    rows = []
    for name, v in {"full": {}, "no_harmonic_notch": {"notch": False},
                    "no_range_prior": {"col_weights": None}}.items():
        kw = {**P_CFG, "col_weights": rw, **v}
        e = multibin_estimate(ph, C.P_FS, **kw)
        if e is None:
            continue
        rows.append({"subject": subj, "condition": f"{posture}/{state}", "variant": name,
                     "hr_est": e["hr"], "br_est": e["br"], "hr_ref": ref})
    b = single_channel_estimate(ph[:, int(np.argmax(ph.std(axis=0)))], C.P_FS,
                                (0.1, 0.5), (0.8, 3.0), 256)
    rows.append({"subject": subj, "condition": f"{posture}/{state}", "variant": "single_bin_peak",
                 "hr_est": b["hr"], "br_est": b["br"], "hr_ref": ref})
    return rows


def _pool(fn, items, w=16):
    out = []
    with ProcessPoolExecutor(max_workers=w) as ex:
        for r in ex.map(fn, items):
            out.extend(r)
    return pd.DataFrame(out)


def _agree(df, est, ref):
    return agreement(df[est].values, df[ref].values)


if __name__ == "__main__":
    S = {}
    dog = _pool(dog_subject, C.DOG_SCEN1, 10)
    dog.to_csv(RES / "vitals_dogs.csv", index=False, float_format="%.4f")
    rec = dog[dog.level == "recording"]
    S["dog_recording"] = {v: {"hr": _agree(g, "hr_est", "hr_ref"), "br": _agree(g, "br_est", "br_ref")}
                          for v, g in rec.groupby("variant")}
    w = dog[dog.level == "window30s"]
    S["dog_window30s"] = {"hr": _agree(w, "hr_est", "hr_ref"), "br": _agree(w, "br_est", "br_ref")}

    vs = _pool(vs_record, [(s, sc) for s in C.VS_SUBJECTS for sc in C.VS_SCENARIOS])
    vs.to_csv(RES / "vitals_vs120.csv", index=False, float_format="%.4f")
    for lev, g in vs.groupby("level"):
        S[f"vs120_{lev}"] = {"hr": _agree(g, "hr_est", "hr_ref"), "br": _agree(g, "br_est", "br_ref"),
                             "by_scenario": {sc: {"hr": _agree(h, "hr_est", "hr_ref"),
                                                  "br": _agree(h, "br_est", "br_ref")}
                                             for sc, h in g.groupby("scenario")}}

    p = _pool(p_record, [(s, a, b) for s in C.P_SUBJECTS for a, b in C.P_REST + C.P_POST])
    p.to_csv(RES / "vitals_p60.csv", index=False, float_format="%.4f")
    S["p60"] = {}
    for v, g in p.groupby("variant"):
        rest = g[g.condition.str.endswith("/Rest")]
        post = g[g.condition.str.endswith("Post-exercise")]
        S["p60"][v] = {"rest": _agree(rest, "hr_est", "hr_ref"),
                       "post_exercise": _agree(post, "hr_est", "hr_ref"),
                       "all": _agree(g, "hr_est", "hr_ref")}
    full_rest = p[(p.variant == "full") & p.condition.str.endswith("/Rest")]
    bins = [(0, 70), (70, 90), (90, 300)]
    S["p60_rest_by_refHR"] = {f"{a}-{b}": _agree(full_rest[(full_rest.hr_ref >= a) & (full_rest.hr_ref < b)],
                                                 "hr_est", "hr_ref") for a, b in bins}
    full_all = p[p.variant == "full"]
    S["p60_all_by_refHR"] = {f"{a}-{b}": _agree(full_all[(full_all.hr_ref >= a) & (full_all.hr_ref < b)],
                                                "hr_est", "hr_ref") for a, b in bins}
    with open(RES / "vitals_summary.json", "w") as f:
        json.dump(S, f, indent=2)
    print(json.dumps({k: (v if k != "p60" else {kk: vv["rest"]["mae"] for kk, vv in v.items()})
                      for k, v in S.items() if k in ("dog_recording", "p60")}, indent=1)[:3000])
