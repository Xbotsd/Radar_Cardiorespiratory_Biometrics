"""Extract all window-level feature tables from the raw public data.

Outputs (in ``results/features``):
  id19_dogs.csv     19-D identification features, 15 s / 3 s windows, every
                    window of every dog (Scenario 2 dogs: full 1800 s).
  inv16_dogs.csv    16-D invariant + 5 rate-dependent features, 20 s / 10 s
                    windows, highest-variance range bin (first 300 s).
  inv16_p60.csv     same for the 60 GHz age-balanced data, all 110 subjects,
                    rest and post-exercise, highest-variance range bin.
  inv16_vs120.csv   same for VITALSENSE 120 GHz (single displacement channel).
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcb import config as C  # noqa: E402
from rcb import io  # noqa: E402
from rcb.dsp import sliding_windows  # noqa: E402
from rcb.features import id19, inv16, rate_dependent  # noqa: E402

OUT = ROOT / "results" / "features"
OUT.mkdir(parents=True, exist_ok=True)


def _id19_subject(subj):
    M = io.load_dog_radar(subj)
    fs = C.DOG_FS
    win, hop = int(15 * fs), int(3 * fs)
    rows = []
    for k, s in enumerate(sliding_windows(len(M), win, hop)):
        F = id19(M[s:s + win], fs)
        rows.append({"subject": subj, "scenario": io.dog_scenario(subj), "win": k,
                     "start_s": s / fs, **F})
    return rows


def _inv_rows(sig, fs, meta, win_s=20, hop_s=10, max_windows=None):
    win, hop = int(win_s * fs), int(hop_s * fs)
    rows, n = [], 0
    for s in sliding_windows(len(sig), win, hop):
        seg = sig[s:s + win]
        F = inv16(seg, fs)
        if F is None:
            continue
        R = rate_dependent(seg, fs)
        rows.append({**meta, "win": n, "start_s": s / fs, **F, **R})
        n += 1
        if max_windows and n >= max_windows:
            break
    return rows


def _inv_dog(subj):
    M = io.load_dog_radar(subj, max_seconds=300)          # first 15000 samples
    sig = M[:, int(np.argmax(M.std(axis=0)))]
    return _inv_rows(sig, C.DOG_FS, {"dataset": "dog_uwb7g", "species": "dog",
                                      "subject": subj, "condition": f"scen{io.dog_scenario(subj)}"},
                     max_windows=20)


def _inv_p(args):
    subj, posture, state = args
    cube, _ = io.load_p_radar(subj, posture, state)
    if cube is None:
        return []
    ph = io.p_phase_matrix(cube)
    sig = ph[:, int(np.argmax(ph.std(axis=0)))]
    return _inv_rows(sig, C.P_FS, {"dataset": "human_fmcw60g", "species": "human",
                                    "subject": subj, "condition": f"{posture}/{state}"},
                     max_windows=10)


def _inv_vs(args):
    subj, scen = args
    d = io.load_vitalsense(subj, scen)
    if d is None:
        return []
    return _inv_rows(d["vital_sig"], C.VS_FS, {"dataset": "human_fmcw120g", "species": "human",
                                                "subject": subj, "condition": scen},
                     max_windows=10)


def _run(fn, items, name, workers=16):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(fn, items):
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / name, index=False, float_format="%.8g")
    print(f"{name}: {len(df)} rows", flush=True)
    return df


if __name__ == "__main__":
    _run(_inv_dog, C.DOG_ALL, "inv16_dogs.csv")
    _run(_inv_p, [(s, p, st) for s in C.P_SUBJECTS for p, st in C.P_REST + C.P_POST],
         "inv16_p60.csv")
    _run(_inv_vs, [(s, sc) for s in C.VS_SUBJECTS for sc in C.VS_SCENARIOS], "inv16_vs120.csv")
    _run(_id19_subject, C.DOG_ALL, "id19_dogs.csv", workers=12)
