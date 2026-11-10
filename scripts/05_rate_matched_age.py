"""Experiment 4 - rate-matched control and age negative result.

(a) Rate-matched control: the dog radar channel is first decimated to the
    10 Hz frame rate of the 60 GHz human data (polyphase anti-alias filter),
    then the same windows are re-featurised.  If a descriptor encodes the
    sampling grid, the dog-vs-human separability attributable to it should
    collapse once the grids are matched; grid-invariant descriptors should
    be unaffected.
(b) Feature drift: per-descriptor relative change between the native 50 Hz
    and the 10 Hz version of the same dog windows.
(c) Age: subject-level regression / 3-class grouping on the 110-subject
    60 GHz set (rest recordings, median of windows per subject), compared
    with mean-prediction and majority baselines and a permutation test.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rcb import config as C  # noqa: E402
from rcb import io  # noqa: E402
from rcb.dsp import resample_to, sliding_windows  # noqa: E402
from rcb.features import INV16_NAMES, RATE_DEPENDENT, inv16, rate_dependent  # noqa: E402

RES = ROOT / "results"


def dog_at_10hz(subj):
    M = io.load_dog_radar(subj, max_seconds=300)
    sig = M[:, int(np.argmax(M.std(axis=0)))]
    x10 = resample_to(sig - sig.mean(), C.DOG_FS, C.P_FS)
    fs = C.P_FS
    win, hop = int(20 * fs), int(10 * fs)
    rows, n = [], 0
    for s in sliding_windows(len(x10), win, hop):
        seg = x10[s:s + win]
        F = inv16(seg, fs)
        if F is None:
            continue
        rows.append({"dataset": "dog_uwb7g_10Hz", "species": "dog", "subject": subj,
                     "condition": f"scen{io.dog_scenario(subj)}", "win": n, **F, **rate_dependent(seg, fs)})
        n += 1
        if n >= 20:
            break
    return rows


if __name__ == "__main__":
    import importlib
    cs = importlib.import_module("04_cross_species")
    S = {}
    with ProcessPoolExecutor(12) as ex:
        rows = [r for rr in ex.map(dog_at_10hz, C.DOG_ALL) for r in rr]
    d10 = pd.DataFrame(rows)
    d10.to_csv(RES / "features" / "inv16_dogs_10Hz.csv", index=False, float_format="%.8g")
    d50 = pd.read_csv(RES / "features" / "inv16_dogs.csv")
    p60 = pd.read_csv(RES / "features" / "inv16_p60.csv")
    inv, grid = INV16_NAMES, RATE_DEPENDENT

    def pair(A, B, feats):
        D = pd.concat([A.assign(y=0), B.assign(y=1)], ignore_index=True)
        return cs.cv_eval(D[feats].values, D.y.values, (D.dataset + ":" + D.subject).values, "RF", True)
    S["rate_matched"] = {}
    for name, dg in [("dog_native_50Hz", d50), ("dog_decimated_10Hz", d10)]:
        S["rate_matched"][name] = {"inv16": pair(dg, p60, inv), "grid5_only": pair(dg, p60, grid),
                                   "inv16+grid5": pair(dg, p60, inv + grid)}
    # drift of each descriptor (same windows, 50 Hz vs 10 Hz)
    m = d50.merge(d10, on=["subject", "win"], suffixes=("_50", "_10"))
    drift = {}
    for f in inv + grid:
        a, b = m[f + "_50"].values, m[f + "_10"].values
        scale = np.median(np.abs(a)) + 1e-12
        drift[f] = {"median_rel_change": float(np.median(np.abs(b - a)) / scale),
                    "median_50": float(np.median(a)), "median_10": float(np.median(b))}
    S["drift_50_to_10Hz"] = drift
    S["n_matched_windows"] = int(len(m))

    # ---------------------------------------------------------------- age ----
    info = io.load_participants()
    age = dict(zip(info["PARTICIPANT ID"], info["AGE"]))
    rest = p60[p60.condition.str.endswith("/Rest")]
    subj = rest.groupby("subject")[inv].median()
    subj = subj[subj.index.isin(age)]
    X = subj.values
    y = np.array([age[s] for s in subj.index], float)
    kf = KFold(5, shuffle=True, random_state=C.SEED)
    A = {"n_subjects": int(len(y)), "age_range": [float(y.min()), float(y.max())]}
    for name, model in {"RF": make_pipeline(StandardScaler(), RandomForestRegressor(300, random_state=C.SEED)),
                        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
                        "Mean": DummyRegressor()}.items():
        pr = cross_val_predict(model, X, y, cv=kf)
        A[name] = {"mae": float(mean_absolute_error(y, pr)), "r2": float(r2_score(y, pr))}
    grp = np.digitize(y, [35, 55.5])
    skf = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    pr = cross_val_predict(make_pipeline(StandardScaler(), RandomForestClassifier(300, random_state=C.SEED)),
                           X, grp, cv=skf)
    A["group3"] = {"balanced_acc": float(balanced_accuracy_score(grp, pr)),
                   "acc": float((pr == grp).mean()), "counts": np.bincount(grp).tolist()}
    rng = np.random.default_rng(1)
    null = []
    for _ in range(200):
        yp = rng.permutation(y)
        null.append(mean_absolute_error(yp, cross_val_predict(
            make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))), X, yp, cv=kf)))
    A["ridge_perm_p"] = float((1 + (np.array(null) <= A["Ridge"]["mae"]).sum()) / 201)
    S["age"] = A
    with open(RES / "rate_matched_age_summary.json", "w") as f:
        json.dump(S, f, indent=2)
    for k, v in S["rate_matched"].items():
        print(k, {kk: (round(vv["balanced_acc"], 3), round(vv["auroc"], 3)) for kk, vv in v.items()})
    print({k: round(v["median_rel_change"], 3) for k, v in drift.items()})
    print(json.dumps(A))
