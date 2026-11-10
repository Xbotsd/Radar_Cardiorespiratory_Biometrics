"""Experiment 2 - individual identification and open-set verification (30 dogs).

Protocols
---------
P0  original: 5-fold stratified CV over windows (first 56 windows per dog,
    15 s windows / 3 s hop).  Overlapping windows of the same recording fall
    into train and test -> optimistic (reported only for reproducibility).
P1  temporal block split inside the same 56-window segment: first 60 % of the
    windows train, last 40 % test, with a 5-window (15 s) guard gap so that no
    test window overlaps a training window.
P2  long-horizon (awake dogs No11-No30, 1800 s): train on the first 56
    windows (~3 min), test on windows that start after 25 min.

Open set: repeated random partitions of the 30 dogs into 10 known / 20
unknown; known windows split temporally (fit / calibration / test with guard
gaps); acceptance threshold set on calibration windows only (90 % target).
The original fixed split (known = anaesthetised No1-No10) is also reported.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcb import config as C  # noqa: E402
from rcb.features import ID_GROUPS  # noqa: E402

RES = ROOT / "results"
GAP = 5          # windows; 5 x 3 s hop = 15 s = one window length
CAP = 56


def models():
    return {
        "RF": lambda: make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=300, random_state=C.SEED, n_jobs=-1)),
        "SVM-RBF": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "LDA": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "kNN-5": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    }


def temporal_split(df, frac=0.6, gap=GAP):
    tr, te = [], []
    for _, g in df.groupby("subject"):
        g = g.sort_values("win")
        n = len(g)
        k = int(round(frac * n))
        tr.append(g.iloc[:k]); te.append(g.iloc[k + gap:])
    return pd.concat(tr), pd.concat(te)


def acc(m, Xtr, ytr, Xte, yte):
    m.fit(Xtr, ytr)
    return float((m.predict(Xte) == yte).mean())


def closed_set(df):
    capped = df[df.win < CAP].copy()
    y = capped.subject.map({s: i for i, s in enumerate(C.DOG_ALL)}).values
    out = {"n_windows_P0": int(len(capped))}
    # P0: original random 5-fold
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    out["P0"] = {}
    for gname, feats in ID_GROUPS.items():
        X = capped[feats].values
        out["P0"][gname] = {}
        for mname, mk in models().items():
            s = [acc(mk(), X[a], y[a], X[b], y[b]) for a, b in cv.split(X, y)]
            out["P0"][gname][mname] = {"mean": float(np.mean(s)), "sd": float(np.std(s)), "folds": s}
    # P1: temporal block with guard gap
    tr, te = temporal_split(capped)
    lab = {s: i for i, s in enumerate(C.DOG_ALL)}
    out["P1"] = {"n_train": int(len(tr)), "n_test": int(len(te))}
    for gname, feats in ID_GROUPS.items():
        out["P1"][gname] = {m: acc(mk(), tr[feats].values, tr.subject.map(lab).values,
                                   te[feats].values, te.subject.map(lab).values)
                            for m, mk in models().items()}
    # P1 ablation: guard gap size and fraction (RF, resp+cardiac)
    feats = ID_GROUPS["resp+cardiac"]
    out["P1_gap_sweep"] = {}
    for gap in [0, 1, 3, 5, 10]:
        tr2, te2 = temporal_split(capped, gap=gap)
        out["P1_gap_sweep"][gap] = acc(models()["RF"](), tr2[feats].values, tr2.subject.map(lab).values,
                                       te2[feats].values, te2.subject.map(lab).values)
    # P2: long horizon (awake dogs)
    aw = df[df.scenario == 2]
    lab2 = {s: i for i, s in enumerate(C.DOG_SCEN2)}
    tr = aw[aw.win < CAP]
    te = aw[aw.start_s >= 25 * 60]
    out["P2"] = {"n_train": int(len(tr)), "n_test": int(len(te)), "chance": 1 / 20}
    for gname, feats in ID_GROUPS.items():
        out["P2"][gname] = {m: acc(mk(), tr[feats].values, tr.subject.map(lab2).values,
                                   te[feats].values, te.subject.map(lab2).values)
                            for m, mk in models().items()}
    # same 20 dogs, P1-style temporal split on first 56 windows (reference for P2)
    aw56 = aw[aw.win < CAP]
    tr, te = temporal_split(aw56)
    out["P1_awake20"] = {m: acc(mk(), tr[feats].values, tr.subject.map(lab2).values,
                                te[feats].values, te.subject.map(lab2).values)
                         for m, mk in models().items()}
    return out


# ----------------------------------------------------------------------------
def open_set(df, n_rep=20):
    feats = ID_GROUPS["resp+cardiac"]
    capped = df[df.win < CAP]
    out = {}

    def run(known, unknown):
        return open_set_run_raw(capped, known, unknown, feats)
    out["fixed_split_anaesth_known"] = run(C.DOG_SCEN1, C.DOG_SCEN2)
    rng = np.random.default_rng(C.SEED)
    reps = []
    for r in range(n_rep):
        perm = rng.permutation(C.DOG_ALL)
        reps.append(run(list(perm[:10]), list(perm[10:])))
    keys = [k for k in reps[0] if isinstance(reps[0][k], dict)]
    out["random_partitions"] = {
        k: {m: {"mean": float(np.mean([r[k][m] for r in reps])),
                "sd": float(np.std([r[k][m] for r in reps]))}
            for m in reps[0][k]} for k in keys}
    out["random_partitions"]["closed_set_acc_known_test"] = {
        "mean": float(np.mean([r["closed_set_acc_known_test"] for r in reps])),
        "sd": float(np.std([r["closed_set_acc_known_test"] for r in reps]))}
    # within-scenario partitions (awake dogs only: 7 known / 13 unknown)
    reps2 = []
    for r in range(n_rep):
        perm = rng.permutation(C.DOG_SCEN2)
        reps2.append(run(list(perm[:7]), list(perm[7:])))
    out["awake_only_partitions"] = {
        k: {m: {"mean": float(np.mean([r[k][m] for r in reps2])),
                "sd": float(np.std([r[k][m] for r in reps2]))}
            for m in reps2[0][k]} for k in keys}
    # original (leaky) threshold rule for reference: threshold from known *test* scores
    return out


def open_set_run_raw(df, known, unknown, feats, target=0.90):
    kn = df[df.subject.isin(known)]
    un = df[df.subject.isin(unknown)]
    fit, cal, test = [], [], []
    for _, g in kn.groupby("subject"):
        g = g.sort_values("win"); n = len(g)
        a, b = int(0.45 * n), int(0.6 * n)
        fit.append(g.iloc[:a]); cal.append(g.iloc[a + GAP:b]); test.append(g.iloc[b + GAP:])
    fit, cal, test = map(pd.concat, (fit, cal, test))
    Xf, yf = fit[feats].values, fit.subject.values
    sc = StandardScaler().fit(Xf)
    A = sc.transform(Xf)
    rf = RandomForestClassifier(300, random_state=C.SEED, n_jobs=-1).fit(A, yf)
    classes = np.unique(yf)
    mu = np.stack([A[yf == c].mean(0) for c in classes])
    ci = np.linalg.inv(np.cov(A.T) + 1e-3 * np.eye(A.shape[1]))

    def S(X):
        Q = sc.transform(X)
        d = Q[:, None, :] - mu[None]
        maha = np.sqrt(np.einsum("nkd,de,nke->nk", d, ci, d)).min(1)
        eucl = np.linalg.norm(d, axis=2).min(1)
        P = np.sort(rf.predict_proba(Q), 1)
        return {"Mahalanobis": -maha, "Euclidean": -eucl, "MaxProb": P[:, -1],
                "Top2Margin": P[:, -1] - P[:, -2]}
    s_cal, s_te, s_un = S(cal[feats].values), S(test[feats].values), S(un[feats].values)
    res = {}
    for k in s_cal:
        thr = np.quantile(s_cal[k], 1 - target)
        a = float((s_te[k] >= thr).mean()); rj = float((s_un[k] < thr).mean())
        y = np.r_[np.ones(len(s_te[k])), np.zeros(len(s_un[k]))]
        res[k] = {"accept_known": a, "reject_unknown": rj,
                  "auroc": float(roc_auc_score(y, np.r_[s_te[k], s_un[k]])),
                  "balanced": 0.5 * (a + rj)}
    res["closed_set_acc_known_test"] = float((rf.predict(sc.transform(test[feats].values))
                                              == test.subject.values).mean())
    return res


if __name__ == "__main__":
    df = pd.read_csv(RES / "features" / "id19_dogs.csv")
    out = {"closed_set": closed_set(df), "open_set": open_set(df)}
    with open(RES / "identification_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    cs = out["closed_set"]
    print("P0 RF resp+cardiac", cs["P0"]["resp+cardiac"]["RF"]["mean"])
    print("P1", {g: {m: round(v, 3) for m, v in d.items()} for g, d in cs["P1"].items() if isinstance(d, dict)})
    print("gap sweep", cs["P1_gap_sweep"])
    print("P2", {g: {m: round(v, 3) for m, v in d.items()} for g, d in cs["P2"].items() if isinstance(d, dict)})
    print("P1 awake20", cs["P1_awake20"])
    for k, v in out["open_set"].items():
        print(k, json.dumps(v)[:900])
