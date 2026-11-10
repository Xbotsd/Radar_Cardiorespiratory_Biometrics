"""Experiment 3 - cross-species discrimination and hardware negative controls.

Windows: 20 s / 10 s hop, highest-variance bin, 16-D invariant set computed
on a common 20 Hz grid; 5 grid-dependent descriptors computed at the native
rate are added only for the "inv16+grid5" negative control.

Protocols: window-stratified 5-fold (original, subjects shared across folds)
versus subject-grouped 5-fold (StratifiedGroupKFold).  Balanced accuracy and
AUROC are reported besides accuracy.

Negative controls (same species, different acquisition):
  NC1 human 60 GHz FMCW  vs human 120 GHz FMCW
  NC2 dog anaesthetised (Scenario 1 setup) vs dog awake (Scenario 2 setup)
  NC3 60 GHz rest vs 60 GHz post-exercise (same radar, physiological change)
Transfer test: train dog vs 60 GHz human, predict 120 GHz human windows.
Permutation test: subject-level label shuffling (grouped CV), 200 repeats.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcb import config as C  # noqa: E402
from rcb.features import INV16_NAMES, RATE_DEPENDENT  # noqa: E402

RES = ROOT / "results"
FEAT = RES / "features"


def mk(name):
    if name == "RF":
        return make_pipeline(StandardScaler(), RandomForestClassifier(200, random_state=C.SEED, n_jobs=-1))
    if name == "LogReg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    if name == "SVM-RBF":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale", probability=True,
                                                   random_state=C.SEED))
    raise ValueError(name)


def cv_eval(X, y, groups, model="RF", grouped=True, seed=C.SEED):
    cv = (StratifiedGroupKFold(5, shuffle=True, random_state=seed) if grouped
          else StratifiedKFold(5, shuffle=True, random_state=seed))
    split = cv.split(X, y, groups) if grouped else cv.split(X, y)
    pred = np.zeros(len(y)); prob = np.zeros(len(y)); accs = []
    for a, b in split:
        m = mk(model).fit(X[a], y[a])
        pred[b] = m.predict(X[b]); prob[b] = m.predict_proba(X[b])[:, 1]
        accs.append(float((pred[b] == y[b]).mean()))
    return {"acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)), "folds": accs,
            "balanced_acc": float(balanced_accuracy_score(y, pred)),
            "auroc": float(roc_auc_score(y, prob)),
            "majority": float(max(y.mean(), 1 - y.mean())),
            "n0": int((y == 0).sum()), "n1": int((y == 1).sum()),
            "n_groups": int(len(np.unique(groups)))}


def iqr_filter(X, k=3.0):
    m = np.ones(len(X), bool)
    for i in range(X.shape[1]):
        q1, q3 = np.percentile(X[:, i], [25, 75]); r = q3 - q1
        m &= (X[:, i] >= q1 - k * r) & (X[:, i] <= q3 + k * r)
    return m


def permutation_p(X, y, groups, observed, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    glab = {g: y[groups == g][0] for g in ug}
    null = []
    for _ in range(n):
        perm = dict(zip(ug, rng.permutation([glab[g] for g in ug])))
        yp = np.array([perm[g] for g in groups])
        null.append(cv_eval(X, yp, groups, "LogReg", True, seed=int(rng.integers(1e9)))["balanced_acc"])
    null = np.array(null)
    return float((1 + (null >= observed).sum()) / (n + 1)), float(null.mean()), float(np.percentile(null, 95))


def effect_sizes(A, B, names):
    out = {}
    for n in names:
        a, b = A[n].values, B[n].values
        s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
        out[n] = float((a.mean() - b.mean()) / s)
    return out


if __name__ == "__main__":
    dog = pd.read_csv(FEAT / "inv16_dogs.csv")
    p60 = pd.read_csv(FEAT / "inv16_p60.csv")
    vs = pd.read_csv(FEAT / "inv16_vs120.csv")
    p60_rest = p60[p60.condition.str.endswith("/Rest")]
    p60_rest30 = p60_rest[p60_rest.subject.isin(C.P_SUBJECTS[:30])]
    S = {}

    def pair(A, B, feats, grouped, model="RF", filt=False):
        D = pd.concat([A.assign(y=0), B.assign(y=1)], ignore_index=True)
        X, y, g = D[feats].values, D.y.values, (D.dataset + ":" + D.subject).values
        if filt:
            m = iqr_filter(X); X, y, g = X[m], y[m], g[m]
        return cv_eval(X, y, g, model, grouped)

    inv, rich = INV16_NAMES, INV16_NAMES + RATE_DEPENDENT
    # --- A. replication of the original configuration (P001-P030 rest) ---------
    S["A_original_config"] = {
        "inv16+grid5_window_cv_iqr": pair(dog, p60_rest30, rich, False, filt=True),
        "inv16_window_cv_iqr": pair(dog, p60_rest30, inv, False, filt=True),
        "inv16_window_cv": pair(dog, p60_rest30, inv, False),
        "inv16_grouped_cv": pair(dog, p60_rest30, inv, True),
        "inv16+grid5_grouped_cv": pair(dog, p60_rest30, rich, True),
    }
    # --- B. all 110 humans, rest + post-exercise, grouped CV, models -----------
    S["B_all110_grouped"] = {m: pair(dog, p60, inv, True, m) for m in ["RF", "LogReg", "SVM-RBF"]}
    S["B_all110_window"] = {m: pair(dog, p60, inv, False, m) for m in ["RF", "LogReg"]}
    # feature-group ablation (RF, grouped)
    groups = {
        "power_fractions_only": ["br_power_frac", "hr_power_frac", "br_hr_power_ratio"],
        "no_power_fractions": [f for f in inv if f not in ("br_power_frac", "hr_power_frac", "br_hr_power_ratio")],
        "rates_only": ["br_freq_hz", "hr_freq_hz", "hr_br_ratio", "br_period_s"],
        "morphology_only": ["br_skew", "br_kurt", "hr_skew", "hr_kurt", "ie_ratio"],
        "variability_only": ["br_period_cv", "hr_period_cv", "hrv_rmssd_s", "cardioresp_coupling"],
    }
    S["B_feature_ablation"] = {k: pair(dog, p60, v, True) for k, v in groups.items()}
    # --- C. negative controls (same species, different acquisition) -------------
    S["NC1_human60_vs_human120"] = {"inv16": pair(p60_rest, vs, inv, True),
                                    "inv16+grid5": pair(p60_rest, vs, rich, True)}
    d1, d2 = dog[dog.condition == "scen1"], dog[dog.condition == "scen2"]
    S["NC2_dog_scen1_vs_scen2"] = {"inv16": pair(d1, d2, inv, True),
                                   "inv16+grid5": pair(d1, d2, rich, True)}
    S["NC3_p60_rest_vs_post"] = {"inv16": pair(p60_rest, p60[~p60.condition.str.endswith("/Rest")], inv, True)}
    # --- D. transfer: train dog vs 60 GHz human; test on unseen 120 GHz humans --
    D = pd.concat([dog.assign(y=0), p60.assign(y=1)], ignore_index=True)
    S["D_transfer_to_120GHz"] = {}
    for m in ["RF", "LogReg", "SVM-RBF"]:
        model = mk(m).fit(D[inv].values, D.y.values)
        pr = model.predict(vs[inv].values)
        S["D_transfer_to_120GHz"][m] = {"frac_predicted_human": float(pr.mean()), "n": int(len(vs)),
                                        "per_scenario": {sc: float(pr[(vs.condition == sc).values].mean())
                                                         for sc in vs.condition.unique()}}
    # reverse transfer: train dog vs 120 GHz human, test 60 GHz humans
    D2 = pd.concat([dog.assign(y=0), vs.assign(y=1)], ignore_index=True)
    S["D_transfer_to_60GHz"] = {}
    for m in ["RF", "LogReg"]:
        pr = mk(m).fit(D2[inv].values, D2.y.values).predict(p60[inv].values)
        S["D_transfer_to_60GHz"][m] = {"frac_predicted_human": float(pr.mean()), "n": int(len(p60))}
    # three-domain classifier (dog / 60 GHz / 120 GHz), grouped
    D3 = pd.concat([dog.assign(y=0), p60_rest.assign(y=1), vs.assign(y=2)], ignore_index=True)
    X3, y3, g3 = D3[inv].values, D3.y.values, (D3.dataset + ":" + D3.subject).values
    pred = np.zeros(len(y3), int)
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=C.SEED).split(X3, y3, g3):
        pred[b] = mk("RF").fit(X3[a], y3[a]).predict(X3[b])
    cm = pd.crosstab(pd.Series(y3, name="true"), pd.Series(pred, name="pred")).values.tolist()
    S["E_three_domain_grouped"] = {"balanced_acc": float(balanced_accuracy_score(y3, pred)),
                                   "confusion": cm, "labels": ["dog_uwb7g", "human_fmcw60g", "human_fmcw120g"]}
    # --- F. permutation test (grouped, LogReg) -----------------------------------
    Dp = pd.concat([dog.assign(y=0), p60.assign(y=1)], ignore_index=True)
    Xp, yp, gp = Dp[inv].values, Dp.y.values, (Dp.dataset + ":" + Dp.subject).values
    obs = cv_eval(Xp, yp, gp, "LogReg", True)["balanced_acc"]
    p, nm, n95 = permutation_p(Xp, yp, gp, obs)
    S["F_permutation_dog_vs_p60"] = {"observed_bal_acc": obs, "p": p, "null_mean": nm, "null_95": n95}
    # --- G. effect sizes ---------------------------------------------------------
    S["G_effect_size_dog_minus_human60"] = effect_sizes(dog, p60_rest, inv)
    S["G_effect_size_human60_minus_human120"] = effect_sizes(p60_rest, vs, inv)
    # RF importance (dog vs 60 GHz, all)
    rf = mk("RF").fit(Dp[inv].values, yp)
    imp = rf[-1].feature_importances_
    S["H_rf_importance"] = dict(sorted(zip(inv, map(float, imp)), key=lambda t: -t[1]))
    with open(RES / "cross_species_summary.json", "w") as f:
        json.dump(S, f, indent=2)

    def brief(r):
        return {k: round(r[k], 3) for k in ("acc_mean", "balanced_acc", "auroc", "majority")}
    for k, v in S.items():
        if k.startswith(("A", "B", "NC")):
            for kk, vv in v.items():
                print(k, kk, brief(vv))
        elif k.startswith(("D", "E", "F")):
            print(k, json.dumps(v)[:400])
    print("importance", list(S["H_rf_importance"].items())[:5])
