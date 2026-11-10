"""Agreement and evaluation statistics."""
from __future__ import annotations

import numpy as np


def agreement(est, ref) -> dict:
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    d = est - ref
    ae = np.abs(d)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    r = float(np.corrcoef(est, ref)[0, 1]) if len(d) > 2 and est.std() > 0 and ref.std() > 0 else float("nan")
    return {
        "n": int(len(d)),
        "mae": float(ae.mean()),
        "median_ae": float(np.median(ae)),
        "max_ae": float(ae.max()),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "bias": float(d.mean()),
        "loa_low": float(d.mean() - 1.96 * sd),
        "loa_high": float(d.mean() + 1.96 * sd),
        "pearson_r": r,
        "within_5": int((ae <= 5).sum()),
        "within_10": int((ae <= 10).sum()),
    }


def bootstrap_ci(values_by_cluster, stat=np.mean, n_boot=2000, seed=0, alpha=0.05):
    """Cluster (subject) bootstrap CI.  ``values_by_cluster`` is a list of arrays."""
    rng = np.random.default_rng(seed)
    k = len(values_by_cluster)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, k, k)
        stats.append(stat(np.concatenate([values_by_cluster[i] for i in idx])))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def mean_sd(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=0))
