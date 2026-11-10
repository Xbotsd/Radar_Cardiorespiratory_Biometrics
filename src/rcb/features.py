"""Feature sets.

* ``id19``   - 19 descriptors used for individual identification (dogs, 50 Hz,
               all range bins).  Identical to the original analysis code.
* ``inv16``  - 16 sampling-rate-invariant descriptors (physical units or
               normalised ratios) computed on a single channel after
               resampling to 20 Hz.  Used for cross-species and age analyses.
* ``rich19`` - 19 descriptors that additionally contain spectral centroid,
               spectral entropy, a log low/high power ratio, zero-crossing rate
               and spectral flatness.  These depend on the sampling grid and are
               kept only as a negative control.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, hilbert, welch
from scipy.stats import kurtosis as _kurt, skew as _skew

from . import config as C
from .dsp import bandpass, resample_to


def _skewness(x):  # pandas-compatible (bias-corrected) skewness / excess kurtosis
    return float(_skew(x, bias=False))


def _kurtosis(x):
    return float(_kurt(x, fisher=True, bias=False))


def _coupling(br_sig, hr_sig):
    if len(br_sig) != len(hr_sig) or len(br_sig) <= 10:
        return 0.0
    ph = np.angle(hilbert(br_sig))
    env = np.abs(hilbert(hr_sig))
    if np.std(env) <= 1e-10:
        return 0.0
    return float(abs(np.corrcoef(np.cos(ph), env)[0, 1]))


# ----------------------------------------------------------------------------
# 19-D identification set (multi-channel, 50 Hz)
# ----------------------------------------------------------------------------
ID_BR_BAND = (0.15, 0.45)
ID_HR_BAND = (1.6, 2.6)
ID19_NAMES = ["br_mean", "br_std", "br_iqr", "br_amp", "hr_mean", "hr_std", "hr_iqr",
              "hr_snr_mean", "hr_br_ratio", "br_interval_mean", "br_interval_std",
              "br_skew", "br_kurt", "hr_interval_mean", "hr_interval_std", "hrv_rmssd",
              "hr_skew", "hr_kurt", "cardioresp_coupling"]
ID_GROUPS = {
    "respiration": [n for n in ID19_NAMES if n.startswith("br_")],
    "cardiac": [n for n in ID19_NAMES if n.startswith("hr_") or n.startswith("hrv_")],
}
ID_GROUPS["resp+cardiac"] = ID_GROUPS["respiration"] + ID_GROUPS["cardiac"]
ID_GROUPS["all19"] = list(ID19_NAMES)


def id19(window: np.ndarray, fs: float = C.DOG_FS) -> dict:
    w = window - window.mean(axis=0, keepdims=True)
    nper = min(512, w.shape[0])
    br_all = bandpass(w, fs, *ID_BR_BAND)
    hr_all = bandpass(w, fs, *ID_HR_BAND)
    f, Pb = welch(br_all, fs, nperseg=nper, axis=0)
    _, Ph = welch(hr_all, fs, nperseg=nper, axis=0)
    mb = (f >= ID_BR_BAND[0]) & (f <= ID_BR_BAND[1])
    mh = (f >= ID_HR_BAND[0]) & (f <= ID_HR_BAND[1])
    F = {}
    br_f = f[mb][np.argmax(Pb[mb], axis=0)]
    br_a = np.sqrt(np.mean(br_all ** 2, axis=0))
    F["br_mean"] = float(np.median(br_f) * 60)
    F["br_std"] = float(np.std(br_f) * 60)
    F["br_iqr"] = float((np.percentile(br_f, 75) - np.percentile(br_f, 25)) * 60)
    F["br_amp"] = float(np.median(br_a))
    ph = Ph[mh]
    snr = ph.max(axis=0) / (np.median(ph, axis=0) + 1e-20)
    hr_f_all = f[mh][np.argmax(ph, axis=0)]
    keep = snr > 1.5
    if keep.any():
        hf = hr_f_all[keep]
        F["hr_mean"] = float(np.median(hf) * 60)
        F["hr_std"] = float(np.std(hf) * 60)
        F["hr_iqr"] = float((np.percentile(hf, 75) - np.percentile(hf, 25)) * 60)
        F["hr_snr_mean"] = float(np.median(snr[keep]))
    else:
        F["hr_mean"] = F["hr_std"] = F["hr_iqr"] = F["hr_snr_mean"] = 0.0
    F["hr_br_ratio"] = F["hr_mean"] / F["br_mean"] if (F["br_mean"] > 0 and F["hr_mean"] > 0) else 0.0
    best = int(np.argmax(snr))
    bs = w[:, best]
    br_sig = bandpass(bs, fs, *ID_BR_BAND)
    pk, _ = find_peaks(br_sig, distance=int(fs * 1.5))
    if len(pk) >= 2:
        per = np.diff(pk) / fs
        F["br_interval_mean"] = float(per.mean()); F["br_interval_std"] = float(per.std())
        F["br_skew"] = _skewness(br_sig); F["br_kurt"] = _kurtosis(br_sig)
    else:
        F["br_interval_mean"] = F["br_interval_std"] = F["br_skew"] = F["br_kurt"] = 0.0
    hr_sig = bandpass(bs, fs, *ID_HR_BAND)
    pk, _ = find_peaks(hr_sig, distance=int(fs * 0.3))
    if len(pk) >= 2:
        per = np.diff(pk) / fs
        F["hr_interval_mean"] = float(per.mean()); F["hr_interval_std"] = float(per.std())
        F["hrv_rmssd"] = float(np.sqrt(np.mean(np.diff(per) ** 2))) if len(per) >= 2 else 0.0
        F["hr_skew"] = _skewness(hr_sig); F["hr_kurt"] = _kurtosis(hr_sig)
    else:
        F["hr_interval_mean"] = F["hr_interval_std"] = F["hrv_rmssd"] = 0.0
        F["hr_skew"] = F["hr_kurt"] = 0.0
    F["cardioresp_coupling"] = _coupling(br_sig, hr_sig)
    return F


# ----------------------------------------------------------------------------
# 16-D sampling-rate-invariant set (single channel, resampled to 20 Hz)
# ----------------------------------------------------------------------------
INV16_NAMES = ["br_freq_hz", "br_period_s", "br_period_cv", "ie_ratio", "br_skew", "br_kurt",
               "hr_freq_hz", "hr_period_cv", "hrv_rmssd_s", "hr_skew", "hr_kurt",
               "hr_br_ratio", "br_power_frac", "hr_power_frac", "br_hr_power_ratio",
               "cardioresp_coupling"]
RATE_DEPENDENT = ["spectral_centroid", "spectral_entropy", "lf_hf_ratio",
                  "br_zero_cross_rate", "br_spectral_flatness"]
"""Legacy descriptors evaluated at the *native* sampling grid.

They are kept as a negative control, not as features.  Their sensitivity to the
grid is not uniform: the centroid and the spectral flatness depend on the
Nyquist frequency and on the number of bins, the entropy and the low/high power
ratio depend on the bin normaliser, and a zero-crossing rate expressed per
second is almost grid-stable.  ``results/rate_matched_age_summary.json`` holds
the measured drift for each of them on real recordings (50 Hz -> 10 Hz)."""


def _ie_ratio(br_peaks, br_troughs, fs):
    r = []
    for p in br_peaks:
        prev = br_troughs[br_troughs < p]
        nxt = br_troughs[br_troughs > p]
        if len(prev) and len(nxt):
            ti = (p - prev[-1]) / fs
            te = (nxt[0] - p) / fs
            if te > 0.1:
                r.append(ti / te)
    return r


def inv16(sig: np.ndarray, fs: float, target_fs: float = C.TARGET_FS) -> dict | None:
    x = sig - np.mean(sig)
    x = resample_to(x, fs, target_fs)
    fs = target_fs
    f, P = welch(x, fs, nperseg=min(512, len(x)))
    F = {}
    bm = (f >= 0.15) & (f <= 0.5)
    hm = (f >= 0.8) & (f <= 2.6)
    if bm.sum() == 0 or hm.sum() == 0:
        return None
    br_f = f[bm][np.argmax(P[bm])]
    F["br_freq_hz"] = float(br_f)
    br_sig = bandpass(x, fs, 0.15, 0.5)
    pk, _ = find_peaks(br_sig, distance=int(fs * 1.5))
    tr, _ = find_peaks(-br_sig, distance=int(fs * 1.5))
    if len(pk) >= 3 and len(tr) >= 2:
        per = np.diff(pk) / fs
        F["br_period_s"] = float(per.mean())
        F["br_period_cv"] = float(per.std() / (per.mean() + 1e-6))
        ie = _ie_ratio(pk, tr, fs)
        F["ie_ratio"] = float(np.mean(ie)) if ie else 0.0
        F["br_skew"] = _skewness(br_sig); F["br_kurt"] = _kurtosis(br_sig)
    else:
        for k in ["br_period_s", "br_period_cv", "ie_ratio", "br_skew", "br_kurt"]:
            F[k] = 0.0
    hr_f = f[hm][np.argmax(P[hm])]
    F["hr_freq_hz"] = float(hr_f)
    hr_sig = bandpass(x, fs, 0.8, 2.6)
    hp, _ = find_peaks(hr_sig, distance=int(fs * 0.3))
    if len(hp) >= 3:
        hper = np.diff(hp) / fs
        F["hr_period_cv"] = float(hper.std() / (hper.mean() + 1e-6))
        F["hrv_rmssd_s"] = float(np.sqrt(np.mean(np.diff(hper) ** 2))) if len(hper) >= 3 else 0.0
        F["hr_skew"] = _skewness(hr_sig); F["hr_kurt"] = _kurtosis(hr_sig)
    else:
        for k in ["hr_period_cv", "hrv_rmssd_s", "hr_skew", "hr_kurt"]:
            F[k] = 0.0
    F["hr_br_ratio"] = float(hr_f / (br_f + 1e-10))
    bp, hpw, tot = P[bm].sum(), P[hm].sum(), P.sum() + 1e-20
    F["br_power_frac"] = float(bp / tot)
    F["hr_power_frac"] = float(hpw / tot)
    F["br_hr_power_ratio"] = float(np.log10((bp + 1e-20) / (hpw + 1e-20)))
    F["cardioresp_coupling"] = _coupling(br_sig, hr_sig)
    return F


def rate_dependent(sig: np.ndarray, fs: float) -> dict:
    """Grid-dependent descriptors computed at the *native* rate (negative control)."""
    x = sig - np.mean(sig)
    f, P = welch(x, fs, nperseg=min(512, len(x)))
    Pn = P / (P.sum() + 1e-20)
    br_sig = bandpass(x, fs, 0.15, 0.5)
    bm = (f >= 0.15) & (f <= 0.5)
    pb = P[bm]
    lf = P[(f >= 0.1) & (f <= 0.5)].sum()
    hf = P[(f >= 0.8) & (f <= 2.6)].sum()
    return {
        "spectral_centroid": float(np.sum(f * Pn)),
        "spectral_entropy": float(-np.sum(Pn * np.log(Pn + 1e-20))),
        "lf_hf_ratio": float(np.log10((lf + 1e-20) / (hf + 1e-20))),
        "br_zero_cross_rate": float(np.sum(np.diff(np.sign(br_sig)) != 0) / len(br_sig) * fs),
        "br_spectral_flatness": float(np.exp(np.mean(np.log(pb + 1e-20))) / (np.mean(pb) + 1e-20))
        if len(pb) > 3 else 0.0,
    }
