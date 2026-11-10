"""Vital-sign estimators (rate estimation from radar phase / magnitude)."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, welch

from .dsp import bandpass, bandstop, weighted_median


# ----------------------------------------------------------------------------
# Multi-bin estimator with respiratory-harmonic suppression
# ----------------------------------------------------------------------------
def _br_peak(x, fs, band, nperseg):
    f, P = welch(x - x.mean(), fs, nperseg=min(nperseg, len(x)))
    m = (f >= band[0]) & (f <= band[1])
    if m.sum() < 2:
        return None, 0.0
    i = int(np.argmax(P[m]))
    return float(f[m][i]), float(P[m][i] / (np.median(P[m]) + 1e-30))


def _hr_peak(x, fs, f_br, band, nperseg, notch=True, n_harm=12, hw=0.025, prefilter=False):
    y = x - x.mean()
    if notch and f_br is not None:
        for k in range(2, n_harm + 1):
            fh = k * f_br
            if band[0] - 0.05 <= fh <= band[1] + 0.05:
                y = bandstop(y, fs, fh, hw)
    if prefilter:
        y = bandpass(y, fs, *band)
    f, P = welch(y, fs, nperseg=min(nperseg, len(y)))
    m = (f >= band[0]) & (f <= band[1])
    if m.sum() < 3:
        return None, 0.0
    fb, pb = f[m], P[m]
    pk, _ = find_peaks(pb)
    if len(pk) == 0:
        return None, 0.0
    i = pk[np.argmax(pb[pk])]
    return float(fb[i]), float(pb[i] / (np.median(pb) + 1e-30))


def multibin_estimate(M, fs, br_band, hr_band, *, br_nperseg, hr_nperseg, notch=True,
                      ratio=(4.0, 12.0), fusion="snr_wmedian", hw=0.025, prefilter=False,
                      hr_bpm_range=None, br_bpm_range=None, col_weights=None):
    """Estimate (BR, HR) in bpm from a matrix of channels (samples x channels).

    For each channel: breathing peak -> notch its harmonics inside the cardiac
    band -> cardiac spectral peak.  Channels that violate the HR/BR ratio (or
    optional physiological ranges) are discarded; remaining channels are fused.
    """
    hrs, ws, brs = [], [], []
    for c in range(M.shape[1]):
        x = M[:, c]
        if not np.any(x - x.mean()):
            continue
        fb, _ = _br_peak(x, fs, br_band, br_nperseg)
        if fb is None:
            continue
        if br_bpm_range and not (br_bpm_range[0] <= fb * 60 <= br_bpm_range[1]):
            continue
        fh, snr = _hr_peak(x, fs, fb, hr_band, hr_nperseg, notch=notch, hw=hw, prefilter=prefilter)
        if fh is None:
            continue
        if hr_bpm_range and not (hr_bpm_range[0] <= fh * 60 <= hr_bpm_range[1]):
            continue
        if ratio and not (ratio[0] <= fh / fb <= ratio[1]):
            continue
        w = snr * (col_weights[c] if col_weights is not None else 1.0)
        hrs.append(fh); ws.append(w); brs.append(fb)
    if not hrs:
        return None
    hrs, ws, brs = map(np.asarray, (hrs, ws, brs))
    if fusion == "snr_wmedian":
        hr = weighted_median(hrs, ws)
    elif fusion == "median":
        hr = float(np.median(hrs))
    elif fusion == "max_snr":
        hr = float(hrs[np.argmax(ws)])
    else:
        raise ValueError(fusion)
    return {"br": float(np.median(brs) * 60), "hr": float(hr * 60), "n_channels": int(len(hrs))}


def single_channel_estimate(x, fs, br_band, hr_band, nperseg):
    """Plain spectral-peak estimator on one channel (baseline)."""
    x = x - x.mean()
    out = {}
    for name, band in (("br", br_band), ("hr", hr_band)):
        y = bandpass(x, fs, *band)
        f, P = welch(y, fs, nperseg=min(nperseg, len(y)))
        m = (f >= band[0]) & (f <= band[1])
        out[name] = float(f[m][np.argmax(P[m])] * 60)
    return out


def ecg_hr(ecg, fs):
    """Median R-R heart rate from a single-lead ECG (reference for 60 GHz data)."""
    e = bandpass(ecg - ecg.mean(), fs, 0.5, 40)
    pk, _ = find_peaks(e, distance=int(fs * 0.4), height=np.percentile(e, 75))
    if len(pk) < 2:
        return None
    return float(60.0 / np.median(np.diff(pk) / fs))
