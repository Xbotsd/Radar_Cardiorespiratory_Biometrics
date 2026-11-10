"""Signal-processing primitives shared by all experiments."""
from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly, welch


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass (forward-backward)."""
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x, axis=0)


def bandstop(x: np.ndarray, fs: float, f0: float, hw: float, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    lo = max(f0 - hw, 1e-3) / nyq
    hi = min(f0 + hw, 0.99 * nyq) / nyq
    if lo >= hi:
        return x
    b, a = butter(order, [lo, hi], btype="bandstop")
    return filtfilt(b, a, x, axis=0)


def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resampling between rational rates (333.33 Hz handled as 1000/3)."""
    if abs(fs_in - fs_out) < 1e-6:
        return x
    # represent rates as integers in units of 1/3 Hz to cover 1000/3 Hz
    a = int(round(fs_in * 3))
    b = int(round(fs_out * 3))
    g = gcd(a, b)
    return resample_poly(x, b // g, a // g, axis=0)


def psd(x: np.ndarray, fs: float, nperseg: int | None = None, nfft: int | None = None):
    n = x.shape[0]
    nperseg = min(nperseg or n, n)
    return welch(x, fs, nperseg=nperseg, nfft=max(nfft or 0, nperseg), axis=0)


def peak_in_band(f: np.ndarray, P: np.ndarray, lo: float, hi: float):
    """Return (peak frequency, peak/median prominence) inside [lo, hi]."""
    m = (f >= lo) & (f <= hi)
    if m.sum() < 3:
        return None, 0.0
    fb, pb = f[m], P[m]
    i = int(np.argmax(pb))
    return float(fb[i]), float(pb[i] / (np.median(pb) + 1e-30))


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(v[min(np.searchsorted(c, c[-1] / 2.0), len(v) - 1)])


def sliding_windows(n: int, win: int, hop: int, max_windows: int | None = None):
    """Start indices of full windows; mirrors ``range(0, n - win, hop)`` of the
    original analysis code (the final partial window is dropped)."""
    starts = list(range(0, n - win, hop))
    if max_windows is not None:
        starts = starts[:max_windows]
    return starts
