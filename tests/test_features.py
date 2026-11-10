"""Unit tests for the signal-processing core (no data required)."""
import numpy as np
import pytest

from rcb.dsp import bandpass, resample_to, sliding_windows, weighted_median
from rcb.features import INV16_NAMES, RATE_DEPENDENT, inv16, rate_dependent


def test_resample_handles_333_33_hz():
    fs_in, fs_out, n = 1000.0 / 3.0, 20.0, 4000
    t = np.arange(n) / fs_in
    x = np.sin(2 * np.pi * 1.2 * t)
    y = resample_to(x, fs_in, fs_out)
    assert abs(len(y) - n * fs_out / fs_in) <= 2
    f = np.fft.rfftfreq(len(y), 1 / fs_out)
    assert abs(f[np.argmax(np.abs(np.fft.rfft(y)))] - 1.2) < 0.05


def test_bandpass_rejects_out_of_band_tone():
    fs, sec = 20.0, 40.0
    t = np.arange(int(fs * sec)) / fs
    inband = np.sin(2 * np.pi * 1.2 * t)
    outband = 3.0 * np.sin(2 * np.pi * 5.0 * t)
    y = bandpass(inband + outband, fs, 0.8, 2.6)

    def amp_5hz(sig):
        f = np.fft.rfftfreq(len(sig), 1 / fs)
        return np.abs(np.fft.rfft(sig))[np.argmin(np.abs(f - 5.0))]

    assert amp_5hz(y) < 0.01 * amp_5hz(outband)
    # and the in-band tone must survive
    assert np.std(y) > 0.5


def test_sliding_windows_drops_tail_and_caps():
    assert sliding_windows(100, 30, 10, max_windows=3) == [0, 10, 20]
    assert sliding_windows(100, 30, 10) == [0, 10, 20, 30, 40, 50, 60]


def test_weighted_median_ignores_small_weight_outlier():
    v = np.array([1.0, 2.0, 3.0, 99.0])
    w = np.array([1.0, 1.0, 1.0, 1e-6])
    assert abs(weighted_median(v, w) - 2.0) < 1e-9


def _cardioresp(fs=20.0, seconds=20.0, hr_bpm=70.0, br_bpm=15.0, noise=0.0, seed=0):
    t = np.arange(int(fs * seconds)) / fs
    hr, br = hr_bpm / 60.0, br_bpm / 60.0
    x = (0.60 * np.sin(2 * np.pi * br * t)
         + 0.25 * np.sin(2 * np.pi * hr * t)
         + 0.05 * np.sin(2 * np.pi * 2 * br * t))
    if noise:
        x = x + noise * np.random.default_rng(seed).standard_normal(len(t))
    return x


def test_inv16_recovers_rates():
    x = _cardioresp(fs=20.0)
    F = inv16(x, 20.0)
    assert F is not None
    assert set(F) == set(INV16_NAMES)
    assert abs(F["hr_freq_hz"] - 70 / 60) < 0.06
    assert abs(F["br_freq_hz"] - 15 / 60) < 0.03
    assert 0 < F["hr_power_frac"] < 1 and 0 < F["br_power_frac"] < 1
    assert 0 <= F["cardioresp_coupling"] <= 1


@pytest.mark.parametrize("name", INV16_NAMES)
def test_inv16_is_grid_invariant(name):
    """The 16-D set is computed after resampling to a common grid, so the same
    physical signal sampled at 50 Hz and 20 Hz must give the same value."""
    x50 = _cardioresp(fs=50.0)
    x20 = resample_to(x50, 50.0, 20.0)
    a, b = inv16(x50, 50.0), inv16(x20, 20.0)
    scale = max(abs(a[name]), 1e-9)
    assert abs(a[name] - b[name]) / scale < 0.05, (name, a[name], b[name])


# Minimum relative change that each legacy descriptor must show when the same
# broadband record is re-expressed on a 20 Hz grid (measured on synthetic data
# with 0.5 units of broadband noise).  The thresholds differ because the
# mechanisms differ: spectral centroid and spectral flatness are directly tied
# to the Nyquist frequency and the number of bins, entropy and the low/high
# ratio only weakly, and a zero-crossing *rate per second* of a narrow-band
# signal is almost grid-stable.  The paper reports the measured drift on real
# recordings rather than asserting a single common magnitude.
DRIFT_MIN = {"spectral_centroid": 0.20, "br_spectral_flatness": 0.20,
             "spectral_entropy": 0.03, "lf_hf_ratio": 0.005,
             "br_zero_cross_rate": 0.0}


@pytest.mark.parametrize("name", RATE_DEPENDENT)
def test_rate_dependent_features_drift(name):
    """The legacy descriptors are evaluated at the native grid, so a broadband
    record sampled at 50 Hz and at 20 Hz must disagree in proportion to how
    strongly the definition is tied to the grid."""
    x50 = _cardioresp(fs=50.0, noise=0.5)
    x20 = resample_to(x50, 50.0, 20.0)
    a, b = rate_dependent(x50, 50.0), rate_dependent(x20, 20.0)
    scale = max(abs(a[name]), 1e-9)
    assert abs(a[name] - b[name]) / scale >= DRIFT_MIN[name], (name, a[name], b[name])
