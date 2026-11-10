"""Loaders for the three public datasets.

The dog CSV files are large (90-285 MB each).  ``load_dog_radar`` converts
them once to float32 ``.npy`` files in ``CACHE_DIR`` and memory-maps them
afterwards; the numerical content is unchanged apart from float32 rounding.
"""
from __future__ import annotations

import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

# The 60 GHz pickles were written with an older NumPy that referenced
# ``numpy.core.numeric``; alias it so that unpickling works on NumPy >= 2.
try:  # pragma: no cover - depends on NumPy version
    import numpy._core.numeric as _ncn  # type: ignore
    sys.modules.setdefault("numpy.core.numeric", _ncn)
except Exception:  # pragma: no cover
    pass


# ----------------------------------------------------------------------------
# Dogs
# ----------------------------------------------------------------------------
def dog_root(subject: str) -> Path:
    return C.DOG_SCEN1_ROOT if subject in C.DOG_SCEN1 else C.DOG_SCEN2_ROOT


def dog_scenario(subject: str) -> int:
    return 1 if subject in C.DOG_SCEN1 else 2


def _dog_csv(subject: str) -> Path:
    radar = dog_root(subject) / subject / "Radar"
    csvs = sorted(p for p in radar.iterdir() if p.suffix.lower() == ".csv")
    if not csvs:
        raise FileNotFoundError(f"no radar CSV for {subject} in {radar}")
    return csvs[0]


def load_dog_radar(subject: str, max_seconds: float | None = None) -> np.ndarray:
    """Return the UWB radar matrix (n_samples x n_range_bins) at 50 Hz."""
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = C.CACHE_DIR / f"dog_{subject}.npy"
    if not cache.exists():
        arr = pd.read_csv(_dog_csv(subject), header=None,
                          dtype=np.float64, engine="c").values.astype(np.float32)
        np.save(cache, arr)
    arr = np.load(cache, mmap_mode="r")
    if max_seconds is not None:
        arr = arr[: int(round(max_seconds * C.DOG_FS))]
    return np.asarray(arr, dtype=np.float64)


def load_dog_reference(subject: str) -> dict:
    """BIONET reference (1 Hz) for the anaesthetised dogs (No1-No10)."""
    bionet = C.DOG_SCEN1_ROOT / subject / "BIONET"
    hr = pd.read_csv(bionet / f"Reference_HR_{subject}.csv", header=None).values.ravel()
    br = pd.read_csv(bionet / f"Reference_BR_{subject}.csv", header=None).values.ravel()
    return {"hr": hr.astype(float), "br": br.astype(float)}


# ----------------------------------------------------------------------------
# VITALSENSE 120 GHz
# ----------------------------------------------------------------------------
def load_vitalsense(subject: str, scenario: str) -> dict | None:
    import scipy.io as sio
    base = C.DATA_ROOT / subject
    radar_file = base / f"{subject}_{scenario}.mat"
    ref_file = base / f"{subject}_{scenario}_Mindray.mat"
    if not radar_file.exists():
        return None
    rm = sio.loadmat(radar_file, squeeze_me=True, struct_as_record=False)
    out = {"subject": subject, "scenario": scenario,
           "vital_sig": np.asarray(rm["VitalSig"], dtype=float).ravel()}
    if ref_file.exists():
        ref = sio.loadmat(ref_file, squeeze_me=True)
        out["hr_series"] = np.asarray(ref["HR"], dtype=float).ravel()   # 1 Hz
        out["rr_series"] = np.asarray(ref["RR"], dtype=float).ravel()   # 1 Hz
    return out


# ----------------------------------------------------------------------------
# Age-balanced 60 GHz FMCW
# ----------------------------------------------------------------------------
def load_p_radar(subject: str, posture: str, state: str):
    """Return (rFFT cube [frames x antennas x bins], range axis) or (None, None)."""
    path = C.DATA_ROOT / subject / posture / state / "radar_rFFTs.zlib"
    if not path.exists():
        return None, None
    with open(path, "rb") as fh:
        obj = pickle.loads(zlib.decompress(fh.read()))
    return np.asarray(obj[0]), np.asarray(obj[1], dtype=float)


def p_phase_matrix(cube: np.ndarray) -> np.ndarray:
    """Coherent antenna sum, then unwrapped phase per range bin (frames x bins)."""
    return np.unwrap(np.angle(cube.sum(axis=1)), axis=0)


def load_p_ecg(subject: str, posture: str, state: str) -> np.ndarray | None:
    path = C.DATA_ROOT / subject / posture / state / "movesense_ecg.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)["mV"].to_numpy(dtype=float)


def load_participants() -> pd.DataFrame:
    return pd.read_excel(C.P_INFO_XLSX)
