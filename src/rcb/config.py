"""Paths and global constants.

All dataset locations are resolved from the environment variable
``RCB_DATA_ROOT`` (a directory that contains the three public datasets in
their original folder layout, see ``docs/DATA.md``).  A binary cache of the
large CSV files is written to ``RCB_CACHE`` (default: ``<RCB_DATA_ROOT>/_cache``).
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("RCB_DATA_ROOT",
                                Path.home() / "Desktop" / "Biometric_Project"))
CACHE_DIR = Path(os.environ.get("RCB_CACHE", DATA_ROOT / "_cache"))

# --- Dogs Vital Sign dataset (Ahmed et al., Sci. Data 2024) ---------------
DOG_SCEN1_ROOT = DATA_ROOT / "dog_dataset"          # No1..No10, anaesthetised, 180 s
DOG_SCEN2_ROOT = DATA_ROOT / "Scenario2_AwakeDogs"  # No11..No30, awake, 1800 s
DOG_FS = 50.0
DOG_SCEN1 = [f"No{i}" for i in range(1, 11)]
DOG_SCEN2 = [f"No{i}" for i in range(11, 31)]
DOG_ALL = DOG_SCEN1 + DOG_SCEN2

# --- VITALSENSE 120 GHz dataset (Sci. Data 2026) ---------------------------
VS_FS = 1.0 / 0.003            # 333.33 Hz displacement signal ("VitalSig")
VS_SUBJECTS = [f"VS{i:02d}" for i in range(1, 25)]
VS_SCENARIOS = ["Resting", "Apnea"]

# --- Age-balanced 60 GHz FMCW dataset (Zenodo 18599983) --------------------
P_FS = 10.0                    # frame rate (PERIODICITY = 100 ms)
P_ECG_FS = 256.0
P_SUBJECTS = [f"P{i:03d}" for i in range(1, 111)]
P_REST = [("Lying", "Rest"), ("Sitting", "Rest")]
P_POST = [("Lying", "Post-exercise"), ("Sitting", "Post-exercise")]
P_INFO_XLSX = DATA_ROOT / "ParticipantsInfo.xlsx"

# --- Shared processing constants --------------------------------------------
TARGET_FS = 20.0               # common grid for sampling-rate-invariant features
SEED = 42
