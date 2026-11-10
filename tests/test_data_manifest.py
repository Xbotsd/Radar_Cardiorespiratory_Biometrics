"""Manifest assertions: fail loudly on a truncated download or a moved file.

Skipped when ``RCB_DATA_ROOT`` is not present, so the suite still runs on a
machine that only has the committed feature tables.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rcb import config as C
from rcb import io

HAS_DATA = C.DATA_ROOT.exists() and (C.DOG_SCEN1_ROOT / "No1").exists()
pytestmark = pytest.mark.skipif(not HAS_DATA, reason="raw datasets not available")


def test_dog_recording_lengths():
    a = io.load_dog_radar("No1")
    b = io.load_dog_radar("No11")
    assert a.shape == (9000, 467)
    assert b.shape == (90000, 312)


def test_dog_reference_is_one_hz():
    ref = io.load_dog_reference("No3")
    assert ref["hr"].shape == (180,)
    assert 60 < ref["hr"].mean() < 200


def test_vitalsense_recordings_exist():
    n = sum(io.load_vitalsense(s, sc) is not None
            for s in C.VS_SUBJECTS for sc in C.VS_SCENARIOS)
    assert n == 48


def test_p60_frame_counts():
    lens = []
    for s in C.P_SUBJECTS[:10]:
        for posture, state in C.P_REST:
            cube, ranges = io.load_p_radar(s, posture, state)
            assert cube is not None and cube.ndim == 3 and cube.shape[1:] == (8, 64)
            lens.append(cube.shape[0])
    assert min(lens) >= 400 and max(lens) <= 600


def test_participants_table():
    info = io.load_participants()
    assert len(info) == 110
    assert {"PARTICIPANT ID", "AGE"} <= set(info.columns)
    assert info["AGE"].between(18, 90).all()
