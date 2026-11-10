"""Convert the 30 dog radar CSV files to float32 .npy (one-off, ~3.5 GB)."""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rcb import config as C  # noqa: E402
from rcb.io import load_dog_radar  # noqa: E402


def _one(s):
    t = time.time()
    a = load_dog_radar(s)
    return s, a.shape, time.time() - t


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=6) as ex:
        for s, shape, dt in ex.map(_one, C.DOG_ALL):
            print(f"{s}: {shape} ({dt:.1f} s)", flush=True)
