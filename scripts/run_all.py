"""Run the whole reproduction pipeline in order.

    python scripts/run_all.py            # everything (needs the raw datasets)
    python scripts/run_all.py --from 03  # resume from a stage
    python scripts/run_all.py --figures  # only the figure + number table stage

Each stage is a separate process so a failure stops the run with the failing
stage name rather than a traceback from an unrelated import.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = [
    ("00", "00_cache_dogs.py", "CSV -> float32 cache (one-off, ~3.5 GB)"),
    ("01", "01_extract_features.py", "window-level feature tables"),
    ("02", "02_vital_signs.py", "vital-sign results and ablation"),
    ("03", "03_identification.py", "closed-set and open-set protocols"),
    ("04", "04_cross_species.py", "species contrast and negative controls"),
    ("05", "05_rate_matched_age.py", "rate-matched control and age regression"),
    ("06", "make_figures.py", "figures/*.pdf and *.png"),
    ("07", "07_number_table.py", "docs/NUMBERS.md"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="start", default="00",
                    help="first stage to run (00, 01, ... 07; default 00)")
    ap.add_argument("--figures", action="store_true",
                    help="only stages 06 and 07")
    args = ap.parse_args()

    ids = [s[0] for s in STAGES]
    if args.start not in ids:
        ap.error(f"unknown stage {args.start!r}; choose one of {ids}")
    selected = STAGES[-2:] if args.figures else STAGES[ids.index(args.start):]

    for stage_id, script, what in selected:
        path = HERE / script
        if not path.exists():
            print(f"[{stage_id}] missing {script}; skip", flush=True)
            continue
        print(f"[{stage_id}] {script}  ({what})", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(path)], cwd=str(HERE.parent))
        if proc.returncode != 0:
            print(f"[{stage_id}] FAILED with exit code {proc.returncode}", file=sys.stderr)
            return proc.returncode
        print(f"[{stage_id}] ok in {time.time() - t0:.0f} s", flush=True)
    print("all stages finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
