"""
test_drift.py
Scores new clip(s) against a saved per-house baseline (see build_baseline.py).

Loads baseline_<house>.pkl, feeds it clip(s) in filename order, and
re-saves the detector's EWMA/exceedance state afterward -- so testing
across separate invocations behaves like a continuously-running device
(state carries forward) rather than resetting on every call.

Usage:
  python test_drift.py <house_dir> <wav_file_or_folder>
"""
# -- run from the project root, wherever this file lives ------------------
# Moved out of the project root in the 2026-09-18 reorganisation. Its paths
# are written relative to the root and the core modules (extract_features_v2,
# drift_detector, spectrum_viz) live there, so put the root on sys.path and
# make it the working directory before anything else runs.
import os as _os, sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
# --------------------------------------------------------------------------

import sys
from pathlib import Path

import joblib
import numpy as np

from extract_features_v2 import extract_features
from drift_detector import DEFAULT_DRIFT_FEATURES


def vec(feats):
    return np.array([feats[k] for k in DEFAULT_DRIFT_FEATURES])


def main():
    if len(sys.argv) != 3:
        print("Usage: python test_drift.py <house_dir> <wav_file_or_folder>")
        sys.exit(1)
    house, target = sys.argv[1], Path(sys.argv[2])

    baseline_path = f"models/baseline_{house}.pkl"
    if not Path(baseline_path).exists():
        print(f"No saved baseline for '{house}' -- run build_baseline.py first "
              f"(expected {baseline_path}).")
        sys.exit(1)
    detector = joblib.load(baseline_path)

    files = sorted(target.glob("*.wav")) if target.is_dir() else [target]
    if not files:
        print(f"No .wav files found at {target}")
        sys.exit(1)

    n_alerted = 0
    first_alert = None
    for f in files:
        feats = extract_features(f)
        if feats is None:
            print(f"{f.name}: too short/silent, skipped")
            continue
        r = detector.update(vec(feats))
        status = "DIRTY (sustained)" if r["drift_dirty"] else "clean"
        print(f"{f.name}: distance={r['distance']:.2f}  ewma={r['ewma']:.2f}  "
              f"threshold={detector.threshold:.2f}  -> {status}")
        if r["drift_dirty"]:
            n_alerted += 1
            if first_alert is None:
                first_alert = f.name

    joblib.dump(detector, baseline_path)

    print(f"\n{n_alerted}/{len(files)} readings in sustained-dirty state.")
    if first_alert:
        print(f"First sustained alert: {first_alert}")


if __name__ == "__main__":
    main()
