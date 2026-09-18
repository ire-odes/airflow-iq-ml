"""
run_house_drift_only.py
Tests the drift half of the pipeline on its own, bypassing
FilterHealthMonitor's classifier-gated warmup.

Why bypass: the global RF classifier was trained only on the lab's
FPR5/FPR12 recordings and doesn't transfer to a new house's mic/room/duct
acoustics -- it called every MERV10 clip "clogged", which blocks the
poisoning guard from ever letting the baseline freeze. That's the
classifier failing to generalize, not evidence the audio itself carries
no signal. This script builds the Mahalanobis baseline directly from the
pre-swap clips (all assumed clean -- it's the currently-installed
filter) and checks whether drift alone separates them from the
known-dirty post-swap clips.

Usage:
  python run_house_drift_only.py AbrahamHouse MERV10 FPR12
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

import numpy as np

from extract_features_v2 import extract_features
from drift_detector import (BaselineProfile, DriftDetector,
                            calibrate_threshold, DEFAULT_DRIFT_FEATURES)

DATA_DIR = Path("data")
WARMUP_FRACTION = 0.7  # rest of the clean set held out to check the FP rate


def vec(feats):
    return np.array([feats[k] for k in DEFAULT_DRIFT_FEATURES])


def main():
    if len(sys.argv) != 4:
        print("Usage: python run_house_drift_only.py <house_dir> <clean_subfolder> <dirty_subfolder>")
        sys.exit(1)
    house, clean_name, dirty_name = sys.argv[1:4]

    clean_files = sorted((DATA_DIR / house / clean_name).glob("*.wav"))
    dirty_files = sorted((DATA_DIR / house / dirty_name).glob("*.wav"))

    clean_feats = [extract_features(f) for f in clean_files]
    clean_feats = [f for f in clean_feats if f is not None]
    dirty_feats = [extract_features(f) for f in dirty_files]
    dirty_feats = [f for f in dirty_feats if f is not None]

    n_warm = int(len(clean_feats) * WARMUP_FRACTION)
    print(f"{clean_name}: {len(clean_feats)} clips -> {n_warm} warmup / "
          f"{len(clean_feats) - n_warm} held-out clean")
    print(f"{dirty_name}: {len(dirty_feats)} clips (post-swap test)")

    baseline = BaselineProfile(len(DEFAULT_DRIFT_FEATURES), min_samples=n_warm)
    for f in clean_feats[:n_warm]:
        baseline.update(vec(f))
    baseline.freeze()

    holdout_clean_d = [baseline.distance(vec(f)) for f in clean_feats[n_warm:]]
    dirty_d = [baseline.distance(vec(f)) for f in dirty_feats]

    thr, fp, det = calibrate_threshold(holdout_clean_d, dirty_d, clean_percentile=95.0)
    print(f"\nCalibrated threshold (95th pct of held-out clean distances): {thr:.2f}")
    print(f"  Held-out clean distances: min {min(holdout_clean_d):.2f} / "
          f"mean {np.mean(holdout_clean_d):.2f} / max {max(holdout_clean_d):.2f}")
    print(f"  Dirty (post-swap) distances: min {min(dirty_d):.2f} / "
          f"mean {np.mean(dirty_d):.2f} / max {max(dirty_d):.2f}")
    print(f"  Expected false-positive rate on clean: {fp:.1%}")
    print(f"  Dirty-clip detection rate at this threshold: {det:.1%}")

    # Sequential run through the EWMA+patience detector, in recording order,
    # to see how quickly a real device would have alerted after the swap.
    detector = DriftDetector(baseline, thr, alpha=0.15, patience=3)
    print(f"\n--- Sequential replay: held-out clean, then post-swap clips ---")
    alert_idx = None
    for i, f in enumerate(clean_feats[n_warm:]):
        r = detector.update(vec(f))
        if r["drift_dirty"] and alert_idx is None:
            alert_idx = i  # would be a false alarm -- still in the clean phase
    n_clean_holdout = len(clean_feats) - n_warm
    if alert_idx is not None:
        print(f"FALSE ALARM during held-out clean phase at clip #{alert_idx}")
    else:
        print(f"No false alarm during {n_clean_holdout} held-out clean clips.")

    first_dirty_alert = None
    for i, f in enumerate(dirty_feats):
        r = detector.update(vec(f))
        if r["drift_dirty"] and first_dirty_alert is None:
            first_dirty_alert = i
    if first_dirty_alert is not None:
        print(f"First sustained 'dirty' alert after swap: clip #{first_dirty_alert} "
              f"of {len(dirty_feats)} ({dirty_files[first_dirty_alert].name})")
    else:
        print("Never sustained a 'dirty' alert across the post-swap set.")


if __name__ == "__main__":
    main()
