"""
build_baseline.py
Freezes a per-house Mahalanobis baseline from clean recordings and saves
it to disk (baseline_<house>.pkl) so test_drift.py can score new clips
against it without rebuilding the baseline from raw wavs every time.

If you also have known-dirty recordings, pass them in to calibrate the
alert threshold from real labeled data: the last CALIBRATION_HOLDOUT_FRACTION
of the clean clips are held out (never used to build the baseline itself)
and their distances are compared against the dirty clips' distances at the
95th percentile -- same procedure as run_house_drift_only.py.

Without a dirty set (the normal case for a real customer install), the
threshold comes from pooled_calibration.json if one exists -- see
calibrate_pooled_threshold.py, which derives a shared threshold from
pilot sites that DO have known clean+dirty recordings, so a threshold
tuned across several real installs carries over instead of relying on
this customer's (nonexistent) dirty data. Only if no pooled calibration
file exists yet does this fall back to the theoretical chi-square guess
(sqrt of the 99th percentile of chi2(k)) -- which we've shown is a
measurably worse estimate once real calibration data exists.

Usage:
  python build_baseline.py <house_dir> <clean_subfolder> [dirty_subfolder]
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from extract_features_v2 import extract_features
from drift_detector import (BaselineProfile, DriftDetector,
                            calibrate_threshold, check_recording_continuity,
                            DEFAULT_DRIFT_FEATURES, MAX_CONTINUITY_GAP_MIN)

DATA_DIR = Path("data")
CALIBRATION_HOLDOUT_FRACTION = 0.3
POOLED_CALIBRATION_PATH = "pooled_calibration.json"


def vec(feats):
    return np.array([feats[k] for k in DEFAULT_DRIFT_FEATURES])


def load_feats(folder):
    feats = []
    for f in sorted(folder.glob("*.wav")):
        v = extract_features(f)
        if v is not None:
            feats.append(v)
    return feats


def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python build_baseline.py <house_dir> <clean_subfolder> [dirty_subfolder]")
        sys.exit(1)
    house, clean_name = sys.argv[1], sys.argv[2]
    dirty_name = sys.argv[3] if len(sys.argv) == 4 else None

    clean_dir = DATA_DIR / house / clean_name
    is_continuous, max_gap = check_recording_continuity(clean_dir)
    if not is_continuous:
        print(f"WARNING: max gap between clips in {clean_dir} is {max_gap:.0f} min "
              f"(> {MAX_CONTINUITY_GAP_MIN} min) -- this looks like multiple disjoint "
              f"recording sessions, not one continuous device warmup. BaselineProfile "
              f"assumes a single continuous run; feeding it a multi-session folder will "
              f"inflate the covariance with cross-session variability instead of genuine "
              f"device noise (this exact mistake dragged a working site's detection rate "
              f"from 58% down to 1.2% when we pooled it in earlier). Aborting.")
        sys.exit(1)

    clean_feats = load_feats(clean_dir)
    print(f"{clean_name}: {len(clean_feats)} usable clips")
    if not clean_feats:
        print("No usable clean clips found -- nothing to build a baseline from.")
        sys.exit(1)

    if dirty_name:
        dirty_feats = load_feats(DATA_DIR / house / dirty_name)
        print(f"{dirty_name}: {len(dirty_feats)} usable clips (for threshold calibration)")

        n_warm = int(len(clean_feats) * (1 - CALIBRATION_HOLDOUT_FRACTION))
        warm, holdout = clean_feats[:n_warm], clean_feats[n_warm:]
        print(f"Using {n_warm} clips to build the baseline, {len(holdout)} held out for calibration")

        baseline = BaselineProfile(len(DEFAULT_DRIFT_FEATURES), min_samples=n_warm)
        for f in warm:
            baseline.update(vec(f))
        baseline.freeze()

        holdout_d = [baseline.distance(vec(f)) for f in holdout]
        dirty_d = [baseline.distance(vec(f)) for f in dirty_feats]
        threshold, fp, det = calibrate_threshold(holdout_d, dirty_d, clean_percentile=95.0)
        print(f"Calibrated threshold: {threshold:.2f}  "
              f"(expected FP {fp:.1%} on clean, {det:.1%} single-reading detection on dirty)")
    else:
        print("No dirty_subfolder given -- building baseline from ALL clean clips.")
        baseline = BaselineProfile(len(DEFAULT_DRIFT_FEATURES), min_samples=len(clean_feats))
        for f in clean_feats:
            baseline.update(vec(f))
        baseline.freeze()

        if Path(POOLED_CALIBRATION_PATH).exists():
            with open(POOLED_CALIBRATION_PATH) as fh:
                pooled = json.load(fh)
            if pooled["drift_features"] != DEFAULT_DRIFT_FEATURES:
                print(f"WARNING: {POOLED_CALIBRATION_PATH} was calibrated on a "
                      f"different feature set than DEFAULT_DRIFT_FEATURES -- ignoring it.")
                threshold = float(np.sqrt(np.percentile(np.random.chisquare(
                    len(DEFAULT_DRIFT_FEATURES), 200000), 99)))
            else:
                threshold = pooled["threshold"]
                print(f"Using pooled threshold from {POOLED_CALIBRATION_PATH}: "
                      f"{threshold:.2f} (calibrated across {pooled['n_sites']} pilot "
                      f"site(s), pooled detection rate {pooled['pooled_detection_rate']:.1%} "
                      f"at {pooled['pooled_fp_rate']:.1%} FP)")
        else:
            k = len(DEFAULT_DRIFT_FEATURES)
            threshold = float(np.sqrt(np.percentile(np.random.chisquare(k, 200000), 99)))
            print(f"No {POOLED_CALIBRATION_PATH} found -- falling back to the theoretical "
                  f"chi-square threshold: {threshold:.2f} (k={k}). Run "
                  f"calibrate_pooled_threshold.py on your pilot sites for a real one.")

    detector = DriftDetector(baseline, threshold, alpha=0.15, patience=3)

    out_path = f"baseline_{house}.pkl"
    joblib.dump(detector, out_path)
    print(f"\nSaved frozen baseline + detector to {out_path}")


if __name__ == "__main__":
    main()
