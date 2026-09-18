"""
calibrate_pooled_threshold.py
Calibrates ONE shared drift-alert threshold from multiple pilot sites that
each have known clean + dirty recordings, so new customer installs (which
will only ever provide clean warmup data) don't have to fall back to the
theoretical chi-square threshold guess in FilterHealthMonitor/build_baseline.py
-- which we already showed is measurably off (small-sample covariance noise,
shrinkage bias, and non-Gaussian features all violate the assumptions that
guess relies on).

Why pooling across sites should transfer to a brand-new customer: each
site gets its OWN BaselineProfile fit to its OWN clean recordings, and the
Mahalanobis distance is computed relative to that site's own baseline --
so despite wildly different absolute acoustic scales between sites (e.g.
lab rig vs. a real house), the resulting distance numbers are already
normalized to "how many baseline-sigma away from THIS install's own
normal." Pooling those normalized distances across several different
physical setups, rather than using just one site's threshold, guards
against overfitting the cutoff to one install's idiosyncrasies.

Usage:
  python calibrate_pooled_threshold.py <clean_dir1> <dirty_dir1> [<clean_dir2> <dirty_dir2> ...]

Example (lab rig + one real house, pooled):
  python calibrate_pooled_threshold.py data/FPR5 data/FPR12 data/AbrahamHouse/MERV10 data/AbrahamHouse/FPR12

Output: pooled_calibration.json -- read by build_baseline.py whenever a new
customer install has no dirty data to calibrate against.
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

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from extract_features_v2 import extract_features
from drift_detector import (BaselineProfile, calibrate_threshold,
                            check_recording_continuity, DEFAULT_DRIFT_FEATURES,
                            MAX_CONTINUITY_GAP_MIN)

WARMUP_FRACTION = 0.7
CLEAN_PERCENTILE = 95.0
OUT_PATH = "models/pooled_calibration.json"


def vec(feats):
    return np.array([feats[k] for k in DEFAULT_DRIFT_FEATURES])


def load_feats(folder):
    feats = []
    for f in sorted(Path(folder).glob("*.wav")):
        v = extract_features(f)
        if v is not None:
            feats.append(v)
    return feats


def process_site(clean_dir, dirty_dir):
    is_continuous, max_gap = check_recording_continuity(clean_dir)
    if not is_continuous:
        print(f"  SKIPPING {clean_dir}: max gap between clips is {max_gap:.0f} min "
              f"(> {MAX_CONTINUITY_GAP_MIN} min) -- this looks like multiple "
              f"disjoint recording sessions, not one continuous device warmup. "
              f"Only pool sites recorded as a single continuous run.")
        return None

    clean_feats = load_feats(clean_dir)
    dirty_feats = load_feats(dirty_dir)
    if len(clean_feats) < 10 or len(dirty_feats) < 10:
        print(f"  WARNING: {clean_dir} / {dirty_dir} -- too few usable clips "
              f"({len(clean_feats)} clean, {len(dirty_feats)} dirty), skipping site")
        return None

    n_warm = int(len(clean_feats) * WARMUP_FRACTION)
    warm, holdout = clean_feats[:n_warm], clean_feats[n_warm:]

    baseline = BaselineProfile(len(DEFAULT_DRIFT_FEATURES), min_samples=n_warm)
    for f in warm:
        baseline.update(vec(f))
    baseline.freeze()

    holdout_d = [baseline.distance(vec(f)) for f in holdout]
    dirty_d = [baseline.distance(vec(f)) for f in dirty_feats]
    own_thr, own_fp, own_det = calibrate_threshold(holdout_d, dirty_d, CLEAN_PERCENTILE)

    print(f"  {clean_dir} vs {dirty_dir}: {len(clean_feats)} clean "
          f"({n_warm} warm / {len(holdout)} holdout), {len(dirty_feats)} dirty")
    print(f"    own single-site threshold={own_thr:.2f}  "
          f"(FP {own_fp:.1%}, detection {own_det:.1%})")

    return {
        "clean_dir": str(clean_dir), "dirty_dir": str(dirty_dir),
        "n_clean": len(clean_feats), "n_dirty": len(dirty_feats),
        "holdout_clean_distances": holdout_d, "dirty_distances": dirty_d,
        "own_threshold": own_thr,
    }


def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2 != 0:
        print("Usage: python calibrate_pooled_threshold.py <clean_dir1> <dirty_dir1> [<clean_dir2> <dirty_dir2> ...]")
        sys.exit(1)
    pairs = list(zip(args[0::2], args[1::2]))

    print(f"Calibrating from {len(pairs)} site(s):")
    sites = []
    for clean_dir, dirty_dir in pairs:
        site = process_site(clean_dir, dirty_dir)
        if site is not None:
            sites.append(site)

    if not sites:
        print("No usable sites -- nothing to calibrate.")
        sys.exit(1)

    pooled_clean = [d for s in sites for d in s["holdout_clean_distances"]]
    pooled_dirty = [d for s in sites for d in s["dirty_distances"]]
    pooled_thr, pooled_fp, pooled_det = calibrate_threshold(
        pooled_clean, pooled_dirty, CLEAN_PERCENTILE)

    print(f"\n=== Pooled across {len(sites)} site(s), "
          f"{len(pooled_clean)} held-out clean / {len(pooled_dirty)} dirty distances ===")
    print(f"Pooled threshold: {pooled_thr:.2f}")
    print(f"Pooled FP rate on held-out clean: {pooled_fp:.1%}")
    print(f"Pooled detection rate on dirty:   {pooled_det:.1%}")

    print(f"\nHow the pooled threshold would perform on EACH site individually "
          f"(vs. that site's own tuned threshold):")
    for s in sites:
        cd, dd = np.array(s["holdout_clean_distances"]), np.array(s["dirty_distances"])
        fp = float(np.mean(cd > pooled_thr))
        det = float(np.mean(dd > pooled_thr))
        print(f"  {s['clean_dir']:35s} own={s['own_threshold']:.2f}  "
              f"pooled={pooled_thr:.2f}  -> FP {fp:.1%}, detection {det:.1%}")

    out = {
        "threshold": pooled_thr,
        "drift_features": DEFAULT_DRIFT_FEATURES,
        "alpha": 0.15,
        "patience": 3,
        "clean_percentile": CLEAN_PERCENTILE,
        "pooled_fp_rate": pooled_fp,
        "pooled_detection_rate": pooled_det,
        "n_sites": len(sites),
        "sites": [{"clean_dir": s["clean_dir"], "dirty_dir": s["dirty_dir"],
                  "n_clean": s["n_clean"], "n_dirty": s["n_dirty"],
                  "own_threshold": s["own_threshold"]} for s in sites],
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
