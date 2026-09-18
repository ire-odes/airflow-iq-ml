"""
run_house_baseline.py
Builds a per-device drift baseline from real "clean" recordings of one
house's HVAC system, then feeds it real recordings taken after a known
filter swap, to see whether the drift detector (and the global
classifier, as a sanity check) pick up the change.

Expects two chronologically-ordered folders of clips under a house dir:
  data/<house>/<clean_subfolder>/*.wav  -- baseline warmup (assumed clean)
  data/<house>/<dirty_subfolder>/*.wav  -- recorded after a filter swap

Usage:
  python run_house_baseline.py AbrahamHouse MERV10 FPR12
"""
import sys
from pathlib import Path

import joblib
import numpy as np

from extract_features_v2 import extract_features
from drift_detector import FilterHealthMonitor

DATA_DIR = Path("data")
MIN_BASELINE_SAMPLES = 40  # short real-world test run, not the full ~2h warmup


def build_classifier_fn():
    bundle = joblib.load("model_bundle.pkl")
    model = bundle["classifier"]
    le = bundle["label_encoder"]
    sel = bundle["selected_features"]
    dirty_idx = int(np.where(le.classes_ == "FPR12")[0][0])

    def classifier_fn(feats):
        x = np.array([[feats[k] for k in sel]])
        p = float(model.predict_proba(x)[0][dirty_idx])
        return ("clogged" if p >= 0.5 else "clean"), p

    return classifier_fn


def run_clips(monitor, files, tag):
    log = []
    for f in files:
        feats = extract_features(f)
        if feats is None:
            print(f"  skipping {f.name}: too short/silent")
            continue
        log.append((f.name, monitor.process(feats)))
    return log


def main():
    if len(sys.argv) != 4:
        print("Usage: python run_house_baseline.py <house_dir> <clean_subfolder> <dirty_subfolder>")
        sys.exit(1)
    house, clean_name, dirty_name = sys.argv[1:4]

    clean_files = sorted((DATA_DIR / house / clean_name).glob("*.wav"))
    dirty_files = sorted((DATA_DIR / house / dirty_name).glob("*.wav"))
    print(f"{clean_name}: {len(clean_files)} clips (baseline warmup)")
    print(f"{dirty_name}: {len(dirty_files)} clips (post-swap test)")

    if len(clean_files) < MIN_BASELINE_SAMPLES:
        print(f"WARNING: only {len(clean_files)} clean clips, fewer than "
              f"MIN_BASELINE_SAMPLES={MIN_BASELINE_SAMPLES}. Baseline may "
              f"never freeze; lower MIN_BASELINE_SAMPLES and rerun.")

    monitor = FilterHealthMonitor(
        build_classifier_fn(),
        min_baseline_samples=MIN_BASELINE_SAMPLES,
        ewma_alpha=0.15,
        patience=3,
    )

    print(f"\n--- Warming up baseline on {clean_name} ---")
    warm_log = run_clips(monitor, clean_files, clean_name)
    froze_at = next((i for i, (_, r) in enumerate(warm_log) if r["state"] == "warm"), None)
    if froze_at is not None:
        print(f"Baseline froze after clip #{froze_at} of {len(warm_log)}")
    else:
        print("Baseline never froze during warmup -- not enough clean-looking "
              "clips reached the classifier's p_dirty<0.5 gate. See flags below.")
    warm_flags = [(n, r["flag"]) for n, r in warm_log if r["flag"]]
    print(f"Warmup flags: {len(warm_flags)}"
          + (f", first: {warm_flags[0]}" if warm_flags else ""))
    warm_clf_dirty = sum(1 for _, r in warm_log if r["classifier_label"] == "clogged")
    print(f"Classifier called {warm_clf_dirty}/{len(warm_log)} warmup clips 'clogged' "
          f"(these were withheld from the baseline, per the poisoning guard)")

    if froze_at is None:
        print("\nCan't evaluate drift on the dirty set without a frozen baseline. Exiting.")
        return

    print(f"\n--- Feeding {dirty_name} (post-swap) through the monitor ---")
    test_log = run_clips(monitor, dirty_files, dirty_name)

    n = len(test_log)
    n_drift_flagged = sum(1 for _, r in test_log if r["decision"] == "dirty")
    n_clf_flagged = sum(1 for _, r in test_log if r["classifier_label"] == "clogged")
    first_drift = next((i for i, (_, r) in enumerate(test_log) if r["decision"] == "dirty"), None)
    distances = [r["distance"] for _, r in test_log]
    ewmas = [r["ewma"] for _, r in test_log]

    print(f"Drift decision 'dirty':      {n_drift_flagged}/{n} clips "
          f"({n_drift_flagged/n:.1%})")
    print(f"Classifier decision 'clogged': {n_clf_flagged}/{n} clips "
          f"({n_clf_flagged/n:.1%})")
    print(f"First drift-flagged clip: #{first_drift} "
          f"({test_log[first_drift][0]})" if first_drift is not None
          else "Drift never crossed the alert threshold on this set.")
    print(f"Mahalanobis distance: min {min(distances):.2f} / "
          f"mean {np.mean(distances):.2f} / max {max(distances):.2f}")
    print(f"EWMA distance: starts {ewmas[0]:.2f}, ends {ewmas[-1]:.2f}, "
          f"threshold {monitor._detector.threshold:.2f}")

    disagreements = [(n_, r["flag"]) for n_, r in test_log if r["flag"]]
    print(f"Disagreement flags during test phase: {len(disagreements)}"
          + (f", first: {disagreements[0]}" if disagreements else ""))


if __name__ == "__main__":
    main()
