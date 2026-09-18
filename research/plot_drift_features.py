"""
plot_drift_features.py
One box-plot comparison per drift feature (DEFAULT_DRIFT_FEATURES in
drift_detector.py) between data/MERV1_CLEAN and data/FPR12 -- the same
5-feature vector BaselineProfile/DriftDetector actually use in
production, so these plots show exactly what the Mahalanobis distance
is judging.

Reuses extract_features_v2.librosa_features() for the feature math so
this can't drift out of sync with what the live pipeline computes.

Usage:
  python plot_drift_features.py
Caches per-clip features to drift_feature_comparison.csv so re-plotting
doesn't require re-extracting from ~3000 wav files every time; delete
that file to force a fresh extraction.
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

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drift_detector import DEFAULT_DRIFT_FEATURES
from extract_features_v2 import SR, librosa_features

CACHE_PATH = Path("results/tables/drift_feature_comparison.csv")

GROUPS = [
    ("data/MERV1_CLEAN", "CLEAN"),
    ("data/FPR12", "DIRTY"),
]

FEATURE_LABELS = {
    "low_freq_energy_ratio": "Low-freq (20-500 Hz) energy fraction",
    "centroid_mean": "Spectral centroid (Hz)",
    "rolloff_mean": "Spectral rolloff, 85% energy (Hz)",
    "flatness_mean": "Spectral flatness (0-1, tonal->noisy)",
    "bandwidth_mean": "Spectral bandwidth (Hz)",
}


def build_dataset():
    rows = []
    for folder, label in GROUPS:
        files = sorted(Path(folder).glob("*.wav"))
        print(f"{label} ({folder}): {len(files)} files")
        for i, f in enumerate(files):
            if i % 200 == 0:
                print(f"  {i}/{len(files)}")
            y, _ = librosa.load(f, sr=SR, mono=True)
            if len(y) < SR * 0.5:
                continue
            feats = librosa_features(y, SR)
            feats["label"] = label
            feats["file"] = f.name
            rows.append(feats)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE_PATH, index=False)
    return df


def main():
    if CACHE_PATH.exists():
        print(f"Using cached {CACHE_PATH} (delete it to re-extract)")
        df = pd.read_csv(CACHE_PATH)
    else:
        df = build_dataset()

    print(f"\n{len(df)} usable clips total")
    print(df.groupby("label").size())

    clean = df[df["label"] == "CLEAN"]
    dirty = df[df["label"] == "DIRTY"]

    for feat in DEFAULT_DRIFT_FEATURES:
        fig, ax = plt.subplots(figsize=(5, 5))
        data = [clean[feat].dropna(), dirty[feat].dropna()]
        bp = ax.boxplot(data, labels=[f"CLEAN\n(n={len(data[0])})", f"DIRTY\n(n={len(data[1])})"],
                        patch_artist=True, showfliers=False, widths=0.5)
        bp["boxes"][0].set_facecolor("#22c55e66")
        bp["boxes"][1].set_facecolor("#ef444466")

        ax.set_title(FEATURE_LABELS.get(feat, feat))
        ax.set_ylabel(feat)
        ax.grid(axis="y", alpha=0.3)

        c_mean, d_mean = clean[feat].mean(), dirty[feat].mean()
        pct = (d_mean - c_mean) / abs(c_mean) * 100 if c_mean else float("nan")
        ax.text(0.5, -0.14, f"mean: {c_mean:.4g} -> {d_mean:.4g}  ({pct:+.1f}%)",
                transform=ax.transAxes, ha="center", fontsize=9, color="#555")

        fig.tight_layout()
        out = f"results/figures/drift_feature_{feat}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
