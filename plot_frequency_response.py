"""
plot_frequency_response.py
Gain-vs-frequency line plot: mean binned FFT energy fraction (20-3350 Hz,
64 bands, RMS-normalized -- extract_features_v2.fft_band_features()) for
data/MERV1_CLEAN vs data/FPR12, shaded by +/-1 SEM across clips.

Same idea/style as analyze_frequency_response.py (which compares
FPR5 vs FPR12 from features.csv), applied to the MERV1_CLEAN/FPR12 pair
plot_drift_features.py already used, with the CLEAN/DIRTY legend labels
from that same comparison.

Usage:
  python plot_frequency_response.py
Caches per-clip band energies to fft_band_comparison.csv; delete it to
force re-extraction.
"""
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extract_features_v2 import SR, FMIN, FMAX, N_FFT_BINS, fft_band_features

CACHE_PATH = Path("fft_band_comparison.csv")

GROUPS = [
    ("data/MERV1_CLEAN", "CLEAN"),
    ("data/FPR12", "DIRTY"),
]
COLOR = {"CLEAN": "#22c55e", "DIRTY": "#ef4444"}


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
            feats = fft_band_features(y, SR)
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

    band_cols = [f"fft_band_{i:02d}" for i in range(N_FFT_BINS)]
    edges = np.linspace(FMIN, FMAX, N_FFT_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    clean = df[df.label == "CLEAN"]
    dirty = df[df.label == "DIRTY"]
    clean_mean, clean_sem = clean[band_cols].mean().values, clean[band_cols].sem().values
    dirty_mean, dirty_sem = dirty[band_cols].mean().values, dirty[band_cols].sem().values

    clean_peak = centers[np.argmax(clean_mean)]
    dirty_peak = centers[np.argmax(dirty_mean)]
    print(f"\nPeak energy band -- CLEAN: {clean_peak:.0f} Hz | DIRTY: {dirty_peak:.0f} Hz")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(centers, clean_mean, color=COLOR["CLEAN"], lw=2, label="CLEAN")
    ax.fill_between(centers, clean_mean - clean_sem, clean_mean + clean_sem,
                    color=COLOR["CLEAN"], alpha=0.2, linewidth=0)
    ax.plot(centers, dirty_mean, color=COLOR["DIRTY"], lw=2, label="DIRTY")
    ax.fill_between(centers, dirty_mean - dirty_sem, dirty_mean + dirty_sem,
                    color=COLOR["DIRTY"], alpha=0.2, linewidth=0)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean normalized spectral energy fraction (gain proxy)")
    ax.set_title("Gain vs. frequency: MERV1_CLEAN vs. FPR12\n(shaded band = ±1 SEM across clips)")
    ax.set_xlim(FMIN, FMAX)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("frequency_response_MERV1_vs_FPR12.png")
    print("\nSaved fft_band_comparison.csv, frequency_response_MERV1_vs_FPR12.png")


if __name__ == "__main__":
    main()
