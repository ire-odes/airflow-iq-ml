"""
analyze_frequency_response.py
Quantifies the frequency-response difference between the clean (FPR5) and
clogged (FPR12) filter recordings, for reporting in the paper.

Reuses features.csv (already extracted by extract_features_v2.py) rather
than re-processing audio -- fft_band_00..63 there is already an RMS-
normalized, binned magnitude spectrum per clip (20-3350 Hz, ~52 Hz/bin),
so averaging it per class directly gives an empirical frequency-response
curve per filter condition.

Outputs:
  frequency_response_summary.csv   -- global spectral descriptors, per
                                       class, with significance + effect size
  frequency_band_comparison.csv    -- per-FFT-bin comparison (52 Hz bins)
  frequency_response_comparison.png -- averaged spectrum plot, clean vs clogged
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

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL_MAP = {"FPR5": "clean", "FPR12": "clogged"}
COLOR = {"clean": "#2a78d6", "clogged": "#e34948"}  # dataviz categorical slots 1 & 8

FMIN, FMAX = 20.0, 3350.0  # matches extract_features_v2.py windowing


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    return (a.mean() - b.mean()) / np.sqrt(pooled_var)


def main():
    df = pd.read_csv("results/features/features.csv")
    df["condition"] = df["label"].map(LABEL_MAP)
    clean = df[df.condition == "clean"]
    dirty = df[df.condition == "clogged"]
    print(f"clean (FPR5): {len(clean)} clips, {clean['session'].nunique()} sessions")
    print(f"clogged (FPR12): {len(dirty)} clips, {dirty['session'].nunique()} sessions")

    # ---- 1. Global spectral descriptors --------------------------------
    descriptors = {
        "centroid_mean": "Spectral centroid (Hz)",
        "centroid_std": "Spectral centroid variability (Hz)",
        "bandwidth_mean": "Spectral bandwidth (Hz)",
        "rolloff_mean": "Spectral rolloff, 85% energy (Hz)",
        "flatness_mean": "Spectral flatness (0-1, tonal->noisy)",
        "low_freq_energy_ratio": "Low-freq (20-500 Hz) energy fraction",
        "zcr_mean": "Zero-crossing rate",
    }

    rows = []
    for col, desc in descriptors.items():
        a, b = clean[col].dropna(), dirty[col].dropna()
        t, p_t = stats.ttest_ind(a, b, equal_var=False)
        u, p_u = stats.mannwhitneyu(a, b)
        d = cohens_d(a, b)
        rows.append({
            "feature": col,
            "description": desc,
            "clean_mean": a.mean(), "clean_std": a.std(),
            "clogged_mean": b.mean(), "clogged_std": b.std(),
            "pct_change": (b.mean() - a.mean()) / abs(a.mean()) * 100,
            "cohens_d": d,
            "welch_t_p": p_t,
            "mannwhitney_p": p_u,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv("results/tables/frequency_response_summary.csv", index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print("\n=== Global spectral descriptors: clean vs clogged ===")
    print(summary[["feature", "clean_mean", "clogged_mean", "pct_change",
                   "cohens_d", "mannwhitney_p"]].to_string(index=False))

    # ---- 2. Binned frequency response (fft_band_00..63) -----------------
    fft_cols = [c for c in df.columns if c.startswith("fft_band_")]
    edges = np.linspace(FMIN, FMAX, len(fft_cols) + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    clean_spec = clean[fft_cols].mean().values
    clean_sem = clean[fft_cols].sem().values
    dirty_spec = dirty[fft_cols].mean().values
    dirty_sem = dirty[fft_cols].sem().values

    band_d = np.array([cohens_d(clean[c], dirty[c]) for c in fft_cols])
    band_df = pd.DataFrame({
        "band_center_hz": centers,
        "band_low_hz": edges[:-1],
        "band_high_hz": edges[1:],
        "clean_mean_energy_frac": clean_spec,
        "clogged_mean_energy_frac": dirty_spec,
        "diff_clogged_minus_clean": dirty_spec - clean_spec,
        "cohens_d": band_d,
    })
    band_df.to_csv("results/tables/frequency_band_comparison.csv", index=False)

    clean_peak = centers[np.argmax(clean_spec)]
    dirty_peak = centers[np.argmax(dirty_spec)]
    print(f"\nPeak energy band -- clean: {clean_peak:.0f} Hz | "
          f"clogged: {dirty_peak:.0f} Hz")

    top = band_df.reindex(band_df.cohens_d.abs().sort_values(ascending=False).index).head(8)
    print("\nTop 8 frequency bands by class separation (|Cohen's d|):")
    print(top[["band_center_hz", "clean_mean_energy_frac",
              "clogged_mean_energy_frac", "cohens_d"]].to_string(index=False))

    # ---- 3. Plot: averaged frequency response ----------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(centers, clean_spec, color=COLOR["clean"], lw=2, label="Clean (FPR5)")
    ax.fill_between(centers, clean_spec - clean_sem, clean_spec + clean_sem,
                    color=COLOR["clean"], alpha=0.18, linewidth=0)
    ax.plot(centers, dirty_spec, color=COLOR["clogged"], lw=2, label="Clogged (FPR12)")
    ax.fill_between(centers, dirty_spec - dirty_sem, dirty_spec + dirty_sem,
                    color=COLOR["clogged"], alpha=0.18, linewidth=0)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean normalized spectral energy fraction")
    ax.set_title("Average frequency response: clean vs. clogged filter\n"
                "(shaded band = ±1 SEM across clips)")
    ax.set_xlim(FMIN, FMAX)
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("results/figures/frequency_response_comparison.png")
    print("\nSaved frequency_response_summary.csv, frequency_band_comparison.csv, "
          "results/figures/frequency_response_comparison.png")


if __name__ == "__main__":
    main()
