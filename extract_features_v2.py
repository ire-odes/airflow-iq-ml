"""
extract_features_v2.py
Binary FPR5 (clean) / FPR12 (dirty) feature extraction for AirFlow IQ.

Feature set = your existing librosa features + the two feature families
from Siegel et al. 2017 that were missing:

  1. Binned FFT energies, windowed 20 Hz -> FMAX (their Section 5.1:
     they window the DFT to the frequency below which ~75% of cumulative
     spectral energy lives; for their vehicles this was ~3350 Hz).
  2. DWT features: Daubechies-4 wavelet, level-10 decomposition,
     mean/std/skew of coefficients at each level = 33 features
     (their exact construction).

Session IDs are parsed from filenames so the training script can do
leak-free, session-grouped splits.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import pywt
from scipy.stats import skew

DATA_DIR = "data"
CLASSES = ["FPR5", "FPR12"]          # binary: clean vs dirty
SR = 16000                            # matches I2S capture rate

# RMS floor below which a clip is treated as silence (HVAC off) rather than
# a real reading -- see extract_features()'s guard for the full reasoning.
# Uncalibrated placeholder; tune against real "known off" recordings.
SILENCE_RMS_THRESHOLD = 0.003

# --- Paper-style FFT windowing -------------------------------------------
FMIN = 20.0        # their lower cutoff: avoids 1/f noise contamination
FMAX = 3350.0      # their 75% cumulative-energy cutoff; recompute for
                   # your rig with compute_energy_cutoff() below and
                   # update this constant if it differs materially.
N_FFT_BINS = 64    # number of energy bins between FMIN and FMAX.
                   # The paper used very fine bins (0.25 Hz) on 48 kHz
                   # audio; at 16 kHz / 1-2 s clips your frequency
                   # resolution is ~0.5-1 Hz, so we aggregate into
                   # coarser bands. 64 bins ~= 52 Hz wide each.

# --- Paper-style DWT config ----------------------------------------------
WAVELET = "db4"    # Daubechies-4, per the paper
DWT_LEVEL = 10     # level-10 decomposition, per the paper


def compute_energy_cutoff(wav_paths, sr=SR, target=0.75):
    """Reproduce the paper's Fig. 12 analysis: find the frequency below
    which `target` fraction of cumulative spectral energy lives, averaged
    across a sample of clips. Run once on your real data to sanity-check
    FMAX; the HVAC blower may concentrate energy differently than a car
    engine at idle."""
    cutoffs = []
    for p in wav_paths:
        y, _ = librosa.load(p, sr=sr, mono=True)
        S = np.abs(librosa.stft(y)).mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr)
        mask = freqs >= FMIN
        S, freqs = S[mask], freqs[mask]
        cdf = np.cumsum(S) / (np.sum(S) + 1e-12)
        cutoffs.append(freqs[np.searchsorted(cdf, target)])
    return float(np.median(cutoffs))


def fft_band_features(y, sr=SR):
    """Binned FFT energies between FMIN and FMAX, RMS-normalized.

    Mirrors the paper's DFT feature block: they normalized each chunk by
    RMS to control for mic position / environmental level differences
    (their Section 4.2), then used binned spectral magnitudes as
    classifier inputs. Normalization matters even more for you, since a
    model keying on absolute loudness is exactly the session-leakage
    trap you already hit."""
    rms = np.sqrt(np.mean(y ** 2)) + 1e-12
    y = y / rms

    S = np.abs(librosa.stft(y, n_fft=2048)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    edges = np.linspace(FMIN, FMAX, N_FFT_BINS + 1)
    feats = {}
    total = np.sum(S[(freqs >= FMIN) & (freqs <= FMAX)]) + 1e-12
    for i in range(N_FFT_BINS):
        band = S[(freqs >= edges[i]) & (freqs < edges[i + 1])]
        # store as fraction of in-window energy -> scale invariant
        feats[f"fft_band_{i:02d}"] = float(np.sum(band) / total)
    return feats


def dwt_features(y):
    """The paper's 33 wavelet features: db4, level 10, then mean/std/skew
    of the coefficient array at each decomposition level (10 detail
    levels + 1 approximation = 11 arrays x 3 stats = 33 features)."""
    coeffs = pywt.wavedec(y, WAVELET, level=DWT_LEVEL)
    feats = {}
    for lvl, c in enumerate(coeffs):
        feats[f"dwt_l{lvl:02d}_mean"] = float(np.mean(c))
        feats[f"dwt_l{lvl:02d}_std"] = float(np.std(c))
        feats[f"dwt_l{lvl:02d}_skew"] = float(skew(c)) if len(c) > 2 else 0.0
    return feats


def librosa_features(y, sr=SR):
    """Your existing feature block, unchanged: MFCC mean/std, spectral
    descriptors, and the physically-motivated low-frequency energy
    ratio."""
    feats = {}

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats[f"mfcc{i}_mean"] = float(np.mean(mfccs[i]))
        feats[f"mfcc{i}_std"] = float(np.std(mfccs[i]))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    feats["centroid_mean"] = float(np.mean(centroid))
    feats["centroid_std"] = float(np.std(centroid))
    feats["bandwidth_mean"] = float(np.mean(bandwidth))
    feats["rolloff_mean"] = float(np.mean(rolloff))
    feats["flatness_mean"] = float(np.mean(flatness))
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))

    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_band = S[(freqs >= 20) & (freqs <= 500), :]
    feats["low_freq_energy_ratio"] = float(np.sum(low_band) / (np.sum(S) + 1e-9))

    return feats


def extract_features(file_path, sr=SR):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    if len(y) < sr * 0.5:                       # guard: too-short clips
        return None
    if np.sqrt(np.mean(y ** 2)) < SILENCE_RMS_THRESHOLD:
        # guard: near-silent clips (HVAC off, mic disconnected, etc.) --
        # skip full feature extraction entirely rather than run it on
        # noise, and critically, keep it out of drift_detector.py's
        # baseline warmup: mixing "unit off" and "unit on" acoustics into
        # one baseline would corrupt the learned covariance. This default
        # is a conservative guess (clearly above digital-silence/mic-noise
        # floor, clearly below expected active-airflow RMS) -- it has NOT
        # been calibrated against real "known off" recordings the way the
        # drift threshold was; do that once you have some (same recipe as
        # calibrate_pooled_threshold.py: collect known-off clips, look at
        # their rms_mean distribution, set this at a percentile of it).
        #
        # Briefly disabled 2026-08-20 while diagnosing zero device_baselines
        # rows fleet-wide -- turned out to be an unrelated Task Scheduler
        # credential failure (the poller had never actually run
        # automatically), not this guard. Confirmed safe to restore: real
        # devices' current recordings measured 33x-113x this threshold, so
        # it was never rejecting genuine "on" audio -- see
        # check_real_rms.py in the scratchpad from that session for the
        # actual numbers if this needs re-checking later.
        return None

    feats = {}
    feats.update(librosa_features(y, sr))
    feats.update(fft_band_features(y, sr))
    feats.update(dwt_features(y))
    return feats


def parse_session_id(filename):
    """Session = date + hour bucket from the recording timestamp.
    Clips sharing a session must never straddle a train/test split."""
    match = re.match(r"recording_(\d{8})_(\d{6})", filename)
    if not match:
        return "unknown"
    date_str, time_str = match.groups()
    return f"{date_str}_{time_str[:2]}"


def build_dataset():
    rows = []
    for label in CLASSES:
        folder = Path(DATA_DIR) / label
        if not folder.exists():
            print(f"Warning: {folder} not found, skipping")
            continue
        wav_files = sorted(folder.glob("*.wav"))
        print(f"{label}: {len(wav_files)} files")
        for f in wav_files:
            feats = extract_features(f)
            if feats is None:
                continue
            feats["label"] = label
            feats["file"] = f.name
            feats["session"] = f"{label}_{parse_session_id(f.name)}"
            rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv("features.csv", index=False)
    n_sessions = df.groupby("label")["session"].nunique()
    print(f"\nSaved {len(df)} rows to features.csv")
    print(f"Sessions per class:\n{n_sessions}")
    if (n_sessions < 3).any():
        print("WARNING: fewer than 3 sessions in a class -> grouped CV "
              "will be unstable. Record more independent sessions.")
    return df


if __name__ == "__main__":
    build_dataset()
