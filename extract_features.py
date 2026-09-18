"""
extract_features.py
Extracts spectral features from WAV clips for FPR rating classification.
"""
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import re
from datetime import datetime

DATA_DIR = "data"
CLASSES = ["FPR5", "FPR12"]
SR = 16000  # matches your I2S capture rate

def extract_features(file_path, sr=SR):
    y, _ = librosa.load(file_path, sr=sr, mono=True)

    # Guard against silent/short clips
    if len(y) < sr * 0.5:
        return None

    features = {}

    # MFCCs (timbre / spectral shape)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc{i}_mean"] = np.mean(mfccs[i])
        features[f"mfcc{i}_std"] = np.std(mfccs[i])

    # Spectral descriptors
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    features["centroid_mean"] = np.mean(centroid)
    features["centroid_std"] = np.std(centroid)
    features["bandwidth_mean"] = np.mean(bandwidth)
    features["rolloff_mean"] = np.mean(rolloff)
    features["flatness_mean"] = np.mean(flatness)
    features["zcr_mean"] = np.mean(zcr)
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)

    # Low-frequency energy ratio — airflow noise tends to concentrate here
    # and differs with static pressure drop across filter ratings
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    low_band = S[(freqs >= 20) & (freqs <= 500), :]
    features["low_freq_energy_ratio"] = np.sum(low_band) / (np.sum(S) + 1e-9)

    return features

def parse_timestamp(filename):
    match = re.match(r"recording_(\d{8})_(\d{6})", filename)
    if not match:
        return None
    date_str, time_str = match.groups()
    return datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")

def assign_sessions(files, gap_minutes=10):
    """
    Groups clips into sessions based on time gaps between consecutive
    recordings. A gap larger than gap_minutes starts a new session.
    Returns dict {filename: session_id}.
    """
    timestamped = [(f, parse_timestamp(f.name)) for f in files]
    timestamped = [(f, ts) for f, ts in timestamped if ts is not None]
    timestamped.sort(key=lambda x: x[1])

    session_map = {}
    session_id = 0
    prev_ts = None
    for f, ts in timestamped:
        if prev_ts is not None and (ts - prev_ts).total_seconds() > gap_minutes * 60:
            session_id += 1
        session_map[f.name] = f"session{session_id}"
        prev_ts = ts
    return session_map

def build_dataset():
    rows = []
    for label in CLASSES:
        folder = Path(DATA_DIR) / label
        if not folder.exists():
            print(f"Warning: {folder} not found, skipping")
            continue
        wav_files = list(folder.glob("*.wav"))
        print(f"{label}: {len(wav_files)} files")

        session_map = assign_sessions(wav_files, gap_minutes=10)
        n_sessions = len(set(session_map.values()))
        print(f"  -> {n_sessions} distinct sessions detected")

        for f in wav_files:
            feats = extract_features(f)
            if feats is None:
                continue
            feats["label"] = label
            feats["file"] = f.name
            # Prefix with label so FPR5/FPR12 sessions can never share an ID
            feats["session"] = f"{label}_{session_map.get(f.name, 'unknown')}"
            rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv("features.csv", index=False)
    print(f"\nSaved {len(df)} rows to features.csv")
    return df

if __name__ == "__main__":
    build_dataset()