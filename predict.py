"""
predict.py
Given a WAV clip, predicts filter condition and prints "clean" or "clogged".

Must use the SAME feature extraction (extract_features_v2) that produced
features.csv in train_classifier.py -- mismatching them (e.g. the old
extract_features.py) raises a KeyError, since the model expects the
FFT-band + DWT columns v2 adds on top of the original librosa features.

Loads model_bundle.pkl -- a single file with the classifier, label
encoder, and feature lists together, so they can never individually go
stale relative to each other the way 4 separate files could (see
train_classifier.py's persist step for why that used to happen).
"""
import sys

import joblib

from extract_features_v2 import extract_features

BUNDLE_PATH = "model_bundle.pkl"

# Raw dataset labels are Filter Performance Ratings used as clean/dirty
# proxies while collecting audio (see train_classifier.py: DIRTY_LABEL =
# "FPR12"): FPR5 = low-restriction filter = clean, FPR12 = high-restriction
# filter = simulates a clogged filter's acoustic signature.
LABEL_MAP = {"FPR5": "clean", "FPR12": "clogged"}

_bundle = joblib.load(BUNDLE_PATH)
clf = _bundle["classifier"]
le = _bundle["label_encoder"]
feature_cols = _bundle["selected_features"]


def predict_fpr(wav_path):
    """Returns (label, confidence, probs) where label is "clean" or
    "clogged", or None if the clip was too short/silent to featurize."""
    feats = extract_features(wav_path)
    if feats is None:
        return None

    missing = [k for k in feature_cols if k not in feats]
    if missing:
        raise ValueError(
            f"Extracted features are missing {len(missing)} column(s) the "
            f"model expects (e.g. {missing[:3]}). This usually means "
            f"predict.py's feature extractor is out of sync with whatever "
            f"produced features.csv for training."
        )

    X = [[feats[k] for k in feature_cols]]
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    raw_label = le.inverse_transform([pred])[0]
    confidence = float(proba[pred])
    probs = {LABEL_MAP.get(cls, cls): float(p)
             for cls, p in zip(le.classes_, proba)}
    return LABEL_MAP.get(raw_label, raw_label), confidence, probs


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_wav>")
        sys.exit(1)

    result = predict_fpr(sys.argv[1])
    if result is None:
        print("Clip too short/silent to classify (need >= 0.5s of audio).")
        sys.exit(1)

    label, conf, probs = result
    print(f"Predicted: {label} ({conf:.1%} confidence)")
    print(f"All probabilities: {probs}")
