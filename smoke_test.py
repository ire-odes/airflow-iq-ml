"""
smoke_test.py
End-to-end sanity check on synthetic data:
  clean (FPR5)  = broadband noise, energy concentrated low
  dirty (FPR12) = energy shifted upward (the paper's Section 2 physics)
Session structure: 4 sessions per class, session-specific background
tone so grouped splitting actually matters.
Then: extract -> train RF -> simulate a device through the
FilterHealthMonitor (clean warmup -> gradual clogging).
"""

import subprocess
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

SR = 16000
DUR = 2.0
RNG = np.random.default_rng(7)


def shaped_noise(shift, session_tone, n=int(SR * DUR)):
    """Noise with a spectral tilt controlled by `shift` in [0,1]:
    0 = low-heavy (clean), 1 = pushed upward (dirty)."""
    white = RNG.normal(size=n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    center = 300 + 900 * shift                      # dominant band moves up
    envelope = np.exp(-((freqs - center) ** 2) / (2 * (250 + 200 * shift) ** 2))
    envelope += 0.05                                # broadband floor
    y = np.fft.irfft(spec * envelope, n)
    # session-specific hum (simulates per-session ambient confound)
    t = np.arange(n) / SR
    y += 0.05 * np.sin(2 * np.pi * session_tone * t)
    return (y / (np.abs(y).max() + 1e-9) * 0.7).astype(np.float32)


def make_data():
    cfg = {"FPR5": 0.0, "FPR12": 1.0}
    for label, shift in cfg.items():
        d = Path("data") / label
        d.mkdir(parents=True, exist_ok=True)
        for s in range(4):                          # 4 sessions/class
            tone = 60 + 15 * s                      # session confound
            date = f"2026070{s + 1}"
            for c in range(12):                     # 12 clips/session
                # small within-class variation
                y = shaped_noise(shift + RNG.normal(0, 0.05), tone)
                sf.write(d / f"recording_{date}_{10 + s:02d}{c:02d}00.wav",
                         y, SR)
    print("Synthetic data written: 4 sessions x 12 clips x 2 classes")


def simulate_device():
    """Run a fake device through the monitor: 130 clean readings
    (warmup + steady state), then gradual clogging over 60 readings."""
    import sys
    sys.path.insert(0, ".")
    from extract_features_v2 import extract_features
    from drift_detector import (FilterHealthMonitor, calibrate_threshold,
                                BaselineProfile, DEFAULT_DRIFT_FEATURES)

    bundle = joblib.load("model_bundle.pkl")
    model = bundle["classifier"]
    le = bundle["label_encoder"]
    sel = bundle["selected_features"]
    dirty_idx = int(np.where(le.classes_ == "FPR12")[0][0])

    def classifier_fn(feats):
        x = np.array([[feats[k] for k in sel]])
        p = model.predict_proba(x)[0][dirty_idx]
        return ("FPR12" if p >= 0.5 else "FPR5", float(p))

    # --- calibrate drift threshold from labeled synthetic data --------
    tmpdir = Path("caldata"); tmpdir.mkdir(exist_ok=True)
    def feats_for(shift, tone, n):
        out = []
        for i in range(n):
            y = shaped_noise(shift, tone)
            p = tmpdir / "c.wav"; sf.write(p, y, SR)
            out.append(extract_features(p))
        return out

    base = BaselineProfile(len(DEFAULT_DRIFT_FEATURES), min_samples=30)
    clean_feats = feats_for(0.0, 72, 60)
    for f in clean_feats[:30]:
        base.update(np.array([f[k] for k in DEFAULT_DRIFT_FEATURES]))
    base.freeze()
    clean_d = [base.distance(np.array([f[k] for k in DEFAULT_DRIFT_FEATURES]))
               for f in clean_feats[30:]]
    dirty_feats = feats_for(1.0, 72, 30)
    dirty_d = [base.distance(np.array([f[k] for k in DEFAULT_DRIFT_FEATURES]))
               for f in dirty_feats]
    thr, fp, det = calibrate_threshold(clean_d, dirty_d)
    print(f"\nCalibrated drift threshold={thr:.2f} "
          f"(expected FP {fp:.1%}, dirty detection {det:.1%})")

    # --- run the monitor ----------------------------------------------
    mon = FilterHealthMonitor(classifier_fn, min_baseline_samples=30,
                              drift_threshold=thr, ewma_alpha=0.15,
                              patience=3)
    log = []
    for i in range(130):                            # clean phase
        y = shaped_noise(0.0, 72)
        p = Path("caldata/s.wav"); sf.write(p, y, SR)
        log.append(mon.process(extract_features(p)))
    for i in range(60):                             # gradual clogging
        shift = min(1.0, i / 40)
        y = shaped_noise(shift, 72)
        p = Path("caldata/s.wav"); sf.write(p, y, SR)
        log.append(mon.process(extract_features(p)))

    first_dirty = next((i for i, r in enumerate(log)
                        if r["decision"] == "dirty"), None)
    n_warm = sum(1 for r in log if r["state"] == "warm")
    fp_clean_phase = sum(1 for r in log[:130] if r["decision"] == "dirty")
    print(f"Monitor: {len(log)} readings, {n_warm} in WARM state")
    print(f"False dirty calls during clean phase: {fp_clean_phase}/130")
    print(f"First 'dirty' decision at reading #{first_dirty} "
          f"(clogging began at #130)")
    flags = [(i, r['flag']) for i, r in enumerate(log) if r['flag']]
    print(f"Flags raised: {len(flags)}"
          + (f", first: #{flags[0][0]}: {flags[0][1]}" if flags else ""))


if __name__ == "__main__":
    make_data()
    subprocess.run(["python", "extract_features_v2.py"], check=True)
    subprocess.run(["python", "train_classifier.py"], check=True)
    simulate_device()
