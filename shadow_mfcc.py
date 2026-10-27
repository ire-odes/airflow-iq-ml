"""
shadow_mfcc.py
A second drift detector that runs beside the real one, on the LoRa
firmware's MFCCs (firmware_mfcc.py, c1..c12) instead of the five spectral
features. It never changes a verdict: it reads the clip, keeps its own state
in a local file, and writes one CSV row per scored clip.

WHY IT EXISTS. Replaying archived audio, the five drift features missed a
back-to-back MERV10 -> FPR12 swap at P3's house entirely (0% detected) where
firmware MFCCs caught 96% with no false alarms. The same MFCCs also flagged
75-100% of a SAME-filter rig recording a month later, which is either real
loading the five features cannot see, or day-to-day sensitivity that would
false-alarm in the field. Archived audio cannot separate those. A live,
controlled test can: both detectors scoring the same clips, against baselines
from the same calibration, while the filter is changed on purpose.

c0 is left out because it is the only coefficient that moves with microphone
gain (the DCT weights for c>=1 sum to zero across the 26 bands); c1..c12 are
comparable across devices without calibrating gain.

LIFECYCLE, tied to the real detector's so the two stay comparable:
  - Builds its baseline only while the real one is warming. A device that is
    already warm when this first sees it waits for its next recalibration
    rather than calibrating on readings that may already be off-baseline.
  - Freezes on the same reading that freezes the real baseline, provided it
    holds MIN_FREEZE_SAMPLES; otherwise it keeps warming until it does.
  - Resets whenever the real baseline's first-sample time changes, i.e. on
    every recalibration.

State: logs/shadow_mfcc_state.json   Output: logs/shadow_mfcc.csv
"""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from drift_detector import BaselineProfile, DriftDetector, threshold_from_baseline
from firmware_mfcc import firmware_mfcc

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "logs" / "shadow_mfcc_state.json"
CSV_PATH = ROOT / "logs" / "shadow_mfcc.csv"

# Opt-in per device, like SHAPE_GATE_MACS: this is an experiment, run where a
# controlled filter change is planned.
SHADOW_MACS = {
    "58E6C5204B28",   # P3 -- FPR12 recalibration 2026-10-27, blockage test to follow
}

COEFFS = list(range(1, 13))
# threshold_from_baseline only uses its empirical path from max(20, 4*d)
# samples; below that the closed form would be the only estimate, which is
# not what the real detector uses either.
MIN_FREEZE_SAMPLES = 4 * len(COEFFS)

CSV_HEADER = (["recorded_at", "device_mac",
               "main_decision", "main_distance", "main_ewma", "main_threshold",
               "shadow_state", "shadow_distance", "shadow_ewma", "shadow_threshold",
               "shadow_decision", "shadow_rule"]
              + [f"mfcc{i}" for i in range(13)])


def _load():
    try:
        return json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.part")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, STATE_PATH)       # never leave a half-written state file


def _fresh(anchor):
    return {"anchor": anchor, "state": "warming", "baseline": None,
            "threshold": None, "detector": None, "recent_ewmas": []}


def _write_row(row):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(CSV_HEADER)
        w.writerow(row)


def score(device_mac, y, ts, main_state, main_result, *, ewma_alpha, patience,
          window_readings, window_min, dirty_fraction, dirty_min):
    """Score one blower-on clip. Returns a one-line note for the poller log, or
    None when this device is not part of the experiment. Raises nothing the
    caller has to handle beyond a broad except: it must never cost a reading."""
    if device_mac not in SHADOW_MACS:
        return None

    mfcc = firmware_mfcc(np.clip(np.asarray(y) * 32768.0, -32768, 32767).astype(np.int16))
    x = mfcc[COEFFS]

    all_state = _load()
    main_baseline = main_state.get("baseline") or {}
    anchor = main_baseline.get("first_ts")
    main_warming = main_state.get("state") in ("cold_start", "warming")
    s = all_state.get(device_mac)

    if s is None or s.get("anchor") != anchor:
        if not main_warming:
            all_state[device_mac] = {"anchor": anchor, "state": "waiting"}
            _save(all_state)
            return "shadow MFCC: waiting for this device's next recalibration"
        s = _fresh(anchor)
    if s["state"] == "waiting":
        return "shadow MFCC: waiting for this device's next recalibration"

    n_feat = len(COEFFS)
    bp = BaselineProfile.from_state(n_feat, 1, s["baseline"]) if s["baseline"] \
        else BaselineProfile(n_feat, min_samples=1)

    dist = ewma = decision = rule = None
    if s["state"] == "warming":
        bp.update(x, ts=ts)
        warmup = s.setdefault("warmup", [])
        warmup.append([float(v) for v in x])
        decision = "calibrating"
        if main_result.get("state") == "warm" and bp.n >= MIN_FREEZE_SAMPLES:
            bp.freeze()
            thr, _ = threshold_from_baseline(bp, 0.99, 1.10, warmup_samples=warmup)
            s.update(state="warm", threshold=thr, detector=None, recent_ewmas=[])
            s.pop("warmup", None)
            decision = "frozen"
    else:
        det = DriftDetector(bp, s["threshold"], alpha=ewma_alpha, patience=patience)
        if s["detector"]:
            det.set_state(s["detector"])
        prior = s["recent_ewmas"][-(window_readings - 1):]
        window = len(prior) + 1
        recent = limit = None
        if window >= window_min:
            recent = sum(1 for e in prior if e > s["threshold"])
            limit = max(dirty_min, int(np.ceil(dirty_fraction * window)))
        r = det.update(x, recent_exceedances=recent, recent_limit=limit)
        dist, ewma, rule = r["distance"], r["ewma"], r["drift_rule"]
        decision = "dirty" if r["drift_dirty"] else "clean"
        s["detector"] = det.get_state()
        s["recent_ewmas"] = (prior + [ewma])[-window_readings:]

    s["baseline"] = bp.get_state()
    all_state[device_mac] = s
    _save(all_state)

    _write_row([datetime.fromtimestamp(ts, timezone.utc).isoformat(), device_mac,
                main_result.get("decision"), main_result.get("distance"),
                main_result.get("ewma"), main_state.get("drift_threshold"),
                s["state"], dist, ewma, s["threshold"], decision, rule]
               + [float(v) for v in mfcc])

    if s["state"] == "warm" and dist is not None:
        return (f"shadow MFCC: {decision} (dist {dist:.2f}, ewma {ewma:.2f} vs "
                f"{s['threshold']:.2f}{f', {rule}' if rule else ''})")
    if decision == "frozen":
        return f"shadow MFCC: baseline frozen from {bp.n} clips, threshold {s['threshold']:.2f}"
    return f"shadow MFCC: calibrating ({bp.n} clips)"
