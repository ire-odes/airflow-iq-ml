"""
service/poll_and_infer.py
Polls the AirFlow IQ Supabase project for new audio uploads, runs the
existing ML pipeline (classifier + per-device drift baseline) against
each one, and writes the result back.

Reuses, unmodified: extract_features_v2.extract_features, the classifier-
loading pattern from ../predict.py (model_bundle.pkl), and
drift_detector.FilterHealthMonitor -- state now persisted via its
get_state()/from_state() round-trip into the device_baselines table
instead of a local baseline_<house>.pkl file, since this runs against N
independent devices, not one local house.

Per run:
  1. Read every audio_logs row (one per device). LoRaWAN rows carry
     mfcc_coefficients instead of a wav (is_lora=true) -- not handled here
     yet (none exist in production as of the schema comments), skipped
     with a log line.
  2. Skip rows already processed (updated_at <= that device's
     last_processed_at) or with no storage_path.
  3. Skip readings where the HVAC probably wasn't running. The audio
     clip's own RMS is the primary, authoritative signal (extract_features_
     v2's own near-silence guard) -- not every device has blower access
     for a windSpeed reading, so loudness is the one signal every device
     actually has. sensor_logs' windSpeed, when available, is logged
     alongside the decision as corroboration, but never independently
     blocks processing on its own (a device with no/stale windSpeed data
     must not get treated differently from one that has it). Neither
     produces a filter_ml_readings row; both still update
     last_processed_at so the same reading isn't rechecked forever.
  4. Otherwise: run the clip through extract_features_v2, the classifier,
     then FilterHealthMonitor. Its decision is "calibrating" (not a raw
     "clean"/"dirty" guess) until that device reaches WARM -- the
     classifier alone is demonstrably unreliable on a brand-new
     environment (see AbrahamHouse), so cold-start/warming never
     surfaces its raw opinion as a verdict.
  5. Also accumulate a baseline frequency spectrum (spectrum_viz.py, a
     Python port of the app's client-side computeSpectrum()) using the
     *same* gate and freeze point as the 5-feature drift baseline, so
     "baseline" means the same thing everywhere -- this is what lets the
     Acoustic Data spectrum chart eventually overlay "now" against
     "normal" instead of just showing the current reading alone.
  6. Persist: a filter_ml_readings row (history), the updated
     device_baselines row (state for next run, including the
     classifier-override patience counter and the spectrum baseline),
     and latest_verdict / latest_confidence back onto audio_logs.

Usage:
  python service/poll_and_infer.py             # one pass, exits
  python service/poll_and_infer.py --loop 900  # repeat every N seconds
                                                # (Task Scheduler running
                                                # this with no --loop is
                                                # the recommended setup)

Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in service/.env (copy
from service/.env.example) -- the service role key bypasses RLS, same as
the existing device-ingestion pipeline. Never expose this key to the app
or commit it.
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
from dateutil.parser import isoparse
from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import librosa

from extract_features_v2 import extract_features, SR
from drift_detector import (FilterHealthMonitor, DEFAULT_DRIFT_FEATURES,
                            threshold_from_baseline)
from spectrum_viz import compute_spectrum

load_dotenv(Path(__file__).resolve().parent / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
BUCKET = "hvac-recordings"

BUNDLE_PATH = ROOT / "model_bundle.pkl"
POOLED_CALIBRATION_PATH = ROOT / "pooled_calibration.json"

N_DRIFT_FEATURES = len(DEFAULT_DRIFT_FEATURES)
MIN_BASELINE_SAMPLES = 120
EWMA_ALPHA = 0.15
PATIENCE = 3

# Minimum wall-clock window the warmup samples must cover before the
# baseline may freeze, on top of MIN_BASELINE_SAMPLES. 20h is what the
# original design got implicitly (120 samples at the 600s default wake =
# 20h), which spans a full day/night cycle so diurnal variation lands
# inside the covariance rather than reading as drift afterwards.
# Sampling mode's 60s wakes reach 120 samples in ~2h and broke that
# assumption -- see 20260829000000_baseline_warmup_span.sql for the
# observed false positives. Devices keep accumulating samples during the
# wait, so the extra time makes the baseline richer, not just later.
MIN_BASELINE_SPAN_SECONDS = 20 * 3600

# Per-device, temporary overrides of that span. This is a test concession, not
# a setting -- every entry here weakens the day/night coverage the 20h default
# exists to guarantee, so each one carries the device, the date and the reason,
# and is meant to be deleted once its experiment is over.
#
# P3 (2026-09-17): flashed with the burst-mode firmware and recalibrated today;
# 16h lets its baseline close tomorrow morning so blower-burst testing can
# start a day earlier. At the 60s calibration interval it banks its 120 samples
# within the first couple of hours, so the span is the only binding constraint
# and shortening it is what actually moves the finish line. REVERT AFTER.
BASELINE_SPAN_OVERRIDE_SECONDS = {
    "58E6C5204B28": 16 * 3600,      # P3 -- burst-mode test, remove when done
}


def baseline_span_seconds(device_mac):
    """Warmup span this device must cover before its baseline may freeze."""
    return BASELINE_SPAN_OVERRIDE_SECONDS.get(device_mac, MIN_BASELINE_SPAN_SECONDS)

# Second alert rule, independent of the PATIENCE consecutive-reading rule:
# how many of a device's last N SCORED readings exceeded its threshold. The
# consecutive rule catches step changes (a filter swap); this one catches
# gradual clogging, where the EWMA crosses repeatedly but falls back before
# reaching three in a row.
#
# WINDOWED BY READING COUNT, NOT BY WALL CLOCK. An HVAC is not always
# running, and clips recorded while it is off are discarded as silence before
# they ever produce a reading -- so a fixed time window holds a wildly
# variable number of samples. The previous "N per 24h" form was unreachable
# on a slow sampler (a 4h wake interval yields 6 readings a day) and
# inconsistent on a fast one, behaving differently on a day the blower ran
# 2 hours versus 12. A fixed count of readings holds the statistical power
# constant however the duty cycle moves, and idle periods simply do not
# enter the window instead of quietly weakening it.
RECENT_WINDOW_READINGS = 30

# 6 of 30. Sized against measured behaviour rather than picked: rolling
# 30-reading windows over the real FPR12 swap gave 0% dirty on all 356 of
# P10's clean windows and a 3% worst case anywhere in the fleet (one reading
# in thirty, on P7), while P10's dirty windows reached 100%. 20% therefore
# sits ~6x above the noisiest clean behaviour on record and far below a
# genuine detection.
RECENT_DIRTY_FRACTION = 0.20

# Floor on the count, so a small window cannot fire off a single outlier.
RECENT_DIRTY_MIN = 3

# Do not judge a window shorter than this at all. Immediately after a
# baseline freezes only a handful of readings have been scored, and 20% of 5
# is 1 -- one noisy reading would condemn a fresh filter.
RECENT_WINDOW_MIN = 12

# Discard readings older than this even when the window is not yet full. A
# device whose blower barely runs could otherwise assemble 30 readings
# spanning weeks, straddling a filter change or a change of season, and
# compare conditions with nothing to do with each other.
RECENT_WINDOW_MAX_AGE_HOURS = 168   # 7 days

# ---------------------------------------------------------------------------
# Spectral silence gate: decide "was the blower running?" from the spectrum's
# LEVEL rather than the clip's loudness.
#
# The RMS gate in extract_features_v2 has never rejected a single clip, because
# recorded level barely moves with blower state -- the hardware normalises it.
# Measured across the labelled corpora: IdleHVAC (blower definitively off)
# sits at RMS 0.333, the running rig at 0.340. Amplitude carries no signal.
#
# The spectrum's dB level does. Per-clip median across 120 labelled clips:
#
#     IdleHVAC (off)      min -6.6   p50 -0.6   max  1.9
#     rig running (on)    min  3.2   p50 10.3   max 12.8
#
# No overlap. The cut below sits mid-gap, and on that corpus it keeps every
# fan-on clip and rejects every fan-off one. Live clips agree: P10 -40.1 dB
# and P12 -8.5 dB (idle) against P3 +5.7 and P1 +5.4 (running).
#
# FLEET-WIDE as of 2026-09-17. It ran on P6 alone for a day first, and the
# evidence that came back is far stronger than the corpus above:
#
#   P6   23 clips, blower on    median +5.60 dB   [ +3.75 ..  +7.49]
#   P3   65 clips, blower off   median -32.19 dB  [-34.82 .. -26.56]
#
# A 30 dB gap with nothing in it, versus the 1.3 dB gap the threshold was
# originally set from. Independent confirmation from the FiltSure corpus:
# all 4,655 FPR 5/7/9/12 clips pass (median +6.2 .. +7.8 dB) and every
# ire_house clip fails (-37.6 dB mean) -- no tuning, same 2.5 dB cut.
#
# What forced the rollout: P3 had no gate, so 65 of the 67 warmup samples in
# its post-recalibration baseline were silence. A baseline fitted to a quiet
# room reports DIRTY the moment real air moves past it. The gate is not an
# optimisation -- ungated calibration in a quiet building produces a broken
# detector.
#
# THE COST, stated honestly: a rejected clip never reaches the baseline, so
# on a site whose blower rarely runs the gate starves calibration rather than
# poisoning it. P6 sat at 23/120 samples for 13 hours because its blower was
# genuinely off. Starved is recoverable; poisoned is not.
SPECTRAL_GATE_ENABLED = True
SPECTRAL_GATE_EXCLUDE_MACS = set()   # opt a device out here if it ever needs it
SPECTRAL_GATE_MIN_MEDIAN_DB = 2.5


def spectral_gate_applies(device_mac):
    """True when this device's clips must clear the silence gate."""
    return SPECTRAL_GATE_ENABLED and device_mac not in SPECTRAL_GATE_EXCLUDE_MACS


# ---------------------------------------------------------------------------
# Blower burst: sample once a minute while air is actually moving.
#
# The gate above already answers "was the blower running?" on every clip. This
# writes that answer back to devices.blower_burst_until, which
# effective_wake_seconds() turns into a 60s wake interval -- so a device
# collects densely while its blower runs and sleeps through the silence, rather
# than spending a fixed budget uniformly across a mostly-quiet day.
#
# WHY A DEADLINE RATHER THAN A FLAG. If this poller stops -- the desktop slept
# on 16 Sep and took it down for ten hours -- a flag would pin the device at
# 60s wakes until someone noticed. A timestamp lapses on its own, so a dead
# poller degrades to the normal interval instead of draining the battery.
#
# The window is deliberately longer than one wake cycle: at 60s wakes, three
# minutes tolerates two consecutive missed or failed uploads before the device
# falls back, which matters because a burst that collapses on one dropped
# packet would thrash between 60s and the slow interval.
BLOWER_BURST_MACS = {
    "58E6C5204B28",      # P3
}
BLOWER_BURST_MINUTES = 3


def set_blower_burst(sb, device_mac, blower_on):
    """Extend or clear this device's 60s burst window. Never raises: a
    failure here must not cost the reading that has already been computed."""
    if device_mac not in BLOWER_BURST_MACS:
        return
    try:
        until = (datetime.now(timezone.utc)
                 + timedelta(minutes=BLOWER_BURST_MINUTES)).isoformat() if blower_on else None
        sb.table("devices").update({"blower_burst_until": until})             .eq("device_mac", device_mac).execute()
        print(f"  {device_mac}: blower burst "
              f"{'extended to ' + until if blower_on else 'cleared -- back to normal interval'}")
    except Exception as e:
        print(f"  {device_mac}: WARNING could not update blower burst: {e}")

# Each device's alert threshold is derived from its OWN baseline the moment
# that baseline freezes, rather than every unit inheriting one fleet-wide
# default. A Mahalanobis cut needs no post-calibration clean readings -- it
# follows from the covariance itself -- so there is nothing to wait for, and
# nothing that has to be re-run later by hand.
#
# The old arrangement was one shared number (3.515) that in practice never
# changed, because the tool for revising it read post-freeze clean history
# that a freshly calibrated device does not have yet. It cannot serve a fleet
# whose baseline spreads differ threefold: on 2026-09-03 a real FPR12 swap
# sat at the 94th percentile of P3's own baseline, while P6's CLEAN readings
# sat at the 100th percentile of its own.
#
# p99 with a 10% margin: the same false-positive posture the project has used
# throughout (a missed dirty filter costs more than a nuisance alert), and the
# margin absorbs the fact that a covariance estimated from a few hundred
# correlated samples is itself uncertain.
AUTO_THRESHOLD_QUANTILE = 0.99
AUTO_THRESHOLD_MARGIN = 1.10

# ---------------------------------------------------------------------------
# Adaptive threshold: keep re-deriving it from the readings that keep arriving,
# instead of fixing it at the moment calibration ends.
#
# WHY IT IS WANTED. A baseline is fitted over ~20h and then never moves again.
# Over a filter's real 30-90 day life the acoustic environment does move --
# ambient temperature, humidity, seasonal duct behaviour, the blower itself
# ageing. A threshold pinned to one 20h window in September is being asked to
# still mean the same thing in December, and it does not.
#
# WHY IT IS DANGEROUS, AND WHAT BOUNDS IT. The signal this system exists to
# detect is a slow upward drift. A threshold that follows the readings will
# follow that drift too, and quietly erase the thing it is watching for --
# textbook adaptive-baseline failure. Every guard below exists to stop that:
#
#   1. CEILING. The threshold may never exceed the calibration-time value by
#      more than ADAPTIVE_MAX_RISE. Environmental drift gets absorbed; a
#      clogging filter cannot walk the threshold up indefinitely to stay under
#      it. This is the guard that actually makes adaptation survivable.
#   2. NEVER ADAPT WHILE DIRTY. Once a device is flagged, its threshold stops
#      moving. Without this, a dirty verdict deletes itself within a day.
#   3. LEARN ONLY FROM SUB-THRESHOLD READINGS. An exceedance is the thing
#      being detected; it must not also be evidence of what "normal" means.
#   4. DOWNWARD MOVES ARE FREE. Tightening can only make the detector more
#      sensitive, so it needs no ceiling -- only the floor below.
ADAPTIVE_THRESHOLD = True

# How many post-freeze readings must exist before adapting at all. Below this
# the percentile is being read off too few points to mean anything.
ADAPTIVE_MIN_READINGS = 50

# What "recent" means. Long enough to span a full weekly cycle at a 10-minute
# wake (~3.5 days), short enough to actually track a season.
ADAPTIVE_WINDOW_READINGS = 500

# Same posture as the freeze-time derivation, so switching between them is not
# also a change of strictness.
ADAPTIVE_PERCENTILE = 99
ADAPTIVE_MARGIN = 1.10

# Re-derive on a time cadence rather than every reading: it costs two queries,
# and a threshold that jitters per-reading is unreadable in history. Keyed off
# the reading's own hour bucket, so it needs no persisted counter and behaves
# identically however often the poller restarts. Readings inside the same
# bucket recompute the same value and are suppressed by the no-op guard, so
# only the first one in each window actually writes.
ADAPTIVE_UPDATE_HOURS = 6

# The ceiling, as a multiple of the calibration-time threshold. 1.25 comfortably
# covers seasonal drift -- across this fleet the spread between the tightest and
# widest baseline is 3.6x, so a quarter is generous for one device over time --
# while capping how far a clogging filter can push its own goalposts.
ADAPTIVE_MAX_RISE = 1.25

# Floor, as a multiple of the same anchor. Stops a spell of unusually quiet
# readings from ratcheting the threshold down to where ordinary noise flags.
ADAPTIVE_MIN_FALL = 0.70

# Below this, sensor_logs.windSpeed (m/s) is logged as "~still" for
# corroboration -- NOT used to gate processing on its own (see
# check_wind_speed's docstring: not every device has blower access for
# this reading, so extract_features_v2.SILENCE_RMS_THRESHOLD is the
# actual "HVAC off" authority, universally available). metrics.js's own
# normalRange for windSpeed while running is [0.5, 5], so anything well
# below that floor is a reasonable "probably off" signal. Uncalibrated
# placeholder, same caveat as SILENCE_RMS_THRESHOLD: tune against real
# data once you have known-off readings.
WIND_SPEED_OFF_THRESHOLD = 0.1
WIND_SPEED_LOOKUP_WINDOW_MIN = 30


def load_classifier():
    bundle = joblib.load(BUNDLE_PATH)
    clf = bundle["classifier"]
    le = bundle["label_encoder"]
    feature_cols = bundle["selected_features"]
    dirty_idx = int(np.where(le.classes_ == "FPR12")[0][0])

    def classifier_fn(feats):
        missing = [k for k in feature_cols if k not in feats]
        if missing:
            raise ValueError(
                f"Extracted features are missing {len(missing)} column(s) "
                f"the model expects (e.g. {missing[:3]}) -- the extractor "
                f"is out of sync with the trained model.")
        x = np.array([[feats[k] for k in feature_cols]])
        p_dirty = float(clf.predict_proba(x)[0][dirty_idx])
        return ("clogged" if p_dirty >= 0.5 else "clean"), p_dirty

    return classifier_fn


def default_threshold():
    """Same fallback build_baseline.py uses for a brand-new device: the
    pooled calibration if one exists, otherwise the theoretical guess."""
    if POOLED_CALIBRATION_PATH.exists():
        with open(POOLED_CALIBRATION_PATH) as fh:
            pooled = json.load(fh)
        if pooled["drift_features"] == DEFAULT_DRIFT_FEATURES:
            return pooled["threshold"]
    return float(np.sqrt(np.percentile(np.random.chisquare(N_DRIFT_FEATURES, 200000), 99)))


# MAC -> devices.id, resolved once per process. The mapping is fixed for a
# device's lifetime, and looking it up per reading cost one round trip on
# every wake of every device for a value that never changes.
_DEVICE_ID_CACHE = {}


def _device_id(sb, device_mac):
    if device_mac not in _DEVICE_ID_CACHE:
        dev = (sb.table("devices").select("id")
               .eq("device_mac", device_mac).maybe_single().execute())
        _DEVICE_ID_CACHE[device_mac] = dev.data["id"] if dev and dev.data else None
    return _DEVICE_ID_CACHE[device_mac]


def check_wind_speed(sb, device_mac, near_time):
    """Best-effort lookup of sensor_logs.windSpeed near the audio upload's
    timestamp -- corroborating context only, logged alongside the RMS-based
    decision, never an independent gate. Not every device has blower
    access for a windSpeed reading (some installs can't reach it at all),
    so it must never be treated as the main "is the HVAC running" signal
    -- the audio clip's own loudness is the one signal every device
    actually has (see extract_features_v2.SILENCE_RMS_THRESHOLD).
    sensor_logs.device_id is devices.id (uuid), not the MAC audio_logs
    keys off, hence the extra lookup.

    Returns (windspeed_or_None, likely_off_or_None). Both None means no
    nearby reading was found -- genuinely unknown, not "on" or "off"."""
    device_id = _device_id(sb, device_mac)
    if device_id is None:
        return None, None

    near = isoparse(near_time)
    window = timedelta(minutes=WIND_SPEED_LOOKUP_WINDOW_MIN)
    lo, hi = (near - window).isoformat(), (near + window).isoformat()

    resp = sb.table("sensor_logs").select("recorded_at, windSpeed") \
        .eq("device_id", device_id).gte("recorded_at", lo).lte("recorded_at", hi) \
        .not_.is_("windSpeed", "null").execute()
    rows = resp.data or []
    if not rows:
        return None, None

    closest = min(rows, key=lambda r: abs(isoparse(r["recorded_at"]) - near))
    ws = float(closest["windSpeed"])
    return ws, ws < WIND_SPEED_OFF_THRESHOLD


def _reshape(flat):
    return np.array(flat, dtype=float).reshape(N_DRIFT_FEATURES, N_DRIFT_FEATURES).tolist()


def _flatten(matrix):
    return [v for row in matrix for v in row]


def update_spectrum_baseline(mean, n, new_spectrum):
    """Welford-style running mean over the 72-bin dB spectrum -- same idea
    as BaselineProfile's mean tracking, without the covariance part, since
    this is purely for the UI chart, not a Mahalanobis distance."""
    new_spectrum = np.array(new_spectrum, dtype=float)
    if mean is None:
        return new_spectrum, 1
    n = n + 1
    mean = mean + (new_spectrum - mean) / n
    return mean, n


def adaptive_threshold(sb, device_mac, mon, mon_state, updated_at):
    """Re-derive this device's threshold from the readings it has produced
    since its baseline froze. Returns None to leave it alone.

    This is what recalibrate_threshold.py does by hand, run continuously. It
    tracks slow environmental change so a threshold fitted in one 20h window
    still means the same thing months later.

    The guards are the whole design -- see the ADAPTIVE_* constants. Chief
    among them: the result is clamped against the CALIBRATION-time threshold,
    which is recomputed from the frozen baseline rather than read from the
    stored value. Anchoring to the stored value would let each update become
    the anchor for the next one, and the ceiling would ratchet upward one step
    at a time -- exactly the runaway it exists to prevent.
    """
    thr_now = mon_state.get("drift_threshold")
    baseline_frozen_at = (mon_state.get("baseline") or {}).get("last_ts")
    if thr_now is None or not baseline_frozen_at:
        return None

    # Guard 2: a flagged device's threshold is frozen. Otherwise the readings
    # that raised the flag become the evidence that dissolves it.
    last = (sb.table("filter_ml_readings").select("decision")
            .eq("device_mac", device_mac)
            .order("recorded_at", desc=True).limit(1).execute().data or [])
    if last and last[0].get("decision") == "dirty":
        return None

    hist = (sb.table("filter_ml_readings").select("ewma_value")
            .eq("device_mac", device_mac)
            .gte("recorded_at", _epoch_iso(baseline_frozen_at))
            .not_.is_("ewma_value", "null")
            .order("recorded_at", desc=True)
            .limit(ADAPTIVE_WINDOW_READINGS).execute().data or [])
    ewmas = [r["ewma_value"] for r in hist]
    if len(ewmas) < ADAPTIVE_MIN_READINGS:
        return None

    # Guard 3: exceedances are the thing being detected, not evidence of what
    # normal looks like.
    clean = [e for e in ewmas if e <= thr_now]
    if len(clean) < ADAPTIVE_MIN_READINGS:
        return None

    proposed = float(np.percentile(clean, ADAPTIVE_PERCENTILE)) * ADAPTIVE_MARGIN

    # Guard 1: clamp against the calibration-time value, RECOMPUTED from the
    # baseline rather than read from device_baselines.threshold -- that column
    # is what adaptation overwrites, so using it would make each update the
    # anchor for the next and let the ceiling ratchet upward a step at a time,
    # which is the runaway it exists to prevent.
    #
    # The warmup vectors are passed so this reproduces the SAME route freeze
    # used. Without them it falls to the closed form, and on a device whose
    # calibration was empirical the two disagree -- a baseline that measured
    # 2.9 empirically would be handed a 3.94 anchor and a ceiling of 4.9,
    # licensing a rise the calibration never justified. Costs one query per
    # device per ADAPTIVE_UPDATE_HOURS.
    anchor, _ = threshold_from_baseline(
        mon.baseline, quantile=AUTO_THRESHOLD_QUANTILE,
        margin=AUTO_THRESHOLD_MARGIN,
        warmup_samples=fetch_warmup_features(sb, device_mac, mon_state))
    if anchor is None:
        anchor = thr_now
    lo, hi = anchor * ADAPTIVE_MIN_FALL, anchor * ADAPTIVE_MAX_RISE
    clamped = min(max(proposed, lo), hi)

    # Ignore trivial moves: rewriting the column every cycle for a 0.1% change
    # makes the history unreadable for no behavioural difference.
    if abs(clamped - thr_now) < 0.02:
        return None
    return clamped, proposed, anchor, len(clean)


def fetch_warmup_features(sb, device_mac, mon_state):
    """This calibration's stored feature vectors, oldest first.

    Only rows recorded at or after the current warmup began: a device that has
    been recalibrated has older rows from previous baselines sitting in the
    same table, and mixing those in would fit the threshold to audio from a
    different filter entirely. baseline_first_sample_at is the boundary.

    Returns None when the vectors are not there -- every baseline frozen
    before the features column existed (migration 20260903000000) is in that
    position, and the caller falls back to the closed-form threshold rather
    than to a bad empirical one.
    """
    first_ts = (mon_state.get("baseline") or {}).get("first_ts")
    if first_ts is None:
        return None
    q = (sb.table("filter_ml_readings").select("features")
         .eq("device_mac", device_mac)
         .gte("recorded_at", _epoch_iso(first_ts))
         .not_.is_("features", "null")
         .order("recorded_at", desc=False)
         # A warmup is bounded by MIN_BASELINE_SAMPLES and the 20h span gate,
         # which in practice lands at 300-450 vectors (measured 2026-09-04:
         # P1 416, P12 382, P3 374, P6 310 -- about 25 kB each). 1000 is
         # headroom over the worst case a 60s sampling-mode warmup could
         # produce, not a real expectation; the old 5000 was an arbitrary
         # number five times larger than anything reachable.
         .limit(1000).execute())
    rows = [r["features"] for r in (q.data or [])
            if r.get("features") and len(r["features"]) == N_DRIFT_FEATURES]
    return rows or None


def _ts_epoch(value):
    """timestamptz string from Supabase -> epoch seconds, or None."""
    if not value:
        return None
    return isoparse(value).timestamp()


def _epoch_dt(value):
    """Epoch seconds -> aware UTC datetime, or None. Aware so it can be
    compared against parsed timestamptz values without a naive/aware clash."""
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _epoch_iso(value):
    """Epoch seconds -> ISO string for a timestamptz column, or None."""
    dt = _epoch_dt(value)
    return dt.isoformat() if dt else None


def baseline_row_exists(sb, device_mac):
    res = sb.table("device_baselines").select("device_mac").eq("device_mac", device_mac).execute()
    return bool(res.data)


def load_monitor(sb, device_mac, classifier_fn, row=None):
    """Returns (monitor, spectrum_mean, spectrum_n, had_row).

    had_row records whether a baseline row existed when this cycle started.
    save_monitor() uses it to detect a recalibration that landed mid-cycle
    -- see the note there.

    row: the device_baselines row, when the caller has already read it.
    run_once reads it anyway to check last_processed_at, and re-reading the
    same row here was a second round trip per reading for identical data --
    the two happen microseconds apart, so there is nothing to be gained by
    reading it twice."""
    if row is None:
        res = (sb.table("device_baselines").select("*")
               .eq("device_mac", device_mac).maybe_single().execute())
        row = res.data if res else None
    had_row = row is not None

    spectrum_mean = None
    spectrum_n = 0
    if row and row.get("baseline_spectrum_db") is not None:
        spectrum_mean = np.array(row["baseline_spectrum_db"], dtype=float)
        spectrum_n = row.get("spectrum_n", 0) or 0

    if not row or row.get("baseline_mean") is None:
        mon = FilterHealthMonitor(classifier_fn, min_baseline_samples=MIN_BASELINE_SAMPLES,
                                  drift_threshold=default_threshold(),
                                  ewma_alpha=EWMA_ALPHA, patience=PATIENCE,
                                  min_baseline_span_seconds=baseline_span_seconds(device_mac),
                                  recent_exceedance_limit=None,
                                  auto_threshold_quantile=AUTO_THRESHOLD_QUANTILE,
                                  auto_threshold_margin=AUTO_THRESHOLD_MARGIN)
        return mon, spectrum_mean, spectrum_n, had_row

    state = {
        "state": row["state"],
        "drift_threshold": row["threshold"],
        "baseline": {
            "n": row["n_warmup_samples"],
            "mean": row["baseline_mean"],
            "m2": _reshape(row["baseline_m2"]) if row.get("baseline_m2") else
                  [[0.0] * N_DRIFT_FEATURES for _ in range(N_DRIFT_FEATURES)],
            "frozen": row["state"] == "warm",
            "cov_inv": _reshape(row["baseline_cov_inv"]) if row.get("baseline_cov_inv") else None,
            # Stored as timestamptz for readability in the dashboard, carried
            # as epoch seconds in BaselineProfile so span maths stays trivial.
            "first_ts": _ts_epoch(row.get("baseline_first_sample_at")),
            "last_ts": _ts_epoch(row.get("baseline_last_sample_at")),
        },
    }
    if row["state"] == "warm":
        state["detector"] = {
            "ewma": row["ewma"] or 0.0,
            "exceed_count": row["exceedance_count"] or 0,
            "initialized": True,
        }
    state["override_streak"] = row.get("override_streak", 0) or 0
    mon = FilterHealthMonitor.from_state(
        classifier_fn, state, min_baseline_samples=MIN_BASELINE_SAMPLES,
        ewma_alpha=EWMA_ALPHA, patience=PATIENCE,
        min_baseline_span_seconds=baseline_span_seconds(device_mac),
        recent_exceedance_limit=None,
        auto_threshold_quantile=AUTO_THRESHOLD_QUANTILE,
        auto_threshold_margin=AUTO_THRESHOLD_MARGIN)
    return mon, spectrum_mean, spectrum_n, had_row


def save_monitor(sb, device_mac, mon, processed_at, spectrum_mean=None, spectrum_n=0,
                 had_row=True):
    # Recalibration race: the app's "Recalibrate" action (Devices.jsx) clears
    # a device's device_baselines row to force a fresh cold start. This
    # function runs at the end of a processing cycle and upserts whatever
    # state was loaded at the *start* of it -- so a delete landing in between
    # gets silently undone, restoring the exact baseline the user just asked
    # to throw away. Observed on two real devices: they were stamped for
    # recalibration but came back warm with n=120 and a 65.93 EWMA intact.
    #
    # If a row existed when this cycle began and is gone now, that delete is
    # the newer intent -- drop this cycle's state rather than resurrecting
    # it. The next cycle starts clean. Only costs a lookup on the path that
    # was about to write anyway, and it does not close the window entirely
    # (a delete arriving after this check still loses), but it removes the
    # wide part of it.
    if had_row and not baseline_row_exists(sb, device_mac):
        print(f"  {device_mac}: baseline cleared mid-cycle (recalibrated) -- "
              f"discarding this cycle's state")
        return

    state = mon.get_state()
    b = state["baseline"]
    detector_state = state.get("detector", {})
    payload = {
        "device_mac": device_mac,
        "baseline_mean": b["mean"],
        "baseline_m2": _flatten(b["m2"]),
        "baseline_cov_inv": _flatten(b["cov_inv"]) if b.get("cov_inv") else None,
        "threshold": state["drift_threshold"],
        "ewma": detector_state.get("ewma"),
        "exceedance_count": detector_state.get("exceed_count", 0),
        "state": state["state"],
        "n_warmup_samples": b["n"],
        "baseline_first_sample_at": _epoch_iso(b.get("first_ts")),
        "baseline_last_sample_at": _epoch_iso(b.get("last_ts")),
        "override_streak": state.get("override_streak", 0),
        "min_baseline_samples": MIN_BASELINE_SAMPLES,
        "min_baseline_span_seconds": baseline_span_seconds(device_mac),
        "baseline_spectrum_db": spectrum_mean.tolist() if spectrum_mean is not None else None,
        "spectrum_n": spectrum_n,
        "last_processed_at": processed_at,
    }
    sb.table("device_baselines").upsert(payload, on_conflict="device_mac").execute()


def process_device(sb, device_mac, storage_path, updated_at, classifier_fn,
                   baseline_row=None):
    mon, spectrum_mean, spectrum_n, had_row = load_monitor(
        sb, device_mac, classifier_fn, row=baseline_row)

    # windSpeed is corroborating context only now -- not every device has
    # blower access for this reading, so it must never independently gate
    # processing (a windSpeed-less device isn't "always on"). Fetched
    # once up front purely so both skip/keep paths below can log it.
    wind_speed, wind_likely_off = check_wind_speed(sb, device_mac, updated_at)
    wind_note = (f", windSpeed={wind_speed:.2f}m/s{' (~still)' if wind_likely_off else ''}"
                if wind_speed is not None else "")

    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        wav_bytes = sb.storage.from_(BUCKET).download(storage_path)
        tmp_path.write_bytes(wav_bytes)
        feats = extract_features(str(tmp_path))
        # Loaded again (not reused from extract_features' own internal
        # load) since that function only returns derived features, not
        # the raw samples -- cheap relative to the STFT/DWT work above.
        y = librosa.load(str(tmp_path), sr=SR, mono=True)[0] if feats is not None else None
    finally:
        tmp_path.unlink(missing_ok=True)

    if feats is None:
        # The authoritative "HVAC off" signal: extract_features_v2's own
        # RMS-based near-silence guard, the one check every device has
        # regardless of blower access. Never feed this into baseline
        # warmup -- mixing "off" and "on" acoustics into one baseline
        # would corrupt the learned covariance, not just produce one bad
        # reading. Same reasoning applies to the spectrum baseline, so
        # it's skipped here too.
        print(f"  {device_mac}: clip too short/silent -- marking processed, no verdict{wind_note}")
        save_monitor(sb, device_mac, mon, updated_at, spectrum_mean, spectrum_n, had_row)
        return

    # Blower-off by spectral level (see SPECTRAL_GATE_ENABLED). Treated exactly
    # like the RMS guard above: no reading row, no baseline sample, but still
    # marked processed so the clip isn't re-examined forever. The median is
    # used rather than the peak because a single loud band -- a door, a voice,
    # a knock on the duct -- lifts the peak without the broadband hiss that
    # actually means air is moving.
    if spectral_gate_applies(device_mac):
        gain_db = float(np.median(compute_spectrum(y, SR)))
        if gain_db <= SPECTRAL_GATE_MIN_MEDIAN_DB:
            print(f"  {device_mac}: blower off by spectral gain "
                  f"({gain_db:+.1f} dB <= {SPECTRAL_GATE_MIN_MEDIAN_DB} dB) "
                  f"-- marking processed, no verdict{wind_note}")
            # Clear BEFORE save_monitor: save_monitor stamps last_processed_at,
            # which is the flag the firmware polls to learn the verdict is in.
            # Writing the burst state first guarantees the device never reads a
            # fresh last_processed_at alongside a stale interval.
            set_blower_burst(sb, device_mac, False)
            save_monitor(sb, device_mac, mon, updated_at, spectrum_mean, spectrum_n, had_row)
            return
        print(f"  {device_mac}: blower on by spectral gain ({gain_db:+.1f} dB)")
        set_blower_burst(sb, device_mac, True)

    # The audio's own upload time, not now(): warmup span must reflect when
    # the sound was recorded, otherwise a backlog processed in one burst
    # would look like it covered a wide window when it didn't.
    # Counted from filter_ml_readings rather than carried in device_baselines:
    # the readings are already stored, so the window survives poller restarts
    # for free and there is no extra state to keep consistent. Threshold is
    # read from the monitor so a recalibration is reflected immediately.
    recent_exceedances = None
    recent_limit = None
    ewmas = []
    prev_reading_at = None
    mon_state = mon.get_state()
    thr = mon_state.get("drift_threshold")
    # Skipped entirely while the baseline is still warming: no drift verdict
    # can be reached, so the query would only spend a round trip per device
    # per cycle, and the EWMAs it would read were scored against the previous
    # baseline anyway.
    # A stored EWMA is only comparable to the current threshold if it was
    # SCORED AGAINST THE CURRENT BASELINE. Recalibrating replaces the mean and
    # covariance, so the same audio yields a completely different distance
    # afterwards -- older EWMAs are a different measurement wearing the same
    # column name.
    #
    # Without this cutoff the rule read straight across a recalibration: on
    # 2026-09-03 P1 froze a fresh baseline at 11:18, and at 11:28 the window
    # pulled 29 readings that ALL predated it, found 29 of 29 "over" the new
    # threshold, and flagged the filter dirty at an actual EWMA of 0.30 --
    # seven minutes before the filter was even touched. Bounding by wall-clock
    # age alone could never have caught that; the boundary is the baseline,
    # not the calendar.
    baseline_frozen_at = (mon_state.get("baseline") or {}).get("last_ts")
    if thr is not None and mon_state.get("state") == "warm" and baseline_frozen_at:
        # Newest-first, so these are the readings that most recently got
        # SCORED. Clips dropped as silence never reached filter_ml_readings,
        # so hours with the HVAC off contribute nothing rather than diluting
        # the window with samples that carry no filter information.
        # One short of the window size: the reading being processed right now
        # is not stored yet, and DriftDetector.update adds its own exceedance,
        # so N-1 prior readings plus this one is exactly N.
        #
        # The age bound still applies on top, for a device whose blower runs
        # so little that 29 readings would otherwise span weeks.
        age_cutoff = isoparse(updated_at) - timedelta(hours=RECENT_WINDOW_MAX_AGE_HOURS)
        cutoff = max(age_cutoff, _epoch_dt(baseline_frozen_at)).isoformat()
        # recorded_at comes back too, at no extra cost: the newest row is this
        # device's PREVIOUS reading, which is what lets the adaptive block
        # below fire once per window instead of on every reading in it.
        hist = sb.table("filter_ml_readings").select("ewma_value, recorded_at") \
            .eq("device_mac", device_mac) \
            .gte("recorded_at", cutoff) \
            .not_.is_("ewma_value", "null") \
            .order("recorded_at", desc=True) \
            .limit(RECENT_WINDOW_READINGS - 1).execute()
        rows_hist = hist.data or []
        if rows_hist:
            prev_reading_at = isoparse(rows_hist[0]["recorded_at"]).timestamp()
        ewmas = [r["ewma_value"] for r in rows_hist]
        window_size = len(ewmas) + 1          # + the reading in hand
        if window_size >= RECENT_WINDOW_MIN:
            recent_exceedances = sum(1 for e in ewmas if e > thr)
            # Fraction of the window actually assembled, not of the nominal
            # 30, so a partly-filled window is judged by the same proportion
            # rather than being effectively stricter while it fills.
            recent_limit = max(RECENT_DIRTY_MIN,
                               math.ceil(RECENT_DIRTY_FRACTION * window_size))

    # Re-derive the threshold from accumulated readings before scoring this
    # one, so a long-running device tracks its environment instead of being
    # judged forever against one 20h window.
    #
    # Fires on the FIRST reading to land in each window, not on every reading
    # inside it. The bucket test on its own was true for every reading in the
    # window -- about six at a 10-minute wake -- and each of those re-ran all
    # three of adaptive_threshold's queries (~900 rows) to recompute a value
    # the no-op guard then threw away. Comparing against the previous
    # reading's bucket makes it exactly one, using a timestamp the window
    # query above already returned for free.
    reading_epoch = isoparse(updated_at).timestamp()
    this_bucket = int(reading_epoch // 3600) // ADAPTIVE_UPDATE_HOURS
    prev_bucket = (int(prev_reading_at // 3600) // ADAPTIVE_UPDATE_HOURS
                   if prev_reading_at is not None else None)
    if (ADAPTIVE_THRESHOLD and mon_state.get("state") == "warm"
            and prev_bucket is not None and this_bucket != prev_bucket):
        adapted = adaptive_threshold(sb, device_mac, mon, mon_state, updated_at)
        if adapted:
            new_thr, raw, anchor, n_clean = adapted
            mon.set_drift_threshold(new_thr)
            mon_state = mon.get_state()
            clamp_note = ""
            if abs(new_thr - raw) > 1e-9:
                clamp_note = (f", clamped from {raw:.2f} "
                              f"(anchor {anchor:.2f}, band "
                              f"{anchor*ADAPTIVE_MIN_FALL:.2f}-{anchor*ADAPTIVE_MAX_RISE:.2f})")
            print(f"  {device_mac}: threshold -> {new_thr:.3f} "
                  f"from p{ADAPTIVE_PERCENTILE} of {n_clean} clean readings{clamp_note}")

    # If this reading is the one that tips the baseline over its freeze gates,
    # the monitor derives the device's threshold from the calibration samples
    # themselves -- so they have to be in hand BEFORE process() runs. Fetched
    # only at that moment (one query per device per calibration, not per
    # reading) because the vectors are large and useless the rest of the time.
    warmup_samples = None
    if mon_state.get("state") in ("cold_start", "warming") and mon.baseline.n + 1 >= MIN_BASELINE_SAMPLES:
        warmup_samples = fetch_warmup_features(sb, device_mac, mon_state)

    result = mon.process(feats, ts=isoparse(updated_at).timestamp(),
                         recent_exceedances=recent_exceedances,
                         recent_limit=recent_limit,
                         warmup_samples=warmup_samples)

    if mon.threshold_detail and result["state"] == "warm":
        td = mon.threshold_detail
        print(f"  {device_mac}: baseline frozen -- threshold "
              f"{mon.get_state()['drift_threshold']:.3f} via {td['method']}"
              + (f" (empirical p{td['quantile']*100:g} over {len(warmup_samples)} "
                 f"warmup samples; closed form would be {td['closed_form']:.3f})"
                 if td["method"] == "empirical" and td.get("closed_form")
                 else f" (n={td['n_samples']}, d={td['n_features']})"))
        mon.threshold_detail = None
    verdict = result["decision"]

    # Mirrors FilterHealthMonitor's own warmup rule (drift_detector.py):
    # the filter is assumed clean for the whole warmup window, so every
    # pre-WARM reading feeds the spectrum average unconditionally, same
    # as the 5-feature baseline -- kept independent of that class since
    # spectrum averaging is a display concern, not a detection one, but
    # "baseline" should mean the same thing (same sample count, same
    # readings) in both places.
    if result["state"] in ("cold_start", "warming"):
        spectrum_mean, spectrum_n = update_spectrum_baseline(
            spectrum_mean, spectrum_n, compute_spectrum(y, SR))

    recent_note = (f", {result['recent_exceedances']}/{result.get('recent_limit')} "
                   f"of last {len(ewmas) + 1}"
                   if result.get("recent_exceedances") is not None else "")
    flag_note = f"  [{result['flag']}]" if result["flag"] else ""
    print(f"  {device_mac}: {verdict} (decided_by={result['decided_by']}, "
          f"state={result['state']}{recent_note}){wind_note}{flag_note}")

    sb.table("filter_ml_readings").insert({
        "device_mac": device_mac,
        "recorded_at": updated_at,
        "classifier_label": result["classifier_label"],
        "classifier_confidence": result["classifier_p_dirty"],
        "mahalanobis_distance": result.get("distance"),
        "ewma_value": result.get("ewma"),
        "decision": verdict,
        "decided_by": result["decided_by"],
        "disagreement_flag": result["flag"],
        # The drift maths' INPUT, not just its output. Kept for calibrating
        # readings too: those are the warmup samples the baseline gets fitted
        # from, and storing them is what lets a frozen baseline be audited
        # later and its threshold derived from measurement rather than from a
        # Gaussian assumption. Ordered as DEFAULT_DRIFT_FEATURES.
        "features": [float(feats[k]) for k in DEFAULT_DRIFT_FEATURES],
    }).execute()

    # NOTE: this used to also write latest_verdict/latest_confidence back onto
    # audio_logs. That created a self-sustaining reprocessing loop: audio_logs
    # has a BEFORE UPDATE trigger (trg_audio_logs_updated_at) that sets
    # updated_at = now() on *any* update, so writing the verdict bumped the
    # very timestamp run_once() uses to decide whether a clip is new. Every
    # poll cycle then saw "new" audio, re-scored the same clip, and bumped it
    # again -- ~92 re-scores per device per hour at a 30s poll interval
    # against 6 real uploads. That inflated EWMA toward the raw distance
    # (EWMA smooths across *different* readings; feeding it one value
    # repeatedly just converges on it) and tripped the consecutive-exceedance
    # rule, producing dirty verdicts on clean filters. It also poisoned
    # warmup: baselines "reached" 120 samples that were really a handful of
    # clips repeated, collapsing the covariance and inflating every later
    # Mahalanobis distance.
    #
    # Nothing reads those two columns -- the app takes verdicts from
    # filter_ml_readings (web/src/lib/audioRecordings.js) -- so the write is
    # simply removed rather than reworked.
    save_monitor(sb, device_mac, mon, updated_at, spectrum_mean, spectrum_n, had_row)


def run_once(sb, classifier_fn):
    resp = sb.table("audio_logs").select("device_id, is_lora, storage_path, updated_at").execute()
    rows = resp.data or []
    print(f"{len(rows)} device(s) in audio_logs")

    for row in rows:
        device_mac = row["device_id"]

        if row.get("is_lora"):
            print(f"  {device_mac}: LoRa row (mfcc_coefficients only, no wav) -- not handled yet, skipping")
            continue
        if not row.get("storage_path"):
            print(f"  {device_mac}: no storage_path yet -- skipping")
            continue

        # The whole row, not just last_processed_at: load_monitor needs the
        # same row microseconds later, and fetching it twice was a wasted
        # round trip on every reading of every device.
        baseline_resp = sb.table("device_baselines").select("*") \
            .eq("device_mac", device_mac).maybe_single().execute()
        baseline_row = baseline_resp.data if baseline_resp else None
        last_processed = (baseline_row or {}).get("last_processed_at")
        if last_processed and row["updated_at"] <= last_processed:
            continue

        try:
            process_device(sb, device_mac, row["storage_path"], row["updated_at"],
                           classifier_fn, baseline_row=baseline_row)
        except Exception as e:
            print(f"  {device_mac}: ERROR {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=None,
                       help="repeat every N seconds instead of running once "
                            "(omit this if scheduling via Task Scheduler)")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set -- copy "
              "service/.env.example to service/.env and fill them in.")
        sys.exit(1)

    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    classifier_fn = load_classifier()

    if args.loop:
        while True:
            run_once(sb, classifier_fn)
            time.sleep(args.loop)
    else:
        run_once(sb, classifier_fn)


if __name__ == "__main__":
    main()
