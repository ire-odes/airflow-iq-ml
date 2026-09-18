"""
service/check_baseline.py
Reports one device's warmup/calibration progress straight from
device_baselines.

A baseline freezes only when BOTH conditions hold (see
drift_detector.BaselineProfile.is_ready):

  1. it has at least min_baseline_samples readings (120), and
  2. those readings span at least min_baseline_span_seconds of
     wall-clock time (20h).

The span condition exists because sample count alone was not enough: at
sampling mode's 60s wakes a device reaches 120 samples in ~2h, and a
baseline fitted to a 2-hour window describes one narrow set of operating
conditions rather than the device's normal range -- three devices flipped
to "dirty" overnight on filters nobody had touched. Both gates are shown
below because either one can be the thing you're actually waiting on, and
which one it is changes what (if anything) is worth doing about it.

Nothing in the app UI surfaces this, so this is the only way to see
calibration progress short of the Supabase SQL editor.

Usage:
  python service/check_baseline.py <device_mac>
"""
import argparse
import os
from pathlib import Path

from dateutil.parser import isoparse
from dotenv import load_dotenv
from supabase import create_client

# Fallback only, for rows written before device_baselines recorded the
# requirement it was judged against. The live value comes from the row --
# see 20260829010000_baseline_min_span_column.sql.
DEFAULT_MIN_SPAN_SECONDS = 20 * 3600

load_dotenv(Path(__file__).resolve().parent / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("device_mac")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set -- copy "
              "service/.env.example to service/.env and fill them in.")
        return

    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    # effective_wake_seconds is a computed column (see
    # supabase/migrations/20260821120000_calibration_sampling_mode.sql):
    # it returns 60 while the device is in post-recalibration sampling
    # mode, otherwise the device's configured wake_interval_seconds.
    # Both are shown below because they differ exactly when sampling
    # mode is active, which is the thing worth seeing at a glance.
    dev = sb.table("devices") \
        .select("id, name, wake_interval_seconds, calibration_started_at, effective_wake_seconds") \
        .eq("device_mac", args.device_mac).maybe_single().execute()
    dev_row = dev.data or {}
    configured_seconds = dev_row.get("wake_interval_seconds") or 600
    wake_seconds = dev_row.get("effective_wake_seconds") or configured_seconds
    calibration_started_at = dev_row.get("calibration_started_at")
    dev_label = dev_row.get("name") or args.device_mac

    row = sb.table("device_baselines").select("*") \
        .eq("device_mac", args.device_mac).maybe_single().execute()

    if not row or not row.data:
        print(f"No device_baselines row yet for {dev_label} ({args.device_mac}) -- "
              f"the poller hasn't successfully processed a reading for this "
              f"device yet. Check that audio is actually arriving in audio_logs "
              f"and that the scheduled task is running without errors.")
        return

    d = row.data
    n = d.get("n_warmup_samples") or 0
    target = d.get("min_baseline_samples") or 120
    span_target = d.get("min_baseline_span_seconds") or DEFAULT_MIN_SPAN_SECONDS
    state = d.get("state") or "cold_start"
    last_processed = d.get("last_processed_at")

    first_at = d.get("baseline_first_sample_at")
    last_at = d.get("baseline_last_sample_at")
    # Timestamped from the audio's own upload time, so this measures the
    # window the recordings cover -- not how long the poller has been up.
    span_seconds = (isoparse(last_at) - isoparse(first_at)).total_seconds() \
        if first_at and last_at else 0.0

    print(f"{dev_label} ({args.device_mac})")
    print(f"  state: {state}")
    if wake_seconds != configured_seconds:
        print(f"  device wake interval: {wake_seconds}s (SAMPLING MODE; "
              f"normally {configured_seconds}s), started {calibration_started_at}")
    else:
        print(f"  device wake interval: {wake_seconds}s ({wake_seconds / 60:.0f} min)")
    print(f"  last processed: {last_processed or 'never'}")

    if state == "warm":
        thr = d.get("threshold")
        ewma = d.get("ewma")
        print(f"\n  Baseline is frozen -- drift detection is live.")
        print(f"  Fitted from {n} samples spanning {span_seconds / 3600:.1f}h"
              + (f" ({first_at} -> {last_at})" if first_at and last_at else
                 "  (span not recorded -- fitted before span tracking existed)"))
        # Thresholds are per-device now (service/recalibrate_threshold.py),
        # derived from each device's own clean EWMA distribution rather than
        # the single pooled 2.341 inherited from one lab site. Printing the
        # headroom matters more than the raw pair: a device sitting just
        # under its threshold is one noisy hour from flagging, and that is
        # invisible if you only see two numbers.
        if isinstance(thr, (int, float)) and isinstance(ewma, (int, float)):
            headroom = thr - ewma
            pct = (ewma / thr * 100) if thr else 0
            print(f"  threshold={thr:.2f}  ewma={ewma:.2f}  "
                  f"({pct:.0f}% of threshold, headroom {headroom:+.2f})")
            if ewma > thr:
                print(f"  -> ewma is ABOVE threshold; {d.get('exceedance_count')} consecutive "
                      f"exceedance(s) so far, flags dirty at 3")
        else:
            print(f"  threshold={thr}, ewma={ewma}")
        print(f"  exceedance_count={d.get('exceedance_count')}")
        if thr and abs(float(thr) - 2.341336741419201) < 1e-6:
            print(f"  NOTE: still on the old pooled threshold (2.341, single lab site, "
                  f"58% detection). Run service/recalibrate_threshold.py to derive one "
                  f"from this device's own measurements.")
        if first_at and last_at and span_seconds < span_target:
            print(f"\n  WARNING: this baseline only covers {span_seconds / 3600:.1f}h, "
                  f"under the {span_target / 3600:.0f}h now required. It was frozen "
                  f"before that rule existed and describes a narrow slice of "
                  f"operating conditions, so it will read ordinary day/night "
                  f"variation as drift. Recalibrate this device to refit it.")
        return

    # Two independent gates -- report both, since either can be the one
    # actually holding the baseline back and they respond to different
    # things (more readings vs. more elapsed time).
    samples_left = max(target - n, 0)
    span_left_s = max(span_target - span_seconds, 0.0)

    print(f"\n  Freezes when BOTH conditions are met:")
    print(f"    samples: {n}/{target}"
          + (f"  ({n / target:.0%})" if target else "")
          + ("  [met]" if samples_left == 0 else f"  [{samples_left} more]"))
    print(f"    span:    {span_seconds / 3600:.1f}h/{span_target / 3600:.0f}h"
          + (f"  ({span_seconds / span_target:.0%})" if span_target else "")
          + ("  [met]" if span_left_s == 0 else f"  [{span_left_s / 3600:.1f}h more]"))
    if first_at:
        print(f"    window:  {first_at}  ->  {last_at}")

    # Sample ETA assumes every future wake yields a usable reading (HVAC on,
    # clip not silent, poller healthy) -- a floor, not a forecast. The span
    # ETA is calendar time and assumes the device keeps reporting at all:
    # span only advances when readings actually arrive, so a device that
    # goes quiet stops accruing it.
    samples_eta_h = (samples_left * wake_seconds) / 3600
    span_eta_h = span_left_s / 3600

    if samples_left == 0 and span_left_s == 0:
        print(f"\n  Both conditions met -- should freeze on the next reading.")
    else:
        binding = "sample count" if samples_eta_h > span_eta_h else "time span"
        print(f"\n  Earliest freeze: ~{max(samples_eta_h, span_eta_h):.1f}h from now, "
              f"limited by {binding}.")
        if span_eta_h > samples_eta_h:
            print(f"  Nothing to fix -- it needs to hear this device across a full "
                  f"day and night before its idea of 'normal' is trustworthy. "
                  f"Faster sampling makes the baseline denser, not readier.")
        else:
            print(f"  If the real pace is slower than that, readings are being "
                  f"skipped -- most likely the HVAC cycling off (silent clips are "
                  f"dropped before reaching the baseline, see "
                  f"extract_features_v2.SILENCE_RMS_THRESHOLD), or the poller "
                  f"failing. Check the Task Scheduler job's last run result and "
                  f"this device's audio_logs.updated_at history.")


if __name__ == "__main__":
    main()
