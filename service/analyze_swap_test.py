"""
service/analyze_swap_test.py
Reports whether the live pipeline actually detected a real filter swap on
one device -- pulls that device's full ML decision history plus its raw
sensor readings (windSpeed, pressure) from Supabase and prints a timeline,
a before/after summary split at the swap time, and a first look at
whether pressure/windSpeed moved when the filter did (an open question --
see extract_features_v2.py's caveat that pressure_pa may just be ambient
barometric pressure, not an in-duct differential reading; this is the
first real data to actually check that against).

Usage:
  python service/analyze_swap_test.py <device_mac> --swap-at "2026-08-20T14:00:00"

--swap-at is optional (in local time or with an explicit offset -- passed
straight to dateutil.parser.isoparse, so any ISO-ish string works). Without
it, this just prints the full timeline with no before/after split.
"""
import argparse
import os
import statistics
from pathlib import Path

from dateutil.parser import isoparse
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).resolve().parent / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")


def fetch_readings(sb, device_mac):
    resp = sb.table("filter_ml_readings").select("*") \
        .eq("device_mac", device_mac).order("recorded_at", desc=False).execute()
    return resp.data or []


def fetch_sensor_logs(sb, device_mac):
    dev = sb.table("devices").select("id").eq("device_mac", device_mac).maybe_single().execute()
    if not dev or not dev.data:
        return []
    resp = sb.table("sensor_logs").select("recorded_at, windSpeed, pressure_pa, temp_c, humidity") \
        .eq("device_id", dev.data["id"]).order("recorded_at", desc=False).execute()
    return resp.data or []


def print_timeline(readings, swap_at):
    # No per-reading "state" column exists in filter_ml_readings (only
    # device_baselines has a current snapshot) -- this column is the
    # classifier's raw label, shown regardless of what actually decided
    # the final verdict (see the decided_by column for that).
    print(f"\n{'TIME':20s} {'DECISION':12s} {'BY':20s} {'CLF_LABEL':10s} {'DIST':>6s} {'EWMA':>6s}  FLAG")
    for r in readings:
        t = isoparse(r["recorded_at"])
        marker = " <<< SWAP" if swap_at and abs((t - swap_at).total_seconds()) < 60 else ""
        dist = f"{r['mahalanobis_distance']:.2f}" if r.get("mahalanobis_distance") is not None else "  -"
        ewma = f"{r['ewma_value']:.2f}" if r.get("ewma_value") is not None else "  -"
        flag = (r.get("disagreement_flag") or "")[:40]
        # decided_by is None for any row from before this column existed --
        # tolerate that rather than crash on older test data.
        decided_by = r.get("decided_by") or "-"
        print(f"{t.strftime('%Y-%m-%d %H:%M:%S'):20s} {r['decision']:12s} "
              f"{decided_by:20s} {r['classifier_label'] or '-':10s} "
              f"{dist:>6s} {ewma:>6s}  {flag}{marker}")


def summarize_period(label, readings):
    if not readings:
        print(f"  {label}: no readings")
        return
    decisions = {}
    for r in readings:
        decisions[r["decision"]] = decisions.get(r["decision"], 0) + 1
    dist_vals = [r["mahalanobis_distance"] for r in readings if r.get("mahalanobis_distance") is not None]
    dist_str = f", mean distance {statistics.mean(dist_vals):.2f}" if dist_vals else ""
    print(f"  {label}: {len(readings)} readings, decisions={decisions}{dist_str}")


def summarize_sensor_period(label, rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        print(f"  {label} {key}: no data")
        return None
    mean = statistics.mean(vals)
    print(f"  {label} {key}: n={len(vals)}, mean={mean:.2f}, "
          f"min={min(vals):.2f}, max={max(vals):.2f}")
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("device_mac")
    parser.add_argument("--swap-at", default=None,
                       help="ISO timestamp of the known filter swap, e.g. 2026-08-20T14:00:00")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set -- copy "
              "service/.env.example to service/.env and fill them in.")
        return

    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    swap_at = isoparse(args.swap_at) if args.swap_at else None

    readings = fetch_readings(sb, args.device_mac)
    if not readings:
        print(f"No filter_ml_readings found for {args.device_mac} yet.")
        return

    print(f"{len(readings)} ML reading(s) for {args.device_mac}")
    print_timeline(readings, swap_at)

    if swap_at:
        before = [r for r in readings if isoparse(r["recorded_at"]) < swap_at]
        after = [r for r in readings if isoparse(r["recorded_at"]) >= swap_at]
        print(f"\n=== Before/after swap at {swap_at} ===")
        summarize_period("Before (should be clean/MERV10)", before)
        summarize_period("After  (should be dirty/FPR12)", after)

        first_dirty_after = next((r for r in after if r["decision"] == "dirty"), None)
        if first_dirty_after:
            lag = isoparse(first_dirty_after["recorded_at"]) - swap_at
            print(f"\n  First 'dirty' decision after swap: {first_dirty_after['recorded_at']} "
                  f"({lag} after the swap) -- SUCCESS if this is a real, sustained detection.")
        else:
            n_calibrating = sum(1 for r in after if r["decision"] == "calibrating")
            print(f"\n  No 'dirty' decision yet in the post-swap window "
                  f"({n_calibrating}/{len(after)} still 'calibrating' -- give it more warmup "
                  f"time before calling this a failure)." if n_calibrating else
                  "\n  No 'dirty' decision in the post-swap window -- worth investigating.")

    # First real look at whether pressure/windSpeed track the swap --
    # genuinely unknown going in, don't assume either way.
    sensor_rows = fetch_sensor_logs(sb, args.device_mac)
    if sensor_rows and swap_at:
        before_sensor = [r for r in sensor_rows if isoparse(r["recorded_at"]) < swap_at]
        after_sensor = [r for r in sensor_rows if isoparse(r["recorded_at"]) >= swap_at]
        print(f"\n=== sensor_logs before/after swap (does pressure or windSpeed move too?) ===")
        for key in ("pressure_pa", "windSpeed"):
            b = summarize_sensor_period("Before", before_sensor, key)
            a = summarize_sensor_period("After ", after_sensor, key)
            if b is not None and a is not None:
                print(f"  -> {key} changed {a - b:+.2f} ({(a - b) / b:+.1%}) after the swap")
    elif sensor_rows:
        print(f"\n{len(sensor_rows)} sensor_logs row(s) available -- pass --swap-at to split them.")


if __name__ == "__main__":
    main()
