"""
service/recalibrate_threshold.py
Re-derives each device's drift threshold from its OWN measured behaviour on
the filter currently fitted, instead of the single pooled number inherited
from one lab site.

WHY
The shipped threshold (2.341) came from pooled_calibration.json: one site
(AbrahamHouse, MERV10 vs FPR12), 95th percentile of clean distances, with a
self-reported 58% detection rate. It was never validated against these
devices. Measured against live data it is far too tight -- P1 and P7 sat
above it for ~90% of readings on filters nobody had touched, which is not a
filter condition, it is a mis-set constant.

Per-device rather than pooled, because the distributions genuinely differ:
P3's EWMA 99th percentile is 2.67 while P6's is 3.71. One number forces a
compromise that is simultaneously too tight for the noisy devices and too
loose for the quiet ones. device_baselines.threshold is already per-device,
so this costs nothing structurally.

CALIBRATED AGAINST EWMA, NOT RAW DISTANCE
DriftDetector.update() compares self.ewma against the threshold, not the
raw Mahalanobis distance. The EWMA is smoothed and therefore much tighter
(P6: distance p99 5.69 vs EWMA p99 3.71), so calibrating on distance
percentiles would set the threshold roughly 50% too high and blind the
detector. Only readings taken after the baseline froze are used, since
warmup readings have no distance at all.

!!! THIS DEFINES "NORMAL" AS WHATEVER THE FILTER IS DOING NOW !!!
The calibration assumes the currently fitted filters are in the state you
want treated as baseline. If a filter is genuinely dirty right now, this
bakes that in and the device will never flag it. Only run this when you
believe the fitted filters are clean -- the same precondition as
recalibrating a baseline.

Usage:
  python service/recalibrate_threshold.py                 # dry run
  python service/recalibrate_threshold.py --yes           # apply
  python service/recalibrate_threshold.py --yes --percentile 99 --margin 1.10
  python service/recalibrate_threshold.py --yes --write-pooled   # also update
                                                                 # pooled_calibration.json
"""
import argparse
import json
import os
import statistics
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(Path(__file__).resolve().parent / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
POOLED_PATH = ROOT / "pooled_calibration.json"

# Below this many post-freeze readings a percentile is too noisy to trust --
# P11 had 75 samples and a p99 of 3.67 against a p50 of 1.38, which is an
# artefact of thin data rather than a real spread. Such devices keep their
# existing threshold and are reported, not silently skipped.
MIN_SAMPLES = 200


def percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="apply (default is a dry run)")
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="percentile of clean EWMA to sit the threshold at (default 99)")
    ap.add_argument("--margin", type=float, default=1.10,
                    help="multiplier applied on top of the percentile (default 1.10). "
                         "EWMA values are autocorrelated, so consecutive exceedances are "
                         "more likely than independence would suggest and the patience "
                         "rule alone does not absorb it.")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    ap.add_argument("--write-pooled", action="store_true",
                    help="also write the median of the new thresholds to "
                         "pooled_calibration.json, which only affects devices "
                         "that have no baseline row yet")
    args = ap.parse_args()

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set in service/.env")
        return

    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    baselines = [b for b in (sb.table("device_baselines")
                 .select("device_mac, state, threshold, baseline_last_sample_at")
                 .execute().data or []) if b.get("state") == "warm"]
    names = {d["device_mac"]: d.get("name") or d["device_mac"]
             for d in (sb.table("devices").select("device_mac, name").execute().data or [])
             if d.get("device_mac")}

    print(f"{'DEVICE':22s} {'N':>5s} {'OLD':>7s} {'p50':>6s} "
          f"{f'p{args.percentile:g}':>6s} {'NEW':>7s}  {'%>OLD':>6s} {'%>NEW':>6s}")

    updates, proposed = [], []
    for b in sorted(baselines, key=lambda x: names.get(x["device_mac"], "")):
        mac = b["device_mac"]
        rows = (sb.table("filter_ml_readings")
                .select("ewma_value, recorded_at")
                .eq("device_mac", mac)
                .gte("recorded_at", b["baseline_last_sample_at"])
                .order("recorded_at", desc=True).limit(2000).execute().data or []) \
               if b.get("baseline_last_sample_at") else []
        ewmas = [r["ewma_value"] for r in rows if r.get("ewma_value") is not None]

        old = b.get("threshold")
        label = names.get(mac, mac)[:21]
        if len(ewmas) < args.min_samples:
            print(f"{label:22s} {len(ewmas):5d} {old or 0:7.2f} "
                  f"{'':>6s} {'':>6s} {'-- keeping (too few samples)':>7s}")
            continue

        p = percentile(ewmas, args.percentile)
        new = round(p * args.margin, 3)
        over_old = 100.0 * sum(1 for e in ewmas if e > (old or 0)) / len(ewmas)
        over_new = 100.0 * sum(1 for e in ewmas if e > new) / len(ewmas)

        print(f"{label:22s} {len(ewmas):5d} {old or 0:7.2f} "
              f"{percentile(ewmas,50):6.2f} {p:6.2f} {new:7.2f}  "
              f"{over_old:5.1f}% {over_new:5.1f}%")
        updates.append((mac, new))
        proposed.append(new)

    if not updates:
        print("\nNothing to update.")
        return

    print(f"\n{len(updates)} device(s) would move to their own threshold "
          f"(p{args.percentile:g} x {args.margin}).")
    print(f"Range {min(proposed):.2f}-{max(proposed):.2f}, median {statistics.median(proposed):.2f} "
          f"(was a single pooled {baselines[0].get('threshold')}).")

    if not args.yes:
        print("\nDry run. Re-run with --yes to apply.")
        print("Only do so if the filters currently fitted are the state you want "
              "treated as normal -- this defines baseline from present behaviour.")
        return

    for mac, new in updates:
        sb.table("device_baselines").update({"threshold": new}).eq("device_mac", mac).execute()
    print(f"\nApplied to {len(updates)} device(s).")
    print("Takes effect on each device's next reading: DriftDetector compares the "
          "existing EWMA against the new threshold, so an elevated exceedance_count "
          "clears itself without any reset, reflash or re-warm.")

    if args.write_pooled:
        pooled = round(statistics.median(proposed), 6)
        data = json.loads(POOLED_PATH.read_text()) if POOLED_PATH.exists() else {}
        data["threshold"] = pooled
        data["calibrated_from"] = (
            f"live per-device EWMA p{args.percentile:g} x {args.margin} across "
            f"{len(updates)} devices; median taken for devices with no baseline yet"
        )
        POOLED_PATH.write_text(json.dumps(data, indent=2))
        print(f"pooled_calibration.json threshold -> {pooled} "
              f"(only used for devices with no baseline row yet)")


if __name__ == "__main__":
    main()
