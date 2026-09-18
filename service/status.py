"""
service/status.py
Fleet status at a glance: for every device, its three most recent ML
readings with how far each sat from that device's threshold, plus when the
baseline was last calibrated.

Reads only -- safe to run any time, disturbs nothing.

Why "distance from threshold" rather than the raw EWMA: thresholds are
per-device now (see service/recalibrate_threshold.py), so 3.1 is comfortably
clean on one device and over the line on another. The headroom is the number
that means the same thing everywhere.

Usage:
  python service/status.py                 # all devices with a baseline
  python service/status.py --all           # include ones with no baseline yet
  python service/status.py --mac E4B3...   # just one
  python service/status.py -n 5            # more readings each
"""
import argparse
import os
from pathlib import Path

from dateutil.parser import isoparse
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).resolve().parent / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# Mirror of the count-window rule in poll_and_infer.py, so its progress is
# visible here rather than only firing silently. Kept in sync by hand; if
# these drift apart this report will misstate how close a device is.
RECENT_WINDOW_READINGS = 30
RECENT_DIRTY_FRACTION = 0.20
RECENT_DIRTY_MIN = 3


def ago(ts):
    if not ts:
        return "never"
    from datetime import datetime, timezone
    secs = (datetime.now(timezone.utc) - isoparse(ts).astimezone(timezone.utc)).total_seconds()
    if secs < 90:
        return f"{secs:.0f}s ago"
    if secs < 5400:
        return f"{secs / 60:.0f}m ago"
    if secs < 172800:
        return f"{secs / 3600:.1f}h ago"
    return f"{secs / 86400:.1f}d ago"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--readings", type=int, default=3)
    ap.add_argument("--mac", default=None)
    ap.add_argument("--all", action="store_true",
                    help="include devices with no baseline row yet")
    args = ap.parse_args()

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set in service/.env")
        return

    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    devices = [d for d in (sb.table("devices")
               .select("device_mac, name, duct_role, calibration_started_at")
               .execute().data or []) if d.get("device_mac")]
    baselines = {b["device_mac"]: b for b in (sb.table("device_baselines")
                 .select("*").execute().data or [])}

    if args.mac:
        want = args.mac.replace(":", "").upper()
        devices = [d for d in devices if d["device_mac"].upper() == want]
    if not args.all:
        devices = [d for d in devices if d["device_mac"] in baselines]

    devices.sort(key=lambda d: (d.get("name") or "~", d["device_mac"]))
    if not devices:
        print("No devices matched.")
        return

    for d in devices:
        mac = d["device_mac"]
        b = baselines.get(mac) or {}
        thr = b.get("threshold")
        state = b.get("state") or "(no baseline)"
        role = f" [{d['duct_role']}]" if d.get("duct_role") else ""

        print(f"\n{'=' * 76}")
        print(f"{d.get('name') or mac}{role}   {mac}")
        print(f"{'=' * 76}")

        span_h = None
        if b.get("baseline_first_sample_at") and b.get("baseline_last_sample_at"):
            span_h = (isoparse(b["baseline_last_sample_at"])
                      - isoparse(b["baseline_first_sample_at"])).total_seconds() / 3600

        print(f"  state          : {state}"
              + (f"  ({b.get('n_warmup_samples')} warmup samples)" if b.get("n_warmup_samples") else ""))
        if state == "warm":
            print(f"  calibrated     : {ago(b.get('baseline_last_sample_at'))}"
                  + (f", fitted over {span_h:.1f}h" if span_h else "")
                  + (f", frozen {str(b.get('baseline_last_sample_at'))[:16]}"
                     if b.get("baseline_last_sample_at") else ""))
            print(f"  threshold      : {thr:.2f}" if thr is not None else "  threshold      : -")
            print(f"  exceedance run : {b.get('exceedance_count', 0)} consecutive "
                  f"(flags dirty at 3)")
        else:
            print(f"  calibrating    : started {ago(d.get('calibration_started_at'))}"
                  if d.get("calibration_started_at") else "  calibrating    : not yet started")
            print(f"  -> no verdicts until the baseline freezes")

        rows = (sb.table("filter_ml_readings")
                .select("recorded_at, decision, decided_by, mahalanobis_distance, "
                        "ewma_value, classifier_confidence, disagreement_flag")
                .eq("device_mac", mac)
                .order("recorded_at", desc=True)
                .limit(args.readings).execute().data or [])

        if not rows:
            print("  readings       : none yet")
            continue

        # The other half of the alert logic, invisible from the per-reading
        # rows alone. Counted over the last N READINGS rather than a fixed
        # number of hours, matching the poller: clips recorded with the
        # blower off are dropped as silence and never become readings, so a
        # wall-clock window would hold a different number of samples on
        # every device and every day.
        # Only meaningful once the baseline is frozen. While a device is
        # warming, the stored EWMAs were produced by the PREVIOUS baseline,
        # so comparing them to the current threshold counts exceedances
        # against a reference that no longer exists -- it reads as an alarm
        # on a device that is not even being judged yet.
        if thr is not None and state == "warm":
            import math
            win = ((sb.table("filter_ml_readings").select("ewma_value")
                    .eq("device_mac", mac)
                    .not_.is_("ewma_value", "null")
                    .order("recorded_at", desc=True)
                    .limit(RECENT_WINDOW_READINGS).execute().data) or [])
            over = sum(1 for r in win if r["ewma_value"] > thr)
            limit = max(RECENT_DIRTY_MIN,
                        math.ceil(RECENT_DIRTY_FRACTION * len(win)))
            print(f"  last {len(win):>2d} reads : {over} over threshold; "
                  f"the window rule fires at {limit}")

        print(f"\n  {'WHEN':16s} {'AGO':>10s} {'DECISION':11s} {'BY':13s} "
              f"{'DIST':>6s} {'EWMA':>6s} {'vs THRESHOLD':>13s}")
        for r in rows:
            t = isoparse(r["recorded_at"])
            dist = f"{r['mahalanobis_distance']:.2f}" if r.get("mahalanobis_distance") is not None else "-"
            ewma = r.get("ewma_value")
            ewma_s = f"{ewma:.2f}" if ewma is not None else "-"
            # Signed headroom: negative means over the line. Reads the same
            # way on every device regardless of its own threshold.
            if ewma is not None and thr is not None:
                head = thr - ewma
                vs = f"{head:+.2f}" + (" OVER" if head < 0 else "")
            else:
                vs = "-"
            print(f"  {t.strftime('%m-%d %H:%M:%S'):16s} {ago(r['recorded_at']):>10s} "
                  f"{r['decision']:11s} {(r.get('decided_by') or '-'):13s} "
                  f"{dist:>6s} {ewma_s:>6s} {vs:>13s}")
            if r.get("disagreement_flag"):
                print(f"      flag: {r['disagreement_flag'][:66]}")


if __name__ == "__main__":
    main()
