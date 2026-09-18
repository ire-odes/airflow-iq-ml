"""
service/recalibrate_all.py
Recalibrates devices from the command line -- the same thing the app's
"Recalibrate" button does, but for the whole fleet at once instead of one
click per device.

Recalibrating means two writes:
  1. delete the device's device_baselines row, so FilterHealthMonitor
     starts from a fresh cold_start instead of a stale/contaminated
     baseline, and
  2. stamp devices.calibration_started_at = now(), which puts the device
     into sampling mode -- effective_wake_seconds() then returns 60 until
     the baseline freezes, so it collects its 120 warmup samples in hours
     rather than a day. See
     supabase/migrations/20260821120000_calibration_sampling_mode.sql.

Only do this right after fitting a filter you believe is clean: the whole
point of the baseline is that it encodes "normal" for this device, and
warmup readings are taken on trust (drift_detector.py assumes the filter
is clean during COLD_START/WARMING).

Uses the service role key, so it bypasses RLS -- it is not subject to the
owner-only delete policy the app's button goes through.

Usage:
  python service/recalibrate_all.py                 # dry run: show targets
  python service/recalibrate_all.py --yes           # actually recalibrate
  python service/recalibrate_all.py --yes --mac AA.. --mac BB..   # subset
  python service/recalibrate_all.py --include-silent  # also devices with
                                                      # no audio_logs row
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).resolve().parent / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true",
                        help="actually perform the recalibration (default is a dry run)")
    parser.add_argument("--mac", action="append", default=None,
                        help="limit to this device MAC (repeatable); default is every device")
    parser.add_argument("--include-silent", action="store_true",
                        help="also include devices that have never uploaded audio")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set in service/.env")
        sys.exit(1)

    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    devices = (sb.table("devices")
               .select("device_mac, name, wake_interval_seconds, calibration_started_at")
               .execute().data or [])
    baselines = {r["device_mac"]: r for r in (
        sb.table("device_baselines")
          .select("device_mac, state, n_warmup_samples, ewma").execute().data or [])}
    with_audio = {r["device_id"] for r in (
        sb.table("audio_logs").select("device_id").execute().data or [])}

    targets = []
    for d in devices:
        mac = d.get("device_mac")
        if not mac:
            continue
        if args.mac and mac not in args.mac:
            continue
        # A device that has never uploaded audio can't warm up a baseline,
        # so recalibrating it is a no-op that just sets it wide awake at 60s
        # and burns battery. Skipped unless explicitly asked for.
        if not args.include_silent and mac not in with_audio:
            continue
        targets.append(d)

    if not targets:
        print("No matching devices.")
        return

    print(f"{'DEVICE':22s} {'MAC':16s} {'STATE':11s} {'SAMPLES':>7s}  EWMA")
    for d in sorted(targets, key=lambda x: x.get("name") or ""):
        b = baselines.get(d["device_mac"]) or {}
        ewma = b.get("ewma")
        print(f"{(d.get('name') or '-')[:21]:22s} {d['device_mac']:16s} "
              f"{b.get('state') or '(no row)':11s} "
              f"{str(b.get('n_warmup_samples') if b.get('n_warmup_samples') is not None else '-'):>7s}"
              f"  {f'{ewma:.2f}' if isinstance(ewma, (int, float)) else '-'}")

    if not args.yes:
        print(f"\nDry run -- {len(targets)} device(s) would be recalibrated: "
              f"baseline cleared and put into 60s sampling mode.")
        print("Re-run with --yes to apply. Only do this if the filters "
              "currently fitted are clean.")
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    print(f"\nRecalibrating {len(targets)} device(s)...")
    for d in targets:
        mac = d["device_mac"]
        # Stamp first: a device with no baseline row yet still needs sampling
        # mode, so this must not depend on the delete matching anything.
        sb.table("devices").update({"calibration_started_at": now_iso}).eq("device_mac", mac).execute()
        sb.table("device_baselines").delete().eq("device_mac", mac).execute()
        print(f"  {mac}: baseline cleared, sampling mode on")

    # Verify rather than trust: the poller upserts baselines continuously, and
    # a delete landing mid-cycle used to get resurrected by it (poll_and_infer
    # save_monitor now guards against that, but confirm rather than assume).
    still = {r["device_mac"] for r in (
        sb.table("device_baselines").select("device_mac, state")
          .in_("device_mac", [d["device_mac"] for d in targets])
          .execute().data or []) if r.get("state") == "warm"}
    if still:
        print(f"\nWARNING: {len(still)} device(s) still hold a warm baseline: "
              f"{', '.join(sorted(still))}")
        print("The poller most likely rewrote them. Stop the scheduled task "
              "'Airflow IQ inference poller' and re-run this.")
    else:
        print(f"\nAll {len(targets)} device(s) reset. They now need "
              f"120 warmup readings each before drift verdicts resume; "
              f"track with:  python service/check_baseline.py <mac>")


if __name__ == "__main__":
    main()
