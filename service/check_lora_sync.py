"""
service/check_lora_sync.py
Reports whether the LoRaWAN nodes are actually slot-synchronised, using
only what they've already uplinked -- no serial cable, and nothing that
disturbs the devices.

WHY NOT JUST WATCH THE SERIAL MONITOR:
the node's clock, slot and LoRaWAN session all live in RTC memory, which
survives deep sleep but NOT a reset. Pressing reset to read the banner
therefore wipes the very time sync you were checking for, and forces a
full OTAA rejoin -- which makes the network drop any config downlink
queued under the old session. Watching over USB destroys the thing being
observed. Everything needed is already in sensor_logs.

WHAT TO LOOK FOR
  boot        climbing (2, 3, 4...) means deep sleep is retaining RTC.
              Stuck at 1 means the node reset or lost power.
  gap         seconds between uplinks. Should converge on the slot.
  into_slot   seconds past the slot boundary the uplink landed. This is
              the real measure of sync: both halves of a pair should sit
              near 0, separated by their transmit offsets (blower +0s,
              filter +4s). A value drifting all over the slot means the
              node is sleeping a fixed duration rather than to a boundary,
              i.e. it never got a time sync.

Usage:
  python service/check_lora_sync.py            # last 2 hours
  python service/check_lora_sync.py --hours 6
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.0)
    args = ap.parse_args()

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set in service/.env")
        return

    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    devices = [d for d in (sb.table("devices")
               .select("id, name, device_mac, duct_role, paired_device_id, "
                       "sample_slot_seconds, sample_slot_synced_at")
               .execute().data or []) if d.get("device_mac")]
    lora = [d for d in devices if d.get("sample_slot_seconds")]

    for d in sorted(lora, key=lambda x: x.get("name") or ""):
        slot = d["sample_slot_seconds"]
        rows = (sb.table("sensor_logs")
                .select("recorded_at, boot, battery")
                .eq("device_id", d["id"])
                .order("recorded_at", desc=True).limit(200).execute().data or [])
        rows = [r for r in rows
                if (isoparse(rows[0]["recorded_at"]) - isoparse(r["recorded_at"])).total_seconds()
                <= args.hours * 3600]
        rows.reverse()

        role = f" [{d['duct_role']}]" if d.get("duct_role") else ""
        print(f"\n=== {d.get('name') or d['device_mac']}{role} "
              f"slot={slot}s  synced_stamp={d.get('sample_slot_synced_at') or 'none'} ===")
        if not rows:
            print("  no uplinks in window")
            continue

        print(f"  {'TIME':22s} {'BOOT':>6s} {'GAP_S':>7s} {'INTO_SLOT':>10s}  BATT")
        prev = None
        for r in rows[-12:]:
            t = isoparse(r["recorded_at"])
            gap = f"{(t - prev).total_seconds():.0f}" if prev else "-"
            into = int(t.timestamp()) % slot
            batt = f"{float(r['battery']):.3f}" if r.get("battery") is not None else "-"
            print(f"  {t.strftime('%Y-%m-%d %H:%M:%S'):22s} {str(r['boot']):>6s} "
                  f"{gap:>7s} {into:>10d}  {batt}")
            prev = t

        boots = [r["boot"] for r in rows if r.get("boot") is not None]
        climbing = sum(1 for a, b in zip(boots, boots[1:]) if b > a)
        resets = sum(1 for a, b in zip(boots, boots[1:]) if b <= a)
        print(f"  -> boot advanced {climbing}x, reset/lost RTC {resets}x")
        if resets and not climbing:
            print("     RTC never survives a cycle: the node is being reset or "
                  "losing power. Time sync cannot persist until that stops.")

        intos = sorted(int(isoparse(r["recorded_at"]).timestamp()) % slot for r in rows)
        if len(intos) >= 3:
            # Circular spread, not max-min. into_slot wraps at the boundary, so
            # a node landing at 890s and 10s is 20s apart, not 880s -- but a
            # plain max-min reads that as nearly the whole slot. The previous
            # version tried to patch this with "or spread > slot - 60", which
            # then called a node scattered across the ENTIRE slot "aligned",
            # because a full scatter also has a large max-min. It reported P4
            # as ALIGNED while it was cycling every 69s.
            #
            # Correct measure: find the largest empty arc between consecutive
            # samples (including the wrap from last back to first); everything
            # else is the cluster, so cluster width = slot - largest gap.
            gaps = [b - a for a, b in zip(intos, intos[1:])]
            gaps.append(intos[0] + slot - intos[-1])   # wrap-around
            cluster_width = slot - max(gaps)
            aligned = cluster_width < 60
            print(f"     into-slot cluster width {cluster_width}s of {slot}s -> "
                  f"{'ALIGNED' if aligned else 'NOT aligned'}")
            if not aligned:
                med_gap = sorted(
                    (isoparse(b["recorded_at"]) - isoparse(a["recorded_at"])).total_seconds()
                    for a, b in zip(rows, rows[1:])
                )[len(rows) // 2] if len(rows) > 2 else None
                if med_gap:
                    print(f"     median cycle {med_gap:.0f}s vs slot {slot}s -- "
                          f"{'node never applied the slot' if abs(med_gap - slot) > 30 else 'close, drifting'}")


if __name__ == "__main__":
    main()
