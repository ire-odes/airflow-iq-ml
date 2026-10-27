"""
service/log_lora_mfcc.py
Keeps every MFCC uplink the LoRaWAN nodes send, one CSV row each.

    <out>/lora_mfcc.csv
        uploaded_at, device_mac, name, duct_role, mfcc0 .. mfcc12,
        battery, temp_c, humidity, pressure_pa, windSpeed

WHY THIS EXISTS. audio_logs holds one row per device and ingest_ttn
overwrites it on every uplink, so a LoRa node's MFCCs survive only until its
next packet -- the same trap as latest.wav on the WiFi side. Nothing can be
calibrated or scored without history, so this copies each set out as it
lands.

The values are stored exactly as audio_logs has them (the firmware's x100
int16 already divided back by ingest_ttn). They are the FIRMWARE's MFCCs:
26 mel bands, log10, unnormalised DCT, c0 included -- not the librosa MFCCs
extract_features_v2 computes from a wav. See firmware_mfcc.py.

Sensor columns come from the sensor_logs row nearest the uplink, so a reading
can later be gated on windSpeed or checked against battery.

Usage:
  python service/log_lora_mfcc.py                    # all LoRa nodes, 7 days
  python service/log_lora_mfcc.py --hours 48 --interval 20
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

from dateutil.parser import isoparse
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "lora_mfcc"
SENSOR_COLS = ["battery", "temp_c", "humidity", "pressure_pa", "windSpeed"]

# A sensor row further than this from the uplink belongs to a different wake.
SENSOR_MATCH_S = 120


def nearest_sensor_row(sb, device_id, uploaded_at):
    t = isoparse(uploaded_at)
    rows = (sb.table("sensor_logs").select("recorded_at," + ",".join(SENSOR_COLS))
            .eq("device_id", device_id).order("recorded_at", desc=True).limit(5)
            .execute().data or [])
    best = min(rows, key=lambda r: abs((isoparse(r["recorded_at"]) - t).total_seconds()),
               default=None)
    if best and abs((isoparse(best["recorded_at"]) - t).total_seconds()) <= SENSOR_MATCH_S:
        return best
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--hours", type=float, default=24 * 7)
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "lora_mfcc.csv"
    new_file = not path.exists()

    # Resume without duplicating: an uplink already in the file is skipped.
    seen = {}
    if not new_file:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seen[row["device_mac"]] = row["uploaded_at"]

    devices = {d["device_mac"]: d for d in
               sb.table("devices").select("id,name,device_mac,duct_role")
               .eq("is_lorawan", True).execute().data or []}
    print(f"{len(devices)} LoRa node(s) -> {path}")

    deadline = time.time() + args.hours * 3600
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["uploaded_at", "device_mac", "name", "duct_role"]
                       + [f"mfcc{i}" for i in range(13)] + SENSOR_COLS)
            fh.flush()

        while time.time() < deadline:
            try:
                rows = (sb.table("audio_logs").select("device_id,updated_at,mfcc_coefficients")
                        .in_("device_id", list(devices)).execute().data or [])
                for r in rows:
                    mac, up, mfcc = r["device_id"], r["updated_at"], r["mfcc_coefficients"]
                    if not mfcc or len(mfcc) != 13 or seen.get(mac) == up:
                        continue
                    d = devices[mac]
                    s = nearest_sensor_row(sb, d["id"], up)
                    w.writerow([up, mac, d["name"], d["duct_role"]] + list(mfcc)
                               + [s.get(c) for c in SENSOR_COLS])
                    fh.flush()
                    seen[mac] = up
                    zero = " (ALL ZERO: mic never started)" if not any(mfcc) else ""
                    print(f"  {isoparse(up).astimezone():%m-%d %H:%M:%S}  {d['name']:<16} "
                          f"c0 {mfcc[0]:+8.2f}{zero}")
            except Exception as e:                      # keep collecting overnight
                print(f"  error ({type(e).__name__}: {e}) -- retrying")
            time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
