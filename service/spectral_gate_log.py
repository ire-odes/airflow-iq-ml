"""
service/spectral_gate_log.py
Logs the spectral "gain" of every clip a device uploads, so a silence gate
can be set from measured behaviour instead of a guess.

WHY GAIN AND NOT LOUDNESS. The RMS silence gate in extract_features_v2 has
never rejected anything: recorded level is near-identical whether the blower
runs or not (IdleHVAC 0.333 vs running 0.340), because the hardware
normalises it. The spectrum's dB level does separate them -- on 120 labelled
clips, fan-off medians top out at 1.9 dB and fan-on bottoms out at 3.2 dB.

Storage keeps only {MAC}/latest.wav, overwritten on each upload, so history
only exists if something records it while the device is awake. That is this
script: poll, and when the file changes, log one row.

Columns: recorded_at, device, median_db, pct_bands_below_0, max_db, rms,
centroid_hz, passes_gate.

Usage:
  python service/spectral_gate_log.py --device E4B3238FAAE0 --hours 14
  python service/spectral_gate_log.py --device E4B3238FAAE0 58E6C5204B28 --threshold 2.5
"""
import argparse
import csv
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent / ".env")

import librosa                                    # noqa: E402
from extract_features_v2 import SR                # noqa: E402
from spectrum_viz import compute_spectrum         # noqa: E402

URL = os.environ["SUPABASE_URL"]
KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
HEADERS = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
BUCKET = "hvac-recordings"

# Mid-gap between the labelled corpora (fan-off max 1.9 dB, fan-on min 3.2).
DEFAULT_THRESHOLD = 2.5


def latest_updated_at(mac):
    r = requests.post(f"{URL}/storage/v1/object/list/{BUCKET}",
                      headers={**HEADERS, "Content-Type": "application/json"},
                      json={"prefix": f"{mac}/", "search": "latest.wav", "limit": 1},
                      timeout=30)
    if r.status_code != 200:
        return None
    files = r.json()
    return files[0].get("updated_at") if files else None


def measure(mac):
    r = requests.get(f"{URL}/storage/v1/object/{BUCKET}/{mac}/latest.wav",
                     headers=HEADERS, timeout=60)
    if r.status_code != 200:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(r.content)
        path = f.name
    try:
        y, _ = librosa.load(path, sr=SR)
    finally:
        os.unlink(path)
    if len(y) < SR * 0.5:
        return None
    spec = np.asarray(compute_spectrum(y, SR), dtype=float)
    return {
        "median_db": float(np.median(spec)),
        "pct_below_0": float(100 * (spec < 0).mean()),
        "max_db": float(spec.max()),
        "rms": float(np.sqrt(np.mean(y ** 2))),
        "centroid_hz": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR)[0])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", nargs="+", required=True, metavar="MAC")
    ap.add_argument("--hours", type=float, default=14)
    ap.add_argument("--interval", type=int, default=45)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "spectral_gate_log.csv"))
    args = ap.parse_args()

    macs = [m.strip().replace(":", "").upper() for m in args.device]
    seen = {m: None for m in macs}
    deadline = time.time() + args.hours * 3600
    new_file = not Path(args.out).exists()

    with open(args.out, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["logged_at", "device", "uploaded_at", "median_db",
                        "pct_bands_below_0", "max_db", "rms", "centroid_hz", "passes_gate"])
            fh.flush()
        print(f"logging {', '.join(macs)} -> {args.out}  (gate {args.threshold} dB)")

        counts = {m: [0, 0] for m in macs}          # [total, passed]
        while time.time() < deadline:
            for mac in macs:
                try:
                    up = latest_updated_at(mac)
                    if not up or up == seen[mac]:
                        continue
                    m = measure(mac)
                    seen[mac] = up
                    if not m:
                        continue
                    passes = m["median_db"] > args.threshold
                    counts[mac][0] += 1
                    counts[mac][1] += int(passes)
                    w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                mac, up, f"{m['median_db']:.2f}", f"{m['pct_below_0']:.1f}",
                                f"{m['max_db']:.2f}", f"{m['rms']:.4f}",
                                f"{m['centroid_hz']:.1f}", int(passes)])
                    fh.flush()
                    tot, ok = counts[mac]
                    print(f"  {mac} {m['median_db']:+7.2f} dB  centroid {m['centroid_hz']:6.0f} Hz  "
                          f"{'PASS' if passes else 'skip'}   ({ok}/{tot} passing)")
                except requests.exceptions.RequestException as e:
                    print(f"  {mac}: network error ({type(e).__name__})")
            time.sleep(args.interval)

    for mac in macs:
        tot, ok = counts[mac]
        pct = (100 * ok / tot) if tot else 0
        print(f"{mac}: {ok}/{tot} clips passed the {args.threshold} dB gate ({pct:.0f}%)")


if __name__ == "__main__":
    main()
