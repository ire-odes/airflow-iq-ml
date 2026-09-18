"""
service/save_audio_by_gate.py
Saves a device's uploaded clips to disk, sorted by whether the spectral
silence gate considers the blower to have been running.

    <out>/Not silent/recording_YYYYMMDD_HHMMSS_5.9dB.wav
    <out>/Silent/recording_YYYYMMDD_HHMMSS_-40.1dB.wav
    <out>/index.csv

WHY THIS EXISTS. Storage holds only {MAC}/latest.wav, overwritten on every
upload, so a clip survives only if something copies it out before the next
one lands. Sorting them as they arrive makes the gate auditable: you can
listen to what it kept and what it threw away, instead of taking the
threshold on trust.

THE GATE. Not loudness -- recorded level barely moves with blower state
because the hardware normalises it (IdleHVAC 0.333 RMS vs running 0.340).
It is the MEDIAN of the log-spaced spectrum in dB, the same curve the app's
acoustic chart draws. Across 120 labelled clips: blower-off tops out at
1.9 dB, blower-on bottoms out at 3.2 dB, so the default cut sits mid-gap.
A clip at or below the threshold is "Silent".

Filenames keep the recording_<date>_<time> prefix that extract_features_v2's
parse_session_id matches (it anchors at position 0), so these folders can be
fed straight into the feature pipeline. The dB reading is appended after it.

Usage:
  python service/save_audio_by_gate.py                       # P6, 24h
  python service/save_audio_by_gate.py --hours 48
  python service/save_audio_by_gate.py --device 58E6C5204B28 --out "C:/tmp/P3"
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
from dateutil.parser import isoparse
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

P6_MAC = "E4B3238FAAE0"
DEFAULT_OUT = r"C:\Users\otc.student\Desktop\P6 audio readings 916"
# Mid-gap between the labelled corpora; matches SPECTRAL_GATE_MIN_MEDIAN_DB
# in service/poll_and_infer.py, which is what the ML itself applies.
DEFAULT_THRESHOLD = 2.5


def latest_updated_at(mac):
    """updated_at of the device's latest.wav, or None. Uses the list API so
    an unchanged clip costs one small request instead of a whole download."""
    r = requests.post(f"{URL}/storage/v1/object/list/{BUCKET}",
                      headers={**HEADERS, "Content-Type": "application/json"},
                      json={"prefix": f"{mac}/", "search": "latest.wav", "limit": 1},
                      timeout=30)
    if r.status_code != 200:
        return None
    files = r.json()
    return files[0].get("updated_at") if files else None


def download(mac):
    r = requests.get(f"{URL}/storage/v1/object/{BUCKET}/{mac}/latest.wav",
                     headers=HEADERS, timeout=60)
    return r.content if r.status_code == 200 else None


def measure(wav_bytes):
    """(median_db, pct_bands_below_0, rms, centroid_hz) or None if unusable."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = f.name
    try:
        y, _ = librosa.load(path, sr=SR)
    finally:
        os.unlink(path)
    if len(y) < SR * 0.5:
        return None
    spec = np.asarray(compute_spectrum(y, SR), dtype=float)
    return (float(np.median(spec)), float(100 * (spec < 0).mean()),
            float(np.sqrt(np.mean(y ** 2))),
            float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR)[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=P6_MAC, metavar="MAC")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()

    mac = args.device.strip().replace(":", "").upper()
    root = Path(args.out)
    loud_dir = root / "Not silent"
    quiet_dir = root / "Silent"
    for d in (loud_dir, quiet_dir):
        d.mkdir(parents=True, exist_ok=True)

    index = root / "index.csv"
    new_index = not index.exists()
    seen = None
    kept = dropped = 0
    deadline = time.time() + args.hours * 3600

    print(f"{mac} -> {root}")
    print(f"gate: median spectrum > {args.threshold} dB = not silent")

    with open(index, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_index:
            w.writerow(["uploaded_at", "file", "verdict", "median_db",
                        "pct_bands_below_0", "rms", "centroid_hz"])
            fh.flush()

        while time.time() < deadline:
            try:
                up = latest_updated_at(mac)
                if up and up != seen:
                    wav = download(mac)
                    if wav:
                        seen = up
                        m = measure(wav)
                        if m:
                            median_db, pct_neg, rms, centroid = m
                            not_silent = median_db > args.threshold
                            # Named from the upload time, in local time, so the
                            # filename lines up with the device's own schedule.
                            stamp = isoparse(up).astimezone().strftime("%Y%m%d_%H%M%S")
                            name = f"recording_{stamp}_{median_db:.1f}dB.wav"
                            dest = (loud_dir if not_silent else quiet_dir) / name
                            # Written aside and renamed, so nothing downstream
                            # can pick up a half-written wav.
                            tmp = dest.with_suffix(".wav.part")
                            tmp.write_bytes(wav)
                            os.replace(tmp, dest)

                            kept += int(not_silent)
                            dropped += int(not not_silent)
                            w.writerow([up, name, "not silent" if not_silent else "silent",
                                        f"{median_db:.2f}", f"{pct_neg:.1f}",
                                        f"{rms:.4f}", f"{centroid:.1f}"])
                            fh.flush()
                            print(f"  {stamp}  {median_db:+7.2f} dB  centroid {centroid:6.0f} Hz  "
                                  f"-> {'Not silent' if not_silent else 'Silent'}   "
                                  f"({kept} kept / {dropped} silent)")
            except requests.exceptions.RequestException as e:
                print(f"  network error ({type(e).__name__}) -- retrying")
            except Exception as e:                      # keep collecting overnight
                print(f"  error on one clip: {e}")
            time.sleep(args.interval)

    total = kept + dropped
    pct = (100 * kept / total) if total else 0
    print(f"\n{mac}: {kept} not silent, {dropped} silent, of {total} clips ({pct:.0f}% kept)")


if __name__ == "__main__":
    main()
