"""
service/fetch_mic_test.py
Downloads the clips P5_mic_test_v2.txt uploads, so you can listen to them.

The test sketch writes to hvac-recordings/MICTEST/<MAC>_ws<pin>_<n>.wav and
never touches audio_logs, so nothing it records reaches the ML pipeline or any
device baseline. This pulls those clips to your Desktop and reports what is in
each one, so a dead mic is obvious without opening the file.

The word-select pin is carried in the filename, which is what tells the two
board revisions apart: ws7 is the newer map, ws14 the older one. A clip that
reads NO SIGNAL on one pin and fine on the other means the pin map was wrong,
not the microphone.

Usage:
  python service\\fetch_mic_test.py                     # all test clips
  python service\\fetch_mic_test.py --device E4B3238E6B10
  python service\\fetch_mic_test.py --out "C:/somewhere/else"
  python service\\fetch_mic_test.py --delete            # clear them afterwards
"""
import argparse
import re
import os
import sys
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(Path(__file__).resolve().parent / ".env")

import librosa                                    # noqa: E402
from extract_features_v2 import SR                # noqa: E402

URL = os.environ["SUPABASE_URL"]
KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
HEADERS = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
BUCKET = "hvac-recordings"
PREFIX = "MICTEST"

# Which board revision a clip's word-select pin corresponds to.
MAPS = {7: "rev 0 newer (ws7)", 14: "rev 1 older (ws14)"}


def ws_pin(name):
    """The word-select pin the sketch recorded on, from <MAC>_ws<pin>_<n>.wav.
    Clips from the v1 sketch carry no pin, and come back as None."""
    m = re.search(r"_ws(\d+)_", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="only this MAC")
    ap.add_argument("--out", default=str(Path.home() / "Desktop" / "mic test clips"))
    ap.add_argument("--delete", action="store_true", help="remove them from Storage after downloading")
    args = ap.parse_args()

    r = requests.post(f"{URL}/storage/v1/object/list/{BUCKET}",
                      headers={**HEADERS, "Content-Type": "application/json"},
                      json={"prefix": f"{PREFIX}/", "limit": 200,
                            "sortBy": {"column": "updated_at", "order": "desc"}},
                      timeout=30)
    r.raise_for_status()
    files = [f for f in r.json() if f.get("name", "").endswith(".wav")]
    if args.device:
        files = [f for f in files if f["name"].startswith(args.device)]
    if not files:
        print("No test clips in Storage yet. Flash P5_mic_test_v2.txt and let it record one.")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} clip(s) -> {out}\n")
    print(f"  {'file':<34}{'map':<22}{'seconds':>8}{'peak-to-peak':>14}"
          f"{'rms':>9}{'dc':>8}   verdict")

    seen_maps = {}
    for f in files:
        name = f["name"]
        wav = requests.get(f"{URL}/storage/v1/object/{BUCKET}/{PREFIX}/{name}",
                           headers=HEADERS, timeout=60)
        if wav.status_code != 200:
            print(f"  {name:<34}  download failed ({wav.status_code})")
            continue
        dest = out / name
        dest.write_bytes(wav.content)

        y, _ = librosa.load(str(dest), sr=SR)
        pcm = np.round(y * 32768).astype(int)
        pp, rms, dc = int(pcm.max() - pcm.min()), float(np.sqrt(np.mean(pcm.astype(float) ** 2))), float(pcm.mean())
        # Same reading as the sketch's own serial output, so the two agree.
        verdict = ("NO SIGNAL -- mic dead, mis-wired, or unpowered" if pp < 32 else
                   "stuck line -- silent but offset" if rms < 20 and abs(dc) > 200 else
                   "working")
        label = MAPS.get(ws_pin(name), "unknown map")
        seen_maps.setdefault(label, []).append(pp)
        print(f"  {name:<34}{label:<22}{len(y)/SR:>8.1f}{pp:>14}"
              f"{rms:>9.1f}{dc:>8.1f}   {verdict}")

        if args.delete:
            requests.delete(f"{URL}/storage/v1/object/{BUCKET}/{PREFIX}/{name}",
                            headers=HEADERS, timeout=30)

    # With clips from both maps in hand the diagnosis is a comparison, not a
    # judgement call: one live and one dead is a pin-map fault, both dead is
    # the microphone itself.
    live = {m for m, pps in seen_maps.items() if max(pps) >= 32}
    if len(seen_maps) > 1:
        if len(live) == 1:
            print(f"\n  Only {live.pop()} produced signal -> the pin map was wrong, "
                  f"not the mic.")
        elif not live:
            print("\n  Neither map produced signal -> the mic itself is unpowered, "
                  "mis-wired or dead.")

    print(f"\nOpen the folder to listen: {out}")
    if args.delete:
        print("Removed from Storage.")


if __name__ == "__main__":
    main()
