"""
service/capture_mic_serial.py
Captures the clip P5_mic_test_v2.txt sends down the serial port and writes it
out as a playable .wav. No network involved at either end -- use this when the
device cannot reach WiFi.

The sketch (with USE_WIFI 0) prints the WAV as base64 between two markers:

    ---BEGIN WAV--- 96044 bytes  4812339 checksum  ws14
    UklGRiR...
    ---END WAV---

Everything else the sketch prints -- the sweep result, the live meter -- is
echoed to your terminal as it arrives, so this doubles as the serial monitor.
Close the Arduino IDE's monitor first: only one program can hold a COM port.

Usage:
  venv\\Scripts\\python.exe service\\capture_mic_serial.py
  venv\\Scripts\\python.exe service\\capture_mic_serial.py --port COM7
  venv\\Scripts\\python.exe service\\capture_mic_serial.py --out "C:/somewhere"
  venv\\Scripts\\python.exe service\\capture_mic_serial.py --once   # exit after one clip
"""
import argparse
import array
import base64
import re
import sys
from datetime import datetime
from pathlib import Path

import serial                                    # pip install pyserial
from serial.tools import list_ports

BEGIN = re.compile(r"---BEGIN WAV---\s+(\d+) bytes\s+(\d+) checksum\s+ws(\d+)")
END = "---END WAV---"
B64_LINE = re.compile(r"^[A-Za-z0-9+/=]+$")

# Which board revision the clip's word-select pin corresponds to.
MAPS = {7: "rev 0 newer (ws7)", 14: "rev 1 older (ws14)"}


def pick_port(requested):
    """The device's port. A single obvious candidate is chosen automatically;
    anything ambiguous is the user's call rather than a guess."""
    ports = list(list_ports.comports())
    if requested:
        return requested
    # COM1 is almost always a motherboard serial port, not a USB device.
    usb = [p for p in ports if p.device.upper() != "COM1"]
    if len(usb) == 1:
        print(f"Using {usb[0].device}  ({usb[0].description})")
        return usb[0].device
    print("Cannot pick a port automatically. Available:")
    for p in ports:
        print(f"   {p.device:<8} {p.description}")
    if not ports:
        print("   (none -- is the board plugged in?)")
    print("\nRe-run with --port COMn")
    return None


def analyse(wav_bytes):
    """Peak-to-peak, rms and dc of a 16-bit mono WAV already in memory.
    Stdlib only, so this runs anywhere pyserial does -- and it reads the PCM
    directly rather than through an audio library, because it has to cope with
    a degenerate file, which is exactly the case we are chasing."""
    pcm = array.array("h")
    pcm.frombytes(wav_bytes[44:len(wav_bytes) - (len(wav_bytes) - 44) % 2])
    if sys.byteorder == "big":                   # the WAV is little-endian
        pcm.byteswap()
    if not pcm:
        return 0, 0.0, 0.0, "EMPTY -- no samples in the file"
    pp = max(pcm) - min(pcm)
    dc = sum(pcm) / len(pcm)
    rms = (sum(float(v) * v for v in pcm) / len(pcm)) ** 0.5
    verdict = ("NO SIGNAL -- mic dead, mis-wired, or unpowered" if pp < 32 else
               "stuck line -- silent but offset" if rms < 20 and abs(dc) > 200 else
               "working")
    return pp, rms, dc, verdict


def save(payload, expect_len, expect_sum, ws, out):
    try:
        raw = base64.b64decode("".join(payload), validate=True)
    except Exception as e:                       # noqa: BLE001 - report, don't crash the capture
        print(f"  !! base64 did not decode ({e}) -- press d on the device to resend")
        return

    if len(raw) != expect_len:
        print(f"  !! length mismatch: got {len(raw)}, expected {expect_len} "
              f"-- press d to resend")
        return
    got_sum = sum(raw)
    if got_sum != expect_sum:
        print(f"  !! checksum mismatch: got {got_sum}, expected {expect_sum} "
              f"-- press d to resend")
        return

    out.mkdir(parents=True, exist_ok=True)
    name = f"mictest_ws{ws}_{datetime.now():%Y%m%d_%H%M%S}.wav"
    dest = out / name
    dest.write_bytes(raw)

    pp, rms, dc, verdict = analyse(raw)
    print(f"\n  saved {dest}")
    print(f"  map {MAPS.get(ws, 'unknown'):<20} peak-to-peak {pp:<8} "
          f"rms {rms:<10.1f} dc {dc:<9.1f} {verdict}\n")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="COM port; auto-detected when unambiguous")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", default=str(Path.home() / "Desktop" / "mic test clips"))
    ap.add_argument("--once", action="store_true", help="exit after the first clip")
    args = ap.parse_args()

    port = pick_port(args.port)
    if not port:
        return 1

    out = Path(args.out)
    try:
        ser = serial.Serial(port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open {port}: {e}")
        print("If the Arduino IDE's serial monitor is open, close it and retry.")
        return 1

    print(f"Listening on {port} at {args.baud}. Ctrl+C to stop.")
    print("Device keys: r = record, d = resend last clip, s = re-run the sweep.\n")

    payload, expect = [], None
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue

            m = BEGIN.search(line)
            if m:
                expect = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                payload = []
                print(f"  receiving {expect[0]} bytes from ws{expect[2]} ...", flush=True)
                continue

            if expect and END in line:
                dest = save(payload, *expect, out=out)
                expect, payload = None, []
                if dest and args.once:
                    return 0
                continue

            if expect is not None:
                # Inside a transfer: base64 goes to the buffer, anything else is
                # the sketch talking and gets echoed.
                if B64_LINE.match(line):
                    payload.append(line)
                else:
                    print(f"   {line}")
                continue

            print(line)                         # ordinary serial monitor output
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
