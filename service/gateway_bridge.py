"""
service/gateway_bridge.py
The workstation half of the ESP gateway. Reads requests the gateway forwards
over USB serial, replays them against Supabase from this machine's wired
connection, and hands the reply back.

    field device --WiFi--> GATEWAY --USB serial--> THIS SCRIPT --> Supabase

WHY THE LOGIC LIVES HERE AND NOT ON THE GATEWAY
The gateway understands nothing about Supabase: it forwards method, path,
content-type, Prefer and body, and returns whatever status and bytes come
back. Everything that might need to change -- the host, auth, retries, which
paths are even allowed -- is in this file, so it changes without reflashing.

AUTH MOVES OFF THE DEVICES
The service_role key is injected here. Devices behind the gateway carry no
database credential in flash at all, which closes the "Known issues" item in
docs/ml-pipeline.md for every device that moves behind it.

WIRE FORMAT (see GATEWAY_FIRMWARE.txt)
    gateway -> here   >>REQ <seq> <method> <path> <ctype> <prefer> <len>
                      <base64 body>
                      <<END
    here -> gateway   >>RES <seq> <httpcode> <len>
                      <base64 body>
                      <<END
    "#" prefixes a human-readable log line in either direction.

Usage:
  venv\\Scripts\\python.exe service\\gateway_bridge.py
  venv\\Scripts\\python.exe service\\gateway_bridge.py --port COM7
  venv\\Scripts\\python.exe service\\gateway_bridge.py --dry-run   # log, don't send
"""
import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests
import serial
from dotenv import load_dotenv
from serial.tools import list_ports

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(Path(__file__).resolve().parent / ".env")

URL = os.environ["SUPABASE_URL"]
KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
BAUD = 921600                      # must match SERIAL_BAUD in the gateway

# Only these prefixes are ever relayed. A gateway on an open SoftAP is
# reachable by anything in radio range, so the blast radius of a stray or
# hostile request is bounded here rather than trusted to the device.
ALLOWED = (
    "/storage/v1/object/hvac-recordings/",
    "/rest/v1/audio_logs",
    "/rest/v1/rpc/device_wake_check",
    "/functions/v1/ingest_logs",
)


def pick_port(requested):
    ports = list(list_ports.comports())
    if requested:
        return requested
    usb = [p for p in ports if p.device.upper() != "COM1"]
    if len(usb) == 1:
        print(f"using {usb[0].device}  ({usb[0].description})")
        return usb[0].device
    print("Cannot pick a port automatically. Available:")
    for p in ports:
        print(f"   {p.device:<8} {p.description}")
    if not ports:
        print("   (none -- is the gateway plugged in?)")
    print("\nRe-run with --port COMn")
    return None


def relay(method, path, ctype, prefer, body, dry_run):
    """Replay one request against Supabase. Returns (status, response bytes).
    Never raises: a gateway left waiting would stall every device behind it,
    so a failure here has to come back as a status code."""
    if not path.startswith(ALLOWED):
        print(f"    REFUSED: {path} is not on the allow-list")
        return 403, b'{"error":"path not allowed by gateway_bridge"}'

    headers = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
    if ctype and ctype != "-":
        headers["Content-Type"] = ctype
        # Mirrors the firmware: storage objects are overwritten in place.
        if ctype == "audio/wav":
            headers["x-upsert"] = "true"
    if prefer and prefer != "-":
        headers["Prefer"] = prefer

    if dry_run:
        print(f"    DRY RUN: would {method} {path} ({len(body)} bytes)")
        return 200, b'{"dry_run":true}'

    try:
        r = requests.request(method, URL + path, headers=headers,
                             data=body if body else None, timeout=25)
        return r.status_code, r.content
    except Exception as e:                       # noqa: BLE001 - must not stall the gateway
        print(f"    upstream failed: {e}")
        return 599, b'{"error":"upstream request failed"}'


def send_response(ser, seq, code, body):
    b64 = base64.b64encode(body).decode()
    lines = [b64[i:i + 76] for i in range(0, len(b64), 76)]
    out = f">>RES {seq} {code} {len(body)}\n" + "".join(l + "\n" for l in lines) + "<<END\n"
    ser.write(out.encode())
    ser.flush()


def read_request(ser, header):
    """Collect the base64 body that follows a >>REQ line. Returns the bytes, or
    None if the frame did not terminate."""
    parts = header.split()
    if len(parts) != 7:
        print(f"  malformed REQ header: {header!r}")
        return None
    _, seq, method, path, ctype, prefer, length = parts
    seq, length = int(seq), int(length)

    chunks = []
    deadline = time.time() + 30
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line == "<<END":
            break
        if not line or line.startswith("#"):     # a gateway log line, not body
            continue
        chunks.append(line)
    else:
        print("  REQ frame never terminated")
        return None

    body = base64.b64decode("".join(chunks)) if chunks else b""
    if len(body) != length:
        print(f"  length mismatch: decoded {len(body)}, header said {length}")
    return seq, method, path, ctype, prefer, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="COM port of the gateway")
    ap.add_argument("--baud", type=int, default=BAUD)
    ap.add_argument("--dry-run", action="store_true",
                    help="log what would be sent without calling Supabase")
    args = ap.parse_args()

    port = pick_port(args.port)
    if not port:
        return 1
    try:
        ser = serial.Serial(port, args.baud, timeout=2)
    except serial.SerialException as e:
        print(f"Could not open {port}: {e}")
        print("If the Arduino serial monitor is open, close it and retry.")
        return 1

    print(f"bridge up on {port} at {args.baud} -> {URL}")
    print(f"relaying only: {', '.join(ALLOWED)}")
    print("Ctrl+C to stop.\n")

    served = 0
    try:
        while True:
            raw = ser.readline().decode("utf-8", errors="replace").strip()
            if not raw:
                continue
            if raw.startswith("#"):                  # gateway's own logging
                print(raw)
                continue
            if not raw.startswith(">>REQ "):
                continue

            got = read_request(ser, raw)
            if not got:
                continue
            seq, method, path, ctype, prefer, body = got

            t0 = time.time()
            code, resp = relay(method, path, ctype, prefer, body, args.dry_run)
            send_response(ser, seq, code, resp)
            served += 1
            ms = (time.time() - t0) * 1000
            flag = "" if 200 <= code < 300 else "   <-- "
            print(f"  #{seq} {method:<4} {path[:52]:<52} -> {code} "
                  f"{len(resp):>5}B {ms:>6.0f}ms{flag}")
            if not 200 <= code < 300 and resp:
                print(f"      {resp[:200].decode('utf-8', errors='replace')}")
    except KeyboardInterrupt:
        print(f"\nstopped after {served} request(s).")
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
