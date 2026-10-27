"""
service/set_ttn_secrets.py
Sets the TTN secrets on the Supabase Edge Functions runtime, with format
checks, so ingest_ttn can actually queue config downlinks.

READ THIS BEFORE TRUSTING WHAT --check PRINTS.
The Supabase Management API returns the SHA-256 DIGEST of each secret, never
its plaintext. On 2026-10-13 this was misread as the value itself, producing
a confident and wrong diagnosis: TTN_APP_ID "holds a 64-char hex string
instead of a slug". It did not -- sha256("afiqtest01") is exactly that hex
string, and the app id had been correct all along. Proof, cheap to repeat:
set TTN_REGION=nam1 and read it back; you get
a512c48648881c4becba70df688cd9b5cbb07b27353f1a43d14afea6eb426a49, which is
sha256("nam1").

So a secret CANNOT be format-checked after the fact. It can only be checked
by hashing a candidate and comparing digests, which is what --check now does.
Values are still validated on the way IN, where the plaintext is in hand.

The downlink URL these feed is

    https://{region}.cloud.thethings.network/api/v3/as/applications/
        {TTN_APP_ID}/devices/{devEui}/down/replace

so with those values every downlink POST fails, and the nodes never receive
the clock/slot/tx-offset config they can only get from that downlink. They
join, they uplink, and they sample unaligned forever.

Nothing validated the format when they were first set, which is the only
reason a 64-char hex string sat there unnoticed. Hence the checks below --
they refuse rather than warn, because a silently wrong secret here produces
no error anywhere in the system.

Usage:
  python service/set_ttn_secrets.py --check        # show current, validate
  python service/set_ttn_secrets.py --from-env     # read from service/.env
  python service/set_ttn_secrets.py \
      --app-id filtsure-nodes --region nam1 --api-key NNSXS....

Reads SUPABASE_ACCESS_TOKEN from service/.env (same token run_migration.py
uses). Secret VALUES are never printed; only whether they are set and whether
they pass validation.
"""
import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

ACCESS_TOKEN = os.environ.get("SUPABASE_ACCESS_TOKEN")
PROJECT_REF = os.environ.get("SUPABASE_PROJECT_REF") or (
    os.environ.get("SUPABASE_URL", "").split("//")[-1].split(".")[0] or None)

API = "https://api.supabase.com/v1/projects/{ref}/secrets"

# TTN v3 application id: a slug, not a hash.
APP_ID_RE = re.compile(r"^[a-z][a-z0-9-]{1,35}$")
# TTN v3 API keys are "NNSXS." followed by the key body.
API_KEY_RE = re.compile(r"^NNSXS\.[A-Z0-9._-]{20,}$", re.IGNORECASE)
REGIONS = {"nam1", "eu1", "au1"}


def validate(app_id=None, api_key=None, region=None):
    """List of complaints; empty means every supplied value is well formed."""
    bad = []
    if app_id is not None and not APP_ID_RE.match(app_id):
        bad.append(
            f"TTN_APP_ID {app_id!r} is not a TTN application id. Expected a "
            f"lowercase slug like 'filtsure-nodes' -- 2-36 chars of [a-z0-9-] "
            f"starting with a letter. Take it from the console URL: "
            f"console.cloud.thethings.network/applications/<THIS PART>.")
    if api_key is not None and not API_KEY_RE.match(api_key):
        bad.append(
            "TTN_API_KEY does not look like a TTN v3 API key: they always "
            "begin 'NNSXS.'. Create one under Application -> API keys with "
            "'Write downlink application traffic' rights.")
    if region is not None and region not in REGIONS:
        bad.append(f"TTN_REGION {region!r} is not one of {sorted(REGIONS)}. "
                   f"It is the cluster in your console URL's hostname.")
    return bad


def fetch_current():
    r = httpx.get(API.format(ref=PROJECT_REF),
                  headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}, timeout=60)
    r.raise_for_status()
    return {s["name"]: s.get("value") for s in r.json()}


def show_current():
    cur = fetch_current()
    print(f"project {PROJECT_REF}\n")
    for name in ("TTN_APP_ID", "TTN_API_KEY", "TTN_REGION", "TTN_CONFIG_FPORT"):
        val = cur.get(name)
        if val is None:
            default = {"TTN_REGION": "nam1", "TTN_CONFIG_FPORT": "10"}.get(name)
            note = f"NOT SET (defaults to {default!r})" if default else "NOT SET"
            print(f"  {name:<18} {note}")
            continue
        # `val` is a sha256 digest, not the secret. Compare against a hash of
        # what service/.env holds, so "is the right thing deployed?" is
        # answerable without ever seeing the deployed plaintext.
        local = os.environ.get("TTN_READ_KEY" if name == "TTN_API_KEY" else name)
        if local:
            match = hashlib.sha256(local.encode()).hexdigest() == val
            print(f"  {name:<18} digest {val[:12]}...  "
                  f"{'matches service/.env' if match else 'DIFFERS from service/.env'}")
        else:
            print(f"  {name:<18} digest {val[:12]}...  (no local value to compare)")
    return cur


def put(pairs):
    r = httpx.post(API.format(ref=PROJECT_REF),
                   headers={"Authorization": f"Bearer {ACCESS_TOKEN}",
                            "Content-Type": "application/json"},
                   json=[{"name": k, "value": v} for k, v in pairs.items()],
                   timeout=60)
    if r.status_code not in (200, 201):
        raise SystemExit(f"Supabase rejected the update: HTTP {r.status_code} "
                         f"{r.text[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="show and validate, change nothing")
    ap.add_argument("--from-env", action="store_true",
                    help="read TTN_APP_ID / TTN_API_KEY / TTN_REGION from service/.env")
    ap.add_argument("--app-id")
    ap.add_argument("--api-key")
    ap.add_argument("--region", default=None)
    args = ap.parse_args()

    if not ACCESS_TOKEN or not PROJECT_REF:
        raise SystemExit("SUPABASE_ACCESS_TOKEN / project ref missing from service/.env")

    if args.check or not (args.from_env or args.app_id or args.api_key or args.region):
        show_current()
        if not args.check:
            print("\nNothing changed. Pass --from-env, or --app-id/--api-key/--region.")
        return

    app_id = args.app_id or (os.environ.get("TTN_APP_ID") if args.from_env else None)
    api_key = args.api_key or (os.environ.get("TTN_API_KEY") if args.from_env else None)
    region = args.region or (os.environ.get("TTN_REGION") if args.from_env else None)

    if not any([app_id, api_key, region]):
        raise SystemExit("No values found. With --from-env, add TTN_APP_ID / "
                         "TTN_API_KEY / TTN_REGION to service/.env first.")

    bad = validate(app_id, api_key, region)
    if bad:
        print("Refusing to set these -- the format is wrong:\n")
        for b in bad:
            print(f"  * {b}\n")
        raise SystemExit(1)

    pairs = {k: v for k, v in (("TTN_APP_ID", app_id),
                               ("TTN_API_KEY", api_key),
                               ("TTN_REGION", region)) if v}
    put(pairs)
    print(f"Set: {', '.join(pairs)}\n")
    show_current()
    print("\nEdge Functions pick up new secrets on their next invocation. If a "
          "downlink still fails, redeploy ingest_ttn to force a fresh runtime.")


if __name__ == "__main__":
    main()
