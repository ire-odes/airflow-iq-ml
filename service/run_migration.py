"""
service/run_migration.py
Applies .sql migration files to the live Supabase Postgres database via
the Supabase Management API's query endpoint -- the same thing you'd do
by pasting into the Dashboard SQL Editor, just scriptable.

WHY THE MANAGEMENT API AND NOT A DIRECT POSTGRES CONNECTION:
this project's direct host (db.<ref>.supabase.co) is IPv6-only and does
not resolve from this machine, and every aws-0-/aws-1-<region> Supavisor
pooler hostname rejects the tenant, so psycopg2 has no reachable route.
POST /v1/projects/{ref}/database/query runs arbitrary SQL over plain
HTTPS and works from anywhere -- it's also what the Supabase MCP server
uses under the hood.

Needs a Supabase personal access token (Dashboard -> account settings ->
Access Tokens). Supply it either as SUPABASE_ACCESS_TOKEN in the
environment, or in service/.env as:

    SUPABASE_ACCESS_TOKEN=sbp_xxxxxxxxxxxx

PROJECT_REF is read from SUPABASE_URL in service/.env when not given.

Usage:
  python service/run_migration.py <path/to/migration.sql> [more.sql ...]
  python service/run_migration.py --list       # show known migration files
  python service/run_migration.py --check      # connection smoke test
  python service/run_migration.py --sql "select 1"   # ad-hoc query

Each file is sent as a single request, so Postgres runs it as one
implicit transaction: if any statement in the file fails, the whole file
rolls back and nothing partial is left behind.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

ACCESS_TOKEN = os.environ.get("SUPABASE_ACCESS_TOKEN")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")

# https://<ref>.supabase.co -> <ref>
_m = re.search(r"https://([a-z0-9]+)\.supabase\.co", SUPABASE_URL)
PROJECT_REF = os.environ.get("SUPABASE_PROJECT_REF") or (_m.group(1) if _m else None)

API = "https://api.supabase.com/v1/projects/{ref}/database/query"

# The app repo's migrations live outside this repo; this is the checkout
# used during development. Override with --migrations-dir if it moves.
DEFAULT_MIGRATIONS_DIR = Path(
    r"C:\Users\OTC~1.STU\AppData\Local\Temp\claude"
    r"\c--Users-otc-student-Desktop-ML"
    r"\7de25362-d77d-4a18-a52c-eed4f618955b\scratchpad"
    r"\airflow-iq-mobile\supabase\migrations"
)


def require_config():
    if not ACCESS_TOKEN:
        print("SUPABASE_ACCESS_TOKEN not set.\n\n"
              "Create one at Supabase Dashboard -> account settings ->\n"
              "Access Tokens, then add it to service/.env as:\n\n"
              "  SUPABASE_ACCESS_TOKEN=sbp_xxxxxxxxxxxx\n")
        sys.exit(1)
    if not PROJECT_REF:
        print("Could not determine the project ref -- set SUPABASE_PROJECT_REF\n"
              "in service/.env, or make sure SUPABASE_URL is set there.")
        sys.exit(1)


def run_sql(sql):
    """Returns the decoded JSON result rows, or raises with the API's error."""
    resp = httpx.post(
        API.format(ref=PROJECT_REF),
        headers={"Authorization": f"Bearer {ACCESS_TOKEN}",
                 "Content-Type": "application/json"},
        json={"query": sql},
        timeout=120,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}")
    try:
        return resp.json()
    except json.JSONDecodeError:
        return resp.text


def apply_file(path):
    sql = Path(path).read_text(encoding="utf-8")
    print(f"\n=== {Path(path).name} ({len(sql)} chars) ===")
    result = run_sql(sql)
    print(f"  applied OK  -> {json.dumps(result)[:200]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help=".sql file(s) to apply, in order")
    parser.add_argument("--list", action="store_true", help="list known migration files")
    parser.add_argument("--check", action="store_true", help="connection smoke test only")
    parser.add_argument("--sql", default=None, help="run one ad-hoc SQL statement")
    parser.add_argument("--migrations-dir", default=str(DEFAULT_MIGRATIONS_DIR))
    args = parser.parse_args()

    if args.list:
        d = Path(args.migrations_dir)
        if not d.exists():
            print(f"No such directory: {d}")
            return
        for f in sorted(d.glob("*.sql")):
            print(f)
        return

    require_config()

    if args.check:
        print(f"project: {PROJECT_REF}")
        print(run_sql("select current_database() as db, current_user as usr, version() as ver;"))
        return

    if args.sql:
        print(json.dumps(run_sql(args.sql), indent=2)[:4000])
        return

    if not args.files:
        parser.error("pass at least one .sql file, or --list / --check / --sql")

    for f in args.files:
        apply_file(f)
    print("\nAll files applied.")


if __name__ == "__main__":
    main()
