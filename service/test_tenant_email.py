"""
service/test_tenant_email.py
End-to-end test of the tenant/landlord email system: queue real emails
through the production pipeline and wait until Resend accepts them.

What it exercises, in order:
  1. send_tenant_test_email()   -- the same function the "Send test email"
                                   button calls, so a pass means the button's
                                   path works, not a lookalike.
  2. (--overdue-sample) a sample "filter_overdue" landlord email carrying a
     TEST acknowledgement link. The link is real and clickable; because the
     token is flagged is_test, pressing its button records nothing.
  3. Polls email_outbox until each queued row is 'sent' or 'failed'. The
     outbox cron runs every minute and only fires when something is pending
     (migration 20260904010000), so expect up to ~70s.
  4. (--overdue-sample) checks the test token describes as 'ready' without
     consuming it, so the emailed link still works when you click it.

Usage:
  python service/test_tenant_email.py                        # odesolaire@gmail.com
  python service/test_tenant_email.py --to someone@example.com
  python service/test_tenant_email.py --overdue-sample       # also the overdue email + link
Exit code 0 only if every queued email was sent.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_migration import run_sql  # noqa: E402  (Management API SQL runner)

DEFAULT_TO = "odesolaire@gmail.com"


def q(sql):
    return run_sql(sql)


def lit(s):
    return "'" + str(s).replace("'", "''") + "'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", default=DEFAULT_TO)
    ap.add_argument("--overdue-sample", action="store_true")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()
    to = args.to.strip().lower()

    start = q("select now()::text as t")[0]["t"]
    print(f"recipient: {to}")

    n = q(f"select public.send_tenant_test_email(null, {lit(to)}) as n")[0]["n"]
    print(f"[1] queued {n} tenant test email(s)")

    token = None
    if args.overdue_sample:
        owned = q(f"""
            select d.id, coalesce(d.name, d.hvac_location, 'Test unit') as name,
                   d.hvac_location, p.name as property
            from devices d
            join auth.users u on u.id = d.owner_id
            left join properties p on p.id = d.property_id
            where lower(u.email) = {lit(to)}
            order by d.created_at limit 1""")
        if not owned:
            print("[2] skipped: the recipient owns no device to attach a test link to")
        else:
            dev = owned[0]
            token = q(f"select public.issue_filter_ack_token({lit(dev['id'])}::uuid, "
                      f"{lit(to)}, now(), true) as t")[0]["t"]
            payload = {
                "device_name": dev["name"], "hvac_location": dev["hvac_location"],
                "property_name": dev["property"], "due_on": time.strftime("%Y-%m-%d"),
                "day_number": 1, "days_total": 5, "recipient_role": "landlord",
                "is_test": True, "ack_url": f"https://www.airfloiq.com/ack?token={token}",
            }
            q(f"select public.enqueue_email_to_address({lit(to)}, 'filter_overdue', "
              f"'TEST - Overdue HVAC filter: reminder 1 of 5', {lit(json.dumps(payload))}::jsonb)")
            print(f"[2] queued sample overdue email for '{dev['name']}' with a TEST acknowledgement link")

    print("[3] waiting for the outbox cron to send...")
    deadline = time.time() + args.timeout
    rows = []
    while time.time() < deadline:
        rows = q(f"""
            select template, status, coalesce(error, '') as error, sent_at::text as sent_at
            from email_outbox
            where lower(to_email) = {lit(to)} and created_at >= {lit(start)}::timestamptz
            order by created_at""")
        if rows and all(r["status"] in ("sent", "failed") for r in rows):
            break
        time.sleep(5)

    ok = bool(rows) and all(r["status"] == "sent" for r in rows)
    for r in rows:
        print(f"    {r['template']:<20} {r['status']:<8} {r['sent_at'] or ''} {r['error'][:160]}")
    if not rows or not all(r["status"] in ("sent", "failed") for r in rows):
        print("    still pending at timeout -- is the process-email-outbox cron running?")

    if token:
        d = q(f"select public.describe_filter_ack_token({lit(token)}) as d")[0]["d"]
        d = json.loads(d) if isinstance(d, str) else d
        link_ok = d.get("status") == "ready" and d.get("is_test")
        print(f"[4] test link describes as '{d.get('status')}' (unused, still clickable): "
              f"{'OK' if link_ok else 'UNEXPECTED'}")
        ok = ok and link_ok

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
