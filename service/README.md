# AirFlow IQ inference service

Watches the Supabase project the mobile/web app already uses, runs the ML
pipeline against any new audio upload, and writes the verdict back. See
`poll_and_infer.py`'s module docstring for exactly what it reads/writes.

## One-time setup

1. **Apply the migration** to the live Supabase project (SQL Editor, or
   `supabase db push` if you use the CLI):
   `airflow-iq-mobile/supabase/migrations/20260731000000_ml_inference.sql`

2. **Get the service role key**: Supabase dashboard → your project →
   Settings → API keys → `service_role`. This bypasses Row Level Security —
   it belongs only in this service's `.env`, never in the app, never
   committed.

3. **Configure**:
   ```
   cd service
   copy .env.example .env
   ```
   Paste the service role key into `.env`.

4. Dependencies are already in the project's `venv` (`supabase`,
   `python-dotenv`, plus everything else the ML pipeline already needed).

## Run it once, manually, to check everything's wired up

```
cd c:\Users\otc.student\Desktop\ML
.\venv\Scripts\python.exe service\poll_and_infer.py
```

Expect one line per device in `audio_logs`, e.g.:
```
3 device(s) in audio_logs
  AA:BB:CC:DD:EE:FF: clean (decided_by=classifier, state=warming)
  11:22:33:44:55:66: no storage_path yet -- skipping
```
Re-running immediately with no new upload should skip every device (nothing
to do — confirms the `last_processed_at` guard is working).

## Schedule it: Windows Task Scheduler

Runs every 15 minutes, which is far more responsive than the devices'
4-hour upload cadence requires.

1. Open **Task Scheduler** → **Create Task** (not "Basic Task", so you get
   the repeat-every-N-minutes option below).
2. **General** tab: name it `AirFlow IQ inference poller`. Under "Security
   options," select **Run whether user is logged on or not** if you want
   it to keep running when you're logged out.
3. **Triggers** tab → New: **Begin the task: On a schedule** → **Daily**,
   recur every 1 day → check **Repeat task every: 15 minutes**, **for a
   duration of: Indefinitely**.
4. **Actions** tab → New:
   - Program/script: `C:\Users\otc.student\Desktop\ML\venv\Scripts\python.exe`
   - Add arguments: `service\poll_and_infer.py`
   - Start in: `C:\Users\otc.student\Desktop\ML`
5. Save. Test it immediately via right-click → **Run**, then check the
   Supabase dashboard (`filter_ml_readings` table) for a new row.

## Known limitation

LoRaWAN devices (`audio_logs.is_lora = true`) send MFCC coefficients
instead of a playable wav and aren't handled by this service yet — it
logs and skips them. Per the app code's own comment, no LoRa rows exist in
production yet, so this hasn't blocked anything so far.
