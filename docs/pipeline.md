# How AirFlow IQ decides when to listen and what it hears

*Wake intervals and the ML baseline pipeline, as running on 18 September 2026.*

A device wakes, records a clip, uploads it, asks the server how long to sleep,
and sleeps. Everything interesting happens in two places: a database function
that answers "how long to sleep", and a poller on the office desktop that turns
each clip into a verdict. They are coupled on purpose -- the poller's verdict
feeds the next sleep.

```
 device wakes ──► records clip ──► uploads ──► asks effective_wake_seconds ──► sleeps
                                       │                     ▲
                                       ▼                     │ sets blower_burst_until
                                 ML poller (desktop) ────────┘
                                       │
                                       ▼
                       verdict in filter_ml_readings ──► web & mobile apps
```

---

## 1. Wake intervals

### The number a device sleeps for

Devices never read `wake_interval_seconds` directly. They request the computed
column **`effective_wake_seconds`**, which resolves three tiers in order -- the
first that applies wins:

| # | Tier | Applies when | Sleeps |
|---|---|---|---|
| 1 | **Calibration** | `calibration_started_at` is under 6 h old **and** the baseline isn't `warm` yet | **60 s** |
| 2 | **Blower burst** | `blower_burst_until` is in the future | **60 s** |
| 3 | **Normal** | otherwise | `wake_interval_seconds` |

Calibration outranks the burst so a recalibration is never slowed down by a
quiet blower. Both fast tiers are 60 s; they differ in *why* the device is awake.

**Why the logic lives in Postgres and not the firmware.** The ESP32 has no wall
clock -- no NTP, no RTC, and `millis()` resets on every deep sleep -- so it cannot
evaluate "has it been six hours since calibration started". Postgres can. The
firmware treats the answer as an opaque number of seconds, which is what lets
new tiers like the burst ship without reflashing anything.

### The normal interval

- **Fleet default: 4 hours** (14400 s, the column default).
- **Allowed range: 10 min – 24 h** (`devices_wake_interval_seconds_range` CHECKs
  600 – 86400). A 2-minute normal interval is impossible without a migration;
  anything faster than 10 minutes has to come from tier 1 or 2.
- **Only admins can change it.** The `guard_wake_interval` trigger rejects any
  change from an app user who isn't an admin, and rejects any value but 14400
  on insert. The service role and direct SQL bypass it.

Current per-device settings (everything else is on the 4 h default):

| Device | Interval | Why |
|---|---|---|
| P3 | 10 min | burst-mode test unit |
| P6 | 10 min | quiet house; denser sampling to catch the blower |
| P11 (rig) | 10 min | its firmware ignores tier 1 -- see *Known issues* |
| P5 – 135 Shore | 1 h | recalibration requested 17 Sep |
| P2 | 24 h | requested 11 Sep |

### Blower burst (new, 17 Sep)

A filter is only audible while air moves through it, and most of a day is
silence. Sampling on a fixed timer spends battery evenly across that silence
and collects little that's usable -- P6 sat at 23 of 120 warmup samples for
thirteen hours because its blower was off.

The burst inverts that: sleep long while it's quiet, and drop to 60 s the moment
a clip shows the blower running. The poller writes the deadline:

- clip **passes** the silence gate → `blower_burst_until = now() + 3 min`
- clip **fails** the gate → `blower_burst_until = null`

Three minutes spans two missed uploads at a 60 s cadence, so one dropped packet
doesn't collapse the burst. It's a **deadline, not a flag**, deliberately: if
the poller dies, a flag would pin the device at 60 s until someone noticed and
flatten the battery. A deadline lapses on its own and the device drifts back
to its normal interval.

The burst is enabled per device in `BLOWER_BURST_MACS`
(`service/poll_and_infer.py`). **Currently P3 only.**

### The verdict handshake (P3 burst firmware only)

Standard firmware asks for its interval immediately after uploading, so it gets
an answer computed from the *previous* clip and runs one cycle behind. For
burst testing that lag matters, so P3's firmware waits for the verdict on the
clip it just sent:

1. Before uploading, read `device_baselines.last_processed_at`.
2. Upload the clip.
3. Poll that field every 4 s until it changes -- the poller stamps it on every
   clip it finishes, including silent ones, so a change means *this* clip is done.
4. Then read `effective_wake_seconds` and sleep.

The wait is capped at **45 s**, from measured upload-to-verdict latency over
993 readings:

| p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|
| 17 s | 26 s | 31 s | 32 s | 51 s |

98.7 % land inside 45 s and the loop exits early, so the typical cost is ~20 s.
The dominant term is the poller's 30 s loop -- shorten it and this shrinks too.
If the poller is down the wait times out and the device uses whatever interval
is current: one cycle of lag, never a stall.

**Cost:** a cycle goes from ~7 s awake to ~7 s plus up to 45 s. Fine on a
powered rig. **Don't ship it to battery units** -- build with
`WAIT_FOR_ML_VERDICT 0` and accept the lag, which costs nothing.

The same build also spends **2 minutes** reading and discarding mic audio on a
power-cycle boot (never on a deep-sleep wake), because the MEMS mic's filters
only converge while clocked and the first clip after power-up was unreliable.

---

## 2. The ML pipeline, clip by clip

The poller is `service/poll_and_infer.py`, run by the scheduled task
**Airflow IQ inference poller** every 30 s.

### Step 1 -- find new clips
Reads every `audio_logs` row (one per device). Skips LoRaWAN rows (they carry
MFCCs, not audio) and any clip whose `updated_at` it has already processed.

### Step 2 -- the silence gate
The question is *"was the blower running?"* -- and loudness can't answer it. The
mic normalises level, so a blower-off room and a running rig record at nearly
the same RMS (0.333 vs 0.340), and silent P3 clips were measured *louder* than
blower-on P6 clips.

The **spectrum's level** answers it. The poller builds a 72-bin log-spaced
spectrum (20 Hz – 8 kHz, the same curve the app draws) and takes its **median in
dB**. At or below **+2.5 dB**, the clip is treated as blower-off.

| Evidence | Median dB |
|---|---|
| P3, 65 clips, blower off | **−32.2** (−34.8 … −26.6) |
| P6, 23 clips, blower on | **+5.6** (+3.7 … +7.5) |
| FiltSure corpus, 4,655 clips, all on | +6.2 … +7.8 |

A 30 dB gap with nothing in it. The median, not the peak, because a door or a
voice lifts one band without the broadband hiss that means air is moving.

A gated clip produces **no reading and no baseline sample**, clears the burst,
and is marked processed. The web app shows the same decision as a small tag on
the Acoustic Data card: amber **Silent** or grey **Blower on**.

**Enabled fleet-wide since 17 Sep**, after P3 -- then ungated -- built a baseline
from 65 silent clips out of 67. A baseline fitted to a quiet room reports
*dirty* the moment air moves.

### Step 3 -- features
Five features per clip, chosen because none of them depend on amplitude:

`low_freq_energy_ratio` · `centroid_mean` · `rolloff_mean` · `flatness_mean` · `bandwidth_mean`

### Step 4 -- calibration (the baseline)
A new or recalibrated device starts in `cold_start`, moves to `warming` with its
first sample, and records every reading as **calibrating** -- no verdict. Each
gate-passing clip updates a running mean and covariance of the five features
(Welford's method, so nothing is stored per clip).

The baseline **freezes** (`warm`) only when *both* hold:

- **120 samples**, and
- **20 hours** between the first and latest sample.

The 20 h rule makes the baseline span a full day and night, so diurnal change
lands inside "normal" rather than reading as drift later. At 60 s wakes a device
reaches 120 samples in ~2 h of blower time; the span is usually what it waits for.

At freeze, the device gets **its own threshold**: the 99th percentile of the
distances its own warmup samples had from its own baseline, × 1.10. No fleet-wide
threshold, because rooms differ enormously.

`BASELINE_SPAN_OVERRIDE_SECONDS` can shorten the 20 h for one device during a
test. It is empty now; P3 used 16 h for the 17 Sep burst test.

### Step 5 -- scoring
Once warm, every gate-passing clip is scored:

1. **Mahalanobis distance** of its features from the baseline -- how unusual it
   is given how the features normally vary together.
2. **EWMA** of that distance (α = 0.15), so one odd clip can't raise an alarm.
3. Two independent alert rules; either one returns **dirty**:
   - **Step change:** the EWMA exceeds the threshold **3 readings in a row**.
     Catches a filter swap -- on 18 Sep it flagged a change on P3 about three
     minutes after it happened.
   - **Gradual loading:** at least **20 % (minimum 3) of the last 30 scored
     readings** exceed it. Catches slow clogging, where the EWMA crosses and
     falls back before reaching three in a row. It needs at least 12 readings,
     looks back at most 7 days, and **never counts readings from before the
     baseline froze** -- those were scored against a different baseline.

The window is counted in *readings*, not hours. Silent clips never become
readings, so a fixed time window would hold wildly different amounts of evidence
on a day the blower ran 2 hours versus 12.

### Step 6 -- the threshold keeps learning
After **50** readings past the freeze, the threshold is re-derived every
**6 hours** from the 99th percentile of the latest **500** readings × 1.10 -- so a
filter's slow, normal change over its life doesn't read as a fault. It's
clamped between **0.70×** and **1.25×** the calibration threshold so it can
neither ratchet up to swallow real clogging nor down into noise, and it
**freezes while a device is flagged dirty**, so a real fault can't teach the
threshold to accept itself.

### Step 7 -- persist
One row in `filter_ml_readings` (the verdict, plus the five features, kept so
any baseline can be audited later), the updated `device_baselines` state, and
`last_processed_at` -- which is also what P3's firmware waits on.

**One row per clip is enforced** by `filter_ml_readings_one_per_clip`. See below.

---

## 3. Recalibrating a device

```
venv\Scripts\python service\recalibrate_all.py --mac <MAC>          # dry run
venv\Scripts\python service\recalibrate_all.py --mac <MAC> --yes
```

It deletes the device's `device_baselines` row and stamps
`calibration_started_at = now()`, which starts tier 1 (60 s) for up to 6 hours.
Warmup samples are taken on trust, so **only recalibrate with a filter you
believe is clean installed** -- the baseline defines "normal".

---

## 4. Known issues and operating rules

**Run exactly one poller.** Until 18 Sep nothing stopped two pollers scoring the
same clip, and one had been started by hand alongside the scheduled task:
669 duplicate readings, back to 24 August, 541 of them on P3. Baselines were
intact -- both copies read the same prior state -- but the gradual-loading rule
counts rows, so doubled history halved its effective window. The duplicates are
gone (exported to `backups/duplicate_readings_2026-09-18.json` first) and the
unique index now turns a second poller into an obvious `duplicate key` error
instead of silent damage.

**The desktop must stay awake.** On 16 Sep it slept at 19:13; P3 kept bursting
until 21:43, and ~130 of its clips were overwritten in storage before anything
scored them. `latest.wav` holds only the newest clip.

**The gate can starve a quiet site.** A starved baseline is recoverable; a
poisoned one isn't -- that's the trade. But P5 (135 Shore) has **0 samples**
eighteen hours into its recalibration, and at 1 h sampling it will collect very
slowly. P6 has the same problem. Pairing those sites with the blower burst is
the natural fix once the P3 test has proved it.

**Mic placement matters more than anything in software.** In the FiltSure
study, a mic at the blower separated FPR 5 / 7 / 9 as 1.8 / 9.6 / 14.7 median
distance, stable across 200 random baseline splits. A mic at the filter
detected *that* a filter changed but couldn't rank the grades reliably.

**P11's firmware ignores tier 1.** It reads `wake_interval_seconds` rather than
`effective_wake_seconds`, so it never bursts during calibration. It's on 10
minutes as a workaround; the real fix is reflashing it.

**The firmware carries the service-role key.** The constant named
`SUPABASE_ANON_KEY` in the WiFi firmware decodes to `role: service_role` -- full
database access that bypasses row-level security, readable by anyone who dumps a
device's flash. It isn't in either git repository. Fixing it means moving device
writes to the anon key (with row-level-security policies or an edge function for
what devices need), reflashing, and then rotating the key -- in that order,
since rotating first would cut off every device and the poller at once.

**Minor:** `effective_wake_seconds` falls back to 600 when the interval is null,
while the column defaults to 14400. No row is ever null, so it's harmless, but
the two should agree.

---

## Reference

| Setting | Value | Where |
|---|---|---|
| Silence gate | median ≤ +2.5 dB → silent | `SPECTRAL_GATE_MIN_MEDIAN_DB` |
| Burst window | 3 min per blower-on clip | `BLOWER_BURST_MINUTES` |
| Burst devices | P3 | `BLOWER_BURST_MACS` |
| Warmup | 120 samples **and** 20 h | `MIN_BASELINE_SAMPLES`, `MIN_BASELINE_SPAN_SECONDS` |
| Threshold at freeze | p99 of warmup distances × 1.10 | `AUTO_THRESHOLD_*` |
| EWMA | α = 0.15 | `EWMA_ALPHA` |
| Step-change rule | 3 consecutive | `PATIENCE` |
| Gradual rule | ≥ 20 % (min 3) of last 30; ≥ 12 readings; ≤ 7 days | `RECENT_*` |
| Adaptive threshold | after 50, every 6 h, p99 of 500 × 1.10, clamp 0.70–1.25× | `ADAPTIVE_*` |
| Calibration burst | 60 s for up to 6 h | `effective_wake_seconds()` |
| Normal interval | 600 – 86400 s, default 14400 | `devices.wake_interval_seconds` |
| Verdict wait (P3) | poll 4 s, cap 45 s | firmware `ML_WAIT_MAX_MS` |

Migrations: `20260821120000_calibration_sampling_mode`,
`20260910000000_fixed_wake_interval`, `20260917000000_blower_burst_sampling`,
`20260918000000_one_reading_per_clip`.
