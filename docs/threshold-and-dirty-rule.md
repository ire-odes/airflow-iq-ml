# How the alert threshold is set, and what triggers a "dirty" rating

*Notes for patent counsel. Describes the system as implemented on 22 September 2026
(`ML/drift_detector.py`, `ML/service/poll_and_infer.py`). Named constants are the
values in force; each is a tunable parameter, not a fixed property of the method.*

---

## 0. Terms used below

**Reading.** One recorded audio clip that has been scored. Clips judged to have
been recorded while the blower was off are discarded before scoring and never
become readings, so they cannot contribute to any count described here.

**Feature vector.** Five scalars extracted per clip: low-frequency energy ratio,
spectral centroid, spectral rolloff, spectral flatness, spectral bandwidth. All
five are scale-invariant — none depends on recorded loudness.

**Baseline.** A per-device statistical model of that device's own normal sound:
the mean vector and covariance matrix of the five features, accumulated by
Welford's online algorithm and then frozen. A baseline freezes only when it holds
at least `MIN_BASELINE_SAMPLES = 120` readings spanning at least
`MIN_BASELINE_SPAN_SECONDS = 20 hours`, so that normal day-to-night variation is
inside the model rather than read as a fault later.

**Distance.** The Mahalanobis distance of a new reading's feature vector from
that device's frozen baseline — how unusual the reading is, measured in units of
how the five features normally vary *and co-vary*.

**Smoothed distance (EWMA).** An exponentially weighted moving average of
successive distances, `EWMA_ALPHA = 0.15`, initialised to the first distance
scored after the baseline freezes. Each new reading moves it 15% of the way
toward that reading's distance. Every rule below tests the smoothed value, never
a single raw distance.

---

## 1. The threshold is derived per device, from that device's own calibration

There is no fleet-wide threshold. Each device receives its own the moment its
baseline freezes, computed from that device's own calibration data.

**Why per device.** Baseline spreads differ across installations by more than
threefold. On 3 September 2026 a genuine filter change on one unit sat at the
94th percentile of its own baseline, while another unit's *clean* readings sat at
the 100th percentile of its baseline. One shared number cannot serve both.

**No post-calibration clean readings are required.** Once the covariance is
frozen, the distribution of the distance is determined, so the percentile cut is
available immediately.

### 1a. Empirical route (preferred)

Used whenever the stored calibration feature vectors are available and number at
least `max(20, 4 × number_of_features)`:

1. Compute the distance of every calibration sample from the finished baseline.
2. Take the `AUTO_THRESHOLD_QUANTILE = 0.99` quantile of those distances.
3. **Correct for in-sample bias.** Those samples helped place the mean and
   covariance they are scored against, so their distances are systematically
   smaller than a fresh reading's. In-sample, the scaled distance follows a
   Beta(d/2, (n−d−1)/2) distribution; multiplying by the ratio of the closed-form
   quantile to the in-sample quantile removes the bias. The effect is small
   (~2% at n=120, ~0.5% at n=400) but it is a bias, not noise, and it points
   toward a threshold that is too tight.
4. Multiply by a safety margin `AUTO_THRESHOLD_MARGIN = 1.10`.

This route assumes no particular distribution. A heavy tail in the calibration
data is itself the pathology worth catching, and an empirical quantile preserves
it where a fitted distribution would smooth it away.

### 1b. Closed-form route (fallback)

Used when the calibration vectors are unavailable. For a fresh observation scored
against a mean and covariance estimated from n samples of dimension d, Hotelling's
T-squared gives

```
D² · n(n − d) / (d(n + 1)(n − 1))  ~  F(d, n − d)
```

so the quantile follows from n and d alone, then × 1.10. This is slightly wider
than the chi-square limit it converges to, because estimating the covariance from
finite data is itself uncertain.

The closed form is always computed even when the empirical route is used, and
both are stored, so any threshold can be audited afterwards rather than appearing
as an unexplained number.

If n ≤ d + 1 no threshold is produced at all; the caller keeps its existing
value rather than being handed a fabricated one.

### 1c. Continuous re-derivation, with clamps

A threshold fitted in one 20-hour window would not still mean the same thing
months later, so it is re-derived as the device runs. Re-derivation occurs at
most once per `ADAPTIVE_UPDATE_HOURS = 6` window, on the first reading to fall
into a new window, and only when the baseline is warm:

- Source data: up to the newest `ADAPTIVE_WINDOW_READINGS = 500` smoothed values
  recorded **since the baseline froze**.
- Only values at or below the current threshold are used
  (`ADAPTIVE_MIN_READINGS = 50` of them minimum). *Exceedances are the thing being
  detected; they are not evidence of what normal looks like.*
- Proposal: the 99th percentile of those values × 1.10.
- **Clamped** to between `ADAPTIVE_MIN_FALL = 0.70` and
  `ADAPTIVE_MAX_RISE = 1.25` times an anchor.
- **The anchor is recomputed from the frozen baseline by the same route used at
  freeze — never read from the stored threshold.** Anchoring to the stored value
  would make each update the anchor for the next, letting the ceiling ratchet
  upward one step at a time, which is the runaway this clamp exists to prevent.
- **Suspended entirely while the device is flagged dirty**, so the readings that
  raised a flag cannot become the evidence that dissolves it.
- Changes smaller than 0.02 are discarded, to keep the stored history readable.

Net effect: the threshold follows slow environmental change and the gradual,
normal ageing of a filter, but cannot drift far enough to swallow a real fault,
nor tighten into noise, nor move at all once a fault is flagged.

---

## 2. What triggers a "dirty" rating

A device must be **warm** (baseline frozen) to receive any verdict. Before that
every reading is recorded as *calibrating* and no verdict is issued.

A reading **exceeds** when its smoothed distance is strictly greater than the
device's current threshold. Two independent rules run on every reading, and
**either one alone returns dirty**:

### Rule A — step change: 3 consecutive exceedances

`PATIENCE = 3`. A counter increments on each exceedance and **resets to zero on
any reading that does not exceed**, so the three must be consecutive. This catches
an abrupt change such as a filter being swapped.

*Observed example.* On 18 September 2026 a device's distance rose from 2.06 to
23.05 between two consecutive clips. The smoothed value crossed the threshold on
that first clip, and the device was flagged on the third consecutive exceedance,
about three minutes later.

### Rule B — gradual loading: a proportion of a recent window

Counts exceedances among the reading in hand plus up to the 29 preceding
readings, giving a window of at most `RECENT_WINDOW_READINGS = 30`.

- **The window is measured in readings, not in elapsed time.** An HVAC system is
  not always running, and clips recorded while it is off never become readings,
  so a fixed time window would hold a wildly variable amount of evidence — on a
  slow sampler it is unreachable, and it behaves differently on a day the blower
  ran two hours than on one where it ran twelve. A fixed count holds the
  statistical power constant however the duty cycle moves, and idle periods
  simply do not enter the window instead of quietly weakening it.
- **Readings scored against a superseded baseline are excluded.** Recalibration
  replaces the mean and covariance, so the same audio yields a different distance
  afterwards; an older smoothed value is a different measurement wearing the same
  name. *Without this boundary the rule misfired: on 3 September 2026 a unit froze
  a fresh baseline at 11:18 and was flagged dirty at 11:28 on 29 of 29 readings
  that all predated the new baseline — seven minutes before the filter was
  touched. A wall-clock age bound alone could not have prevented this; the
  boundary must be the baseline, not the calendar.*
- Readings older than `RECENT_WINDOW_MAX_AGE_HOURS = 168` (7 days) are also
  excluded, for a device whose blower runs so rarely that 30 readings would
  otherwise span weeks and straddle a filter change or a change of season.
- **No verdict from this rule until the window holds at least
  `RECENT_WINDOW_MIN = 12` readings.** Immediately after a freeze only a handful
  have been scored, and a proportion of a 5-reading window is not the same rule
  as a proportion of a 30-reading one.
- **Trigger count:** `max(RECENT_DIRTY_MIN = 3, ceil(RECENT_DIRTY_FRACTION = 0.20
  × window size))`. The fraction is taken of the window actually assembled, not
  of the nominal 30, so a partly filled window is judged by the same proportion
  rather than being effectively stricter while it fills.

| Window size | Exceedances needed |
|---|---|
| 12 (minimum) | 3 |
| 15 | 3 |
| 20 | 4 |
| 25 | 5 |
| 30 (full) | 6 |

This catches slow clogging, where the smoothed value crosses the threshold and
falls back repeatedly without ever reaching three in a row.

**Sizing evidence.** Rolling 30-reading windows were measured across a real
filter change: all 356 clean windows on one unit produced 0% dirty, the noisiest
clean behaviour anywhere in the fleet was one reading in thirty (3%), and dirty
windows reached 100%. The 20% trigger therefore sits roughly six times above the
noisiest clean behaviour on record and far below a genuine detection.

### The short answer on counts

**Three exceeding readings is the minimum that can ever produce a dirty rating** —
three consecutive under Rule A, or three within a 12-to-15-reading window under
Rule B. At a full 30-reading window, Rule B requires six. Because every rule
tests the smoothed value rather than an individual reading, no single clip can
produce a dirty rating regardless of how extreme it is.

The verdict records which rule fired, so a step change is distinguishable from
slow creep in the history rather than both landing as an undifferentiated
"dirty".

---

## 3. Features a drafter may wish to note as distinctive

1. A per-device alert threshold derived from that device's own calibration data
   at the moment its statistical baseline is frozen, requiring no post-calibration
   labelled or clean observations.
2. Correcting that empirical threshold for the in-sample bias arising because the
   calibration samples helped estimate the very mean and covariance they are
   scored against.
3. Continuous re-derivation of the threshold clamped against an anchor
   **recomputed from the frozen baseline** rather than the previous threshold,
   which prevents cumulative upward drift of the alert ceiling.
4. Suspending threshold adaptation while a device is flagged, so detected
   anomalies cannot be absorbed into the definition of normal.
5. Two parallel triggers on the same smoothed statistic — consecutive-exceedance
   and proportion-of-window — targeting step changes and gradual degradation
   respectively.
6. Defining the evaluation window in **number of scored readings rather than
   elapsed time**, where readings are produced only from clips in which the
   monitored equipment was determined to be running, so that idle periods neither
   dilute the window nor stall the rule.
7. Bounding that window at the baseline-replacement boundary, so measurements
   taken against a superseded model cannot contribute to a verdict.
