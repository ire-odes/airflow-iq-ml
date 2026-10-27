"""
gate_threshold.py
Learns each device's own blower-on/off boundary from the clips it has actually
recorded, instead of comparing every device against one fleet-wide dB value.

WHY THIS EXISTS
The gate statistic -- median of a 72-bin log spectrum -- separates blower-on
from blower-off extremely well. What does not transfer is the ABSOLUTE level.
Measured over 1232 labelled clips:

    set              label     p05    median     p95
    IdleHVAC         off      -4.6      -0.5    +1.5
    FPR5             on       +9.5     +11.4   +12.9
    FPR12            on       +8.1     +10.6   +12.1
    MERV1_CLEAN      on       +9.0     +10.5   +12.2
    P3 (Abraham)     off     -44.1     -36.9   -34.2
    P3 (Abraham)     on      -18.6     -17.7   -17.5

Within a site the two states are far apart. Across sites they overlap almost
completely: 309 of 312 off-clips sit ABOVE the quietest on-clip. The shipped
+2.5 dB cut is correct for the lab rig it was measured on and ~20 dB above
everything P3 can physically produce, which is why P3 recorded 82 consecutive
wakes that were all discarded as silent.

THE DEADLOCK THIS ALSO FIXES
The device-relative rule in poll_and_infer could have caught this, but it needs
20 reference clips and reference clips only accumulate from clips that PASS the
gate. Too high a cut therefore means: nothing passes -> no references -> the
relative rule never activates -> the cut keeps deciding. Forever. The fix is
that history here is appended for EVERY clip, pass or fail, so a device can
always learn its own boundary even while the gate is rejecting everything.

HOW THE BOUNDARY IS CHOSEN
Otsu's method: the split that maximises between-class variance. It was built
for exactly this -- two populations of unknown size on one axis -- and unlike a
largest-gap rule it still works when the two modes touch, which they do on the
lab rig (off tops out at +2.0, FPR12 starts at +1.9).

A split is only ACCEPTED when both of these hold, because Otsu will always
return something, even for a single population:
  - the two class means are at least SEP_MIN_DB apart, and
  - the smaller class holds at least MIN_MASS of the clips.
The mass test is what stops a skewed single-state distribution from having its
own tail declared the other mode. IdleHVAC alone has a tail reaching -34 dB
against a median of -0.5; without the mass test that tail looks bimodal.

When a split is refused the device has only ever seen one state, and guessing
where the other one lies would be exactly the mistake the +2.5 dB cut made. It
falls back to "louder than this device's own quiet floor by a margin", which
makes no claim about the absent mode. That fallback is not a consolation prize:
on P3's off-only history it yields -31.9 dB, which still correctly admits P3's
real blower-on clips at -17.5 dB.
"""
import numpy as np

# How many clips before a device's own history is trusted at all. 30 is about
# five hours at a 600 s interval, and enough that one loud transient cannot
# invent a mode.
MIN_HISTORY = 30

# How many recent medians to keep per device. 200 at 600 s is ~33 h, which
# spans several HVAC cycles without letting a gain change linger for weeks.
HISTORY_LEN = 200

# Class means must be at least this far apart to call it two states. Chosen by
# measurement, not taste. Across 9 cases -- 4 genuinely bimodal, 5 single-state:
#
#   case                  truth   sep dB   within sd   ratio   mass
#   P3 mixed              two      20.2        3.03     6.65    0.39
#   IdleHVAC + FPR5       two      12.6        1.97     6.43    0.50
#   IdleHVAC + FPR12      two      11.2        1.66     6.73    0.49
#   IdleHVAC + MERV1      two      11.5        1.48     7.79    0.50
#   IdleHVAC only         one       4.4        1.06     4.19    0.13
#   FPR5 only             one       9.3        0.91    10.29    0.04
#   FPR12 only            one       5.7        1.05     5.47    0.05
#   P3 off-clips only     one       7.0        1.69     4.18    0.25
#   P3 on-clips only      one       0.8        0.13     6.27    0.38
#
# `sep` separates the two groups cleanly: every true case >= 11.2, every false
# case <= 9.3. A separation-to-spread RATIO does not -- FPR5-only scores 10.29,
# higher than any genuinely bimodal case -- so it is not used.
#
# The margin is ~2 dB on 9 cases, so this is a reasonable boundary rather than
# a bulletproof one. That is tolerable because refusing is safe: when a split is
# refused the floor fallback still admits real blower-on clips (on P3's off-only
# history it lands at -31.9 dB, well below P3's real on-clips at -17.5).
SEP_MIN_DB = 10.0

# The smaller class must hold at least this fraction. IdleHVAC's low tail is
# ~5-8% of its clips, so 0.15 excludes it while admitting any real duty cycle
# (a blower that runs 15% of the time is still a blower).
MIN_MASS = 0.15

# Fallback when no confident split exists: "louder than this device's own quiet
# by a margin".
FLOOR_PERCENTILE = 10
FLOOR_MARGIN_DB = 12.0

# The floor must NOT be a plain percentile of the current window. A blower that
# runs for hours fills the window with loud clips, the percentile rises with
# them, and the floor-plus-margin boundary ends up ABOVE the blower-on state --
# so the louder the device gets, the less willing the gate is to call it on.
#
# Measured on P3's real 200-clip history: a genuine -17.5 dB blower-on clip was
# rejected at 131 of 171 points (77%) by the window percentile alone.
#
# So the floor is tracked instead, the way a noise floor always is: it drops
# immediately to any quieter evidence, and is allowed to rise only slowly, so a
# long loud run cannot drag it up. RISE_MAX_DB is per update (one clip).
FLOOR_RISE_MAX_DB = 0.25

# Medians at or below this are not an acoustic state, they are a broken signal
# path. compute_spectrum() floors at -100 dB, and a clip of digital zeros --
# the exact symptom of a mic clocked on the wrong word-select pin, as P5 and
# P12 both produced -- lands there. Letting those into the history would be
# actively harmful: Otsu would see -100 against a real -20 as an 80 dB
# separation, declare it bimodal, and put the boundary near -60, after which
# every genuinely silent clip gets called "blower on".
DEAD_DB = -90.0


def _otsu(sorted_vals):
    """Threshold index k maximising between-class variance, splitting into
    sorted_vals[:k+1] and sorted_vals[k+1:]. Returns (k, variance)."""
    n = sorted_vals.size
    csum = np.cumsum(sorted_vals)
    total = csum[-1]
    best_k, best_v = -1, -1.0
    for k in range(n - 1):
        n0 = k + 1
        n1 = n - n0
        mu0 = csum[k] / n0
        mu1 = (total - csum[k]) / n1
        v = (n0 / n) * (n1 / n) * (mu0 - mu1) ** 2
        if v > best_v:
            best_v, best_k = v, k
    return best_k, best_v


def learn_boundary(history, stored=None, stored_floor=None):
    """(boundary_db, note, is_fresh).

    is_fresh is True only when a confident bimodal split was found in the history
    just now. The caller should persist the boundary in that case and pass it
    back as `stored` on later calls.

    WHY A BOUNDARY IS REMEMBERED RATHER THAN RECOMPUTED FROM SCRATCH EVERY TIME
    The mass test assumes a device sees both states reasonably often inside the
    window. A blower that runs for hours breaks that: the quiet class shrinks
    below MIN_MASS, the split is refused, and the floor rule then measures the
    LOUD state -- pushing the boundary up until real blower-on clips are rejected.

    Not hypothetical. On 2026-10-01 P3 learned -26.9 dB, ran for 137 minutes, and
    by evening its window held 147 loud clips against 53 quiet (13% mass). The
    split was refused, the boundary drifted to -9.8 dB, and a run of genuine
    -17 dB blower-on clips between 00:01 and 01:41 the next morning was rejected.
    It recovered once quiet clips returned, but those detections were lost.

    So a boundary learned from good evidence is HELD while evidence is poor, and
    replaced only when a new confident split appears. Same principle as freezing
    the drift threshold while a device reads dirty: never re-derive a parameter
    from data that has just failed its own test.
    """
    h = np.asarray([v for v in history if v is not None and np.isfinite(v)], dtype=float)

    if h.size < MIN_HISTORY:
        if stored is not None:
            return (float(stored),
                    f"only {h.size} clips of history, holding the boundary learned "
                    f"earlier ({float(stored):+.1f} dB)", False)
        return None, f"only {h.size} clips of history, need {MIN_HISTORY}", False

    s = np.sort(h)
    k, _ = _otsu(s) if s[-1] - s[0] > 0 else (-1, 0.0)

    if k >= 0:
        quiet, loud = s[:k + 1], s[k + 1:]
        sep = float(loud.mean() - quiet.mean())
        mass = min(quiet.size, loud.size) / s.size
        if sep >= SEP_MIN_DB and mass >= MIN_MASS:
            boundary = float((quiet[-1] + loud[0]) / 2)
            return (boundary,
                    f"bimodal: means {quiet.mean():+.1f} / {loud.mean():+.1f} dB "
                    f"({sep:.1f} apart), {quiet.size} quiet / {loud.size} loud, "
                    f"boundary {boundary:+.1f} dB", True)
        reasons = []
        if sep < SEP_MIN_DB:
            reasons.append(f"means only {sep:.1f} dB apart")
        if mass < MIN_MASS:
            reasons.append(f"smaller class just {100 * mass:.0f}% of clips")
        why = "; ".join(reasons)
    else:
        why = "history has no spread at all"

    # No confident split right now. Prefer a boundary this device has already
    # earned over one derived from evidence that just failed its own test.
    if stored is not None:
        return (float(stored),
                f"no split ({why}) -- holding the boundary learned earlier "
                f"({float(stored):+.1f} dB)", False)

    floor = track_floor(h, stored_floor)
    boundary = floor + FLOOR_MARGIN_DB
    return (boundary,
            f"one state only ({why}): tracked floor {floor:+.1f} "
            f"+ {FLOOR_MARGIN_DB:.0f} dB -> {boundary:+.1f} dB", False)


def track_floor(history, stored_floor=None):
    """This device's quiet floor. Drops instantly to quieter evidence, rises at
    most FLOOR_RISE_MAX_DB per clip, so a long blower run cannot carry it up."""
    h = np.asarray([v for v in history if v is not None and np.isfinite(v)], dtype=float)
    if h.size == 0:
        return float(stored_floor) if stored_floor is not None else 0.0
    observed = float(np.percentile(h, FLOOR_PERCENTILE))
    if stored_floor is None:
        return observed
    stored_floor = float(stored_floor)
    if observed <= stored_floor:
        return observed                                   # quieter: believe it at once
    return min(observed, stored_floor + FLOOR_RISE_MAX_DB)


def push(history, median_db, limit=HISTORY_LEN):
    """Append one clip's median and trim to the window. Called for EVERY clip,
    whatever the gate decided -- that is what breaks the deadlock. Returns
    (history, kept) so the caller can log why a clip was ignored."""
    h = list(history or [])
    if median_db is None or not np.isfinite(median_db):
        return h[-limit:], False
    if median_db <= DEAD_DB:
        return h[-limit:], False          # dead signal path, not a quiet room
    h.append(float(median_db))
    return h[-limit:], True
