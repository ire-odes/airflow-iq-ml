"""
drift_detector.py
Per-device baseline drift detection + classifier reconciliation for
AirFlow IQ production units.

Design (see walkthrough):

  BaselineProfile   -- rolling mean/covariance over a warmup window;
                       Mahalanobis distance of new readings vs baseline.
  DriftDetector     -- EWMA smoothing of the distance stream + a
                       consecutive-exceedance rule, so single noisy
                       readings can't trigger and slow clogging can't
                       hide (their asymptotic-loading concern, Sec. 2).
  calibrate_threshold -- sets the alert threshold from labeled data with
                       a deliberate false-positive bias (paper Table 7:
                       a missed dirty filter is the expensive error).
  FilterHealthMonitor -- the production state machine:
                       COLD_START/WARMING -> filter is assumed clean (a
                                     warmup only starts right after an
                                     install or an explicit recalibration
                                     -- see Devices.jsx's "Recalibrate"
                                     action). Every reading unconditionally
                                     feeds the baseline; the classifier is
                                     evaluated for logging only, never used
                                     to gate warmup, since it has no
                                     baseline to compare against yet and is
                                     the same lab-trained model already
                                     confirmed to misfire on new
                                     environments (AbrahamHouse).
                       WARM       -> drift is primary, classifier is a
                                     periodic sanity check; strong
                                     disagreement raises a flag instead
                                     of being silently ignored.

Only numpy required at runtime -- the math here (running mean/cov
update, one matrix solve per reading) is deliberately light enough to
port to the ESP32-C6 in fixed-point or float32 later.
"""

import re
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

# Drift features: use scale-invariant spectral descriptors, NOT absolute
# energy -- a per-device baseline keyed on loudness would drift with mic
# aging and ambient noise, not filter state.
DEFAULT_DRIFT_FEATURES = [
    "low_freq_energy_ratio",
    "centroid_mean",
    "rolloff_mean",
    "flatness_mean",
    "bandwidth_mean",
]

MAX_CONTINUITY_GAP_MIN = 30  # BaselineProfile assumes one continuous warmup
                             # recording. A folder with a bigger gap than this
                             # between consecutive clips is really multiple
                             # disjoint sessions stitched together, which
                             # inflates the fitted covariance with cross-
                             # session variability instead of genuine device
                             # noise (confirmed: pooling in an 8-day, 7-gap
                             # lab corpus this way dragged a working site's
                             # detection rate from 58% down to 1.2%).


def check_recording_continuity(folder, max_gap_min=MAX_CONTINUITY_GAP_MIN):
    """Returns (is_continuous, max_gap_minutes) for a folder of
    recording_YYYYMMDD_HHMMSS.wav clips. Use this before feeding a folder
    into BaselineProfile -- it only models one continuous device warmup,
    not a multi-session corpus."""
    ts = []
    for f in sorted(Path(folder).glob("*.wav")):
        m = re.match(r"recording_(\d{8})_(\d{6})", f.name)
        if m:
            ts.append(datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))
    ts.sort()
    if len(ts) < 2:
        return True, 0.0
    gaps = [(ts[i + 1] - ts[i]).total_seconds() / 60 for i in range(len(ts) - 1)]
    max_gap = max(gaps)
    return max_gap <= max_gap_min, max_gap


class BaselineProfile:
    """Welford-style running mean + covariance of the drift feature
    vector, accumulated during the warmup window (the 'listen for a
    couple hours' phase). Frozen once `is_ready`."""

    def __init__(self, n_features, min_samples=120, min_span_seconds=None):
        # min_samples: needs to be comfortably > n_features for a stable
        # covariance estimate.
        #
        # min_span_seconds: minimum wall-clock time the warmup samples must
        # cover before the baseline may freeze. Sample count alone is not
        # enough. A device sampling every 60s reaches 120 samples in ~2h,
        # which covers a single narrow slice of operating conditions -- one
        # ambient temperature, one duty cycle, one household noise floor.
        # The covariance then encodes "this afternoon" rather than "this
        # device", and readings taken under any other condition score as
        # large deviations. Observed directly: three devices warmed over a
        # 2.3h afternoon window on unchanged filters and all three drifted
        # up overnight, two crossing the dirty threshold (distances rising
        # from ~2.5 to 4-6). Requiring the window to span a full day/night
        # cycle folds that variation into the covariance instead, which is
        # what the original 120-samples-at-10-minute-wakes (~20h) design
        # got for free. None disables the check.
        self.n = 0
        self.min_samples = min_samples
        self.min_span_seconds = min_span_seconds
        self.mean = np.zeros(n_features)
        self._M2 = np.zeros((n_features, n_features))
        self._frozen = False
        self._cov_inv = None
        # Epoch seconds of the earliest/latest sample fed in, when callers
        # supply timestamps. Left None by callers that don't (build_baseline,
        # test scripts), which disables the span gate for them.
        self.first_ts = None
        self.last_ts = None

    @property
    def span_seconds(self):
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        return max(0.0, self.last_ts - self.first_ts)

    @property
    def is_ready(self):
        if self.n < self.min_samples:
            return False
        if self.min_span_seconds and self.span_seconds < self.min_span_seconds:
            return False
        return True

    def update(self, x, ts=None):
        if self._frozen:
            raise RuntimeError("Baseline frozen; not accepting updates.")
        x = np.asarray(x, dtype=float)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._M2 += np.outer(delta, x - self.mean)
        if ts is not None:
            # min/max rather than assuming arrival order: readings can be
            # processed slightly out of order across poller restarts.
            self.first_ts = ts if self.first_ts is None else min(self.first_ts, ts)
            self.last_ts = ts if self.last_ts is None else max(self.last_ts, ts)

    def freeze(self, shrinkage=0.10):
        """Finalize covariance with diagonal shrinkage regularization --
        keeps the inverse stable when warmup samples are correlated
        (they will be: it's hours of the same blower)."""
        cov = self._M2 / max(self.n - 1, 1)
        diag = np.diag(np.diag(cov))
        cov = (1 - shrinkage) * cov + shrinkage * diag
        cov += 1e-9 * np.eye(cov.shape[0])
        self._cov_inv = np.linalg.inv(cov)
        self._frozen = True

    def distance(self, x):
        """Mahalanobis distance of a reading from the baseline."""
        if self._cov_inv is None:
            raise RuntimeError("Call freeze() before distance().")
        d = np.asarray(x, dtype=float) - self.mean
        return float(np.sqrt(d @ self._cov_inv @ d))

    def get_state(self):
        """Serializable snapshot -- lets a fresh process pick up warmup
        accumulation (or a frozen baseline) where a previous run left off,
        e.g. across separate poller invocations backed by a DB row instead
        of an in-memory object."""
        return {
            "n": self.n,
            "mean": self.mean.tolist(),
            "m2": self._M2.tolist(),
            "frozen": self._frozen,
            "cov_inv": self._cov_inv.tolist() if self._cov_inv is not None else None,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
        }

    @classmethod
    def from_state(cls, n_features, min_samples, state, min_span_seconds=None):
        bp = cls(n_features, min_samples=min_samples, min_span_seconds=min_span_seconds)
        if state:
            bp.n = state["n"]
            bp.mean = np.array(state["mean"], dtype=float)
            bp._M2 = np.array(state["m2"], dtype=float)
            bp._frozen = state["frozen"]
            cov_inv = state.get("cov_inv")
            bp._cov_inv = np.array(cov_inv, dtype=float) if cov_inv is not None else None
            # .get() so rows written before the span gate existed still load.
            bp.first_ts = state.get("first_ts")
            bp.last_ts = state.get("last_ts")
        return bp


class DriftDetector:
    """EWMA-smoothed Mahalanobis distance with a consecutive-exceedance
    alert rule. EWMA because clogging is gradual: a slow upward creep in
    distance accumulates in the smoothed statistic, while one-off noise
    (door slam, HVAC startup transient) washes out."""

    def __init__(self, baseline, threshold, alpha=0.05, patience=5,
                 recent_limit=None):
        self.baseline = baseline
        self.threshold = threshold
        self.alpha = alpha              # small alpha = long memory
        self.patience = patience        # consecutive exceedances to alert
        # Second, independent alert rule: how many of the last N readings
        # exceeded the threshold. The consecutive rule is tuned for a STEP
        # change -- swap a clean filter for a clogged one and the EWMA climbs
        # and stays up, so three in a row arrives quickly. Real clogging is
        # gradual: the distance creeps up and crosses in scattered bursts that
        # fall back before three consecutive, so a filter can degrade for
        # weeks while the consecutive rule never fires.
        #
        # The window is defined by the CALLER and counted in READINGS, not
        # hours. An HVAC is not always running, and clips recorded while it is
        # off are dropped as silence before ever reaching here -- so a
        # wall-clock window holds a wildly variable number of samples, and the
        # same rule behaves differently on a day the unit ran 2 hours versus
        # 12. A fixed count of readings keeps statistical power constant
        # however the duty cycle moves. None disables it.
        self.recent_limit = recent_limit
        self.ewma = 0.0
        self._exceed_count = 0
        self._initialized = False

    def update(self, x, recent_exceedances=None, recent_limit=None):
        """recent_exceedances: how many of this device's PRECEDING readings
        had an EWMA above its threshold, counted by the caller from stored
        history. Passed in rather than tracked here so the window survives
        process restarts without another persisted buffer -- the readings are
        already in the database.

        This reading is not in that history yet (the caller stores it only
        after this returns), so its own exceedance is added below. Without
        that the window would lag a reading behind, and the count reported in
        the result would not match the readings it claims to cover.

        recent_limit: how many of the window constitute dirty, overriding
        self.recent_limit. Supplied per reading because the window is short
        just after warmup, and a fraction of a 5-reading window is not the
        same rule as a fraction of a 30-reading one."""
        d = self.baseline.distance(x)
        if not self._initialized:
            self.ewma = d
            self._initialized = True
        else:
            self.ewma = self.alpha * d + (1 - self.alpha) * self.ewma

        exceeded = self.ewma > self.threshold
        if exceeded:
            self._exceed_count += 1
        else:
            self._exceed_count = 0

        limit = recent_limit if recent_limit is not None else self.recent_limit
        window_count = (recent_exceedances + (1 if exceeded else 0)
                        if recent_exceedances is not None else None)
        by_streak = self._exceed_count >= self.patience
        by_recent = (limit is not None
                     and window_count is not None
                     and window_count >= limit)

        return {
            "distance": d,
            "ewma": self.ewma,
            "drift_dirty": by_streak or by_recent,
            # Which rule fired, so a slow-creep detection is distinguishable
            # from a step change in the history rather than both landing as
            # an undifferentiated "dirty".
            "drift_rule": "streak" if by_streak else ("recent" if by_recent else None),
            # The inclusive count, so what gets logged and stored is the
            # number of exceedances in the window that produced this verdict.
            "recent_exceedances": window_count,
            "recent_limit": limit,
        }

    def get_state(self):
        return {
            "ewma": self.ewma,
            "exceed_count": self._exceed_count,
            "initialized": self._initialized,
        }

    def set_state(self, state):
        self.ewma = state["ewma"]
        self._exceed_count = state["exceed_count"]
        self._initialized = state["initialized"]


def calibrate_threshold(clean_distances, dirty_distances,
                        clean_percentile=95.0):
    """Set the alert threshold from labeled validation data.

    Paper Table 7 logic: bias toward false positives. We place the
    threshold at the `clean_percentile` of clean-filter distances
    (accepting ~5% FP pressure) and then REPORT what fraction of dirty
    readings clear it, rather than optimizing a symmetric metric.

    Returns (threshold, expected_fp_rate, dirty_detection_rate).
    """
    thr = float(np.percentile(clean_distances, clean_percentile))
    fp = float(np.mean(np.asarray(clean_distances) > thr))
    det = float(np.mean(np.asarray(dirty_distances) > thr))
    return thr, fp, det


def _f_quantile(q, dfn, dfd):
    """Quantile of the F distribution, with a Monte Carlo fallback so this
    module stays importable without scipy (F = (chi2_dfn/dfn)/(chi2_dfd/dfd))."""
    try:
        from scipy.stats import f as _f
        return float(_f.ppf(q, dfn, dfd))
    except Exception:
        rng = np.random.default_rng(0)
        a = rng.chisquare(dfn, 200_000) / dfn
        b = rng.chisquare(dfd, 200_000) / dfd
        return float(np.percentile(a / b, q * 100.0))


def _beta_quantile(q, a, b):
    """Quantile of the Beta distribution, same fallback rationale."""
    try:
        from scipy.stats import beta as _beta
        return float(_beta.ppf(q, a, b))
    except Exception:
        rng = np.random.default_rng(1)
        return float(np.percentile(rng.beta(a, b, 200_000), q * 100.0))


def threshold_from_baseline(baseline, quantile=0.99, margin=1.10,
                            warmup_samples=None):
    """Derive this device's alert threshold from its own frozen baseline.

    A Mahalanobis threshold needs NO observed clean readings. The instant the
    covariance is frozen, the distribution of the distance is known, so the
    percentile cut is available immediately -- there is nothing to wait for.
    Before this existed every device ran on one shared default (3.515), which
    cannot serve a fleet whose baseline spreads differ by 3x: on 2026-09-03 a
    real filter swap sat at the 94th percentile of P3's own baseline while
    P6's *clean* readings sat at the 100th percentile of its own.

    Two routes, preferring measurement over theory:

    1. EMPIRICAL, when the warmup feature vectors are available: the actual
       distance of every calibration sample from the finished baseline. Makes
       no distributional assumption, and a fat tail here is exactly the
       pathology worth catching.
    2. CLOSED FORM otherwise. For a fresh observation scored against a mean
       and covariance estimated from n samples, Hotelling's T-squared gives
           D^2 * n(n-d) / (d(n+1)(n-1))  ~  F(d, n-d)
       so the quantile follows from n and d alone. Slightly wider than the
       chi-square limit it converges to, because estimating the covariance
       from finite data is itself uncertain.

    Returns (threshold, detail_dict) -- the dict records which route was used
    and both candidate values, so a threshold can be audited later rather
    than appearing as an unexplained number in a column.
    """
    n = int(baseline.n)
    d = int(len(baseline.mean))
    detail = {"n_samples": n, "n_features": d, "quantile": quantile,
              "margin": margin, "method": None,
              "empirical": None, "closed_form": None}

    # Closed form is always computed: it is the cross-check that says whether
    # an empirical number is plausible or whether the baseline is malformed.
    if n > d + 1:
        f_q = _f_quantile(quantile, d, n - d)
        d2 = d * (n + 1) * (n - 1) / (n * (n - d)) * f_q
        detail["closed_form"] = float(np.sqrt(d2))

    if warmup_samples is not None and len(warmup_samples) >= max(20, d * 4):
        dists = np.array([baseline.distance(np.asarray(s, dtype=float))
                          for s in warmup_samples], dtype=float)
        emp = float(np.percentile(dists, quantile * 100.0))

        # These samples FITTED the mean and covariance, so their distances are
        # systematically smaller than a fresh reading's -- scoring a point
        # against a centre it helped place. In-sample the scaled distance is
        # Beta((d)/2, (n-d-1)/2); correcting by the ratio of the two theoretical
        # quantiles removes the bias. It is a small effect (~0.5% at n=400,
        # ~2% at n=120) but it is a bias, not noise, and it points the wrong
        # way -- toward a threshold that is too tight.
        if n > d + 2 and detail["closed_form"]:
            beta_q = _beta_quantile(quantile, d / 2.0, (n - d - 1) / 2.0)
            in_sample = float(np.sqrt((n - 1) ** 2 / n * beta_q))
            if in_sample > 0:
                emp *= detail["closed_form"] / in_sample
                detail["in_sample_correction"] = detail["closed_form"] / in_sample

        detail["empirical"] = emp
        detail["method"] = "empirical"
        return emp * margin, detail

    if detail["closed_form"] is None:
        # Too few samples to say anything. The caller keeps whatever default
        # it had rather than being handed a fabricated number.
        detail["method"] = "insufficient"
        return None, detail

    detail["method"] = "closed_form"
    return detail["closed_form"] * margin, detail


class MonitorState(Enum):
    COLD_START = "cold_start"
    WARMING = "warming"
    WARM = "warm"


class FilterHealthMonitor:
    """Production reconciliation layer.

    classifier_fn(feature_dict) -> (label_str, prob_dirty)
    drift_features: ordered list of keys pulled from feature_dict for
                    the baseline/drift path.
    """

    def __init__(self, classifier_fn, drift_features=None,
                 min_baseline_samples=120, drift_threshold=None,
                 classifier_dirty_conf=0.85, ewma_alpha=0.05,
                 patience=5, min_baseline_span_seconds=None,
                 recent_exceedance_limit=None,
                 auto_threshold_quantile=None, auto_threshold_margin=1.10):
        self.classifier_fn = classifier_fn
        self.drift_features = drift_features or DEFAULT_DRIFT_FEATURES
        self.baseline = BaselineProfile(len(self.drift_features),
                                        min_samples=min_baseline_samples,
                                        min_span_seconds=min_baseline_span_seconds)
        self._drift_threshold = drift_threshold
        self._detector = None
        self._ewma_alpha = ewma_alpha
        self._patience = patience
        self.classifier_dirty_conf = classifier_dirty_conf
        self._recent_exceedance_limit = recent_exceedance_limit
        # Consecutive WARM-state readings where the classifier was
        # confident-dirty against a calm baseline -- the override below
        # only fires once this reaches `patience`, same reasoning as
        # DriftDetector's own exceedance counter: a single confident-but-
        # wrong classifier reading (which we've directly observed happens
        # on new environments -- see AbrahamHouse) shouldn't flip a
        # decision a calm, calibrated baseline disagrees with.
        self._override_streak = 0
        # When set, the threshold is derived from the device's own baseline at
        # freeze time (see threshold_from_baseline) instead of being inherited
        # from whatever default the caller passed in. None keeps the old
        # behaviour, which the offline scripts rely on.
        self._auto_threshold_quantile = auto_threshold_quantile
        self._auto_threshold_margin = auto_threshold_margin
        # Populated at freeze: which route produced the threshold and what the
        # alternative would have been, so the number stays explainable.
        self.threshold_detail = None
        self.state = MonitorState.COLD_START

    def _vec(self, feats):
        return np.array([feats[k] for k in self.drift_features])

    def set_drift_threshold(self, value):
        """Replace the alert threshold on a live monitor.

        Both copies have to move: the monitor holds it for persistence, and
        DriftDetector holds its own reference which is what update() actually
        compares the EWMA against. Setting only the first is a silent no-op --
        the stored number changes, the behaviour does not.
        """
        self._drift_threshold = float(value)
        if self._detector is not None:
            self._detector.threshold = float(value)

    def get_state(self):
        """Serializable snapshot of everything process() needs to resume
        correctly in a fresh process -- see BaselineProfile.get_state /
        DriftDetector.get_state for the pieces this wraps."""
        state = {
            "state": self.state.value,
            "drift_threshold": self._drift_threshold,
            "baseline": self.baseline.get_state(),
            "override_streak": self._override_streak,
        }
        if self._detector is not None:
            state["detector"] = self._detector.get_state()
        return state

    @classmethod
    def from_state(cls, classifier_fn, state, drift_features=None,
                   min_baseline_samples=120, classifier_dirty_conf=0.85,
                   ewma_alpha=0.05, patience=5, min_baseline_span_seconds=None,
                   recent_exceedance_limit=None,
                   auto_threshold_quantile=None, auto_threshold_margin=1.10):
        """Rehydrates a monitor from get_state()'s output (None/{} for a
        brand-new device -- behaves exactly like a fresh __init__)."""
        mon = cls(classifier_fn, drift_features=drift_features,
                  min_baseline_samples=min_baseline_samples,
                  drift_threshold=(state or {}).get("drift_threshold"),
                  classifier_dirty_conf=classifier_dirty_conf,
                  ewma_alpha=ewma_alpha, patience=patience,
                  min_baseline_span_seconds=min_baseline_span_seconds,
                  recent_exceedance_limit=recent_exceedance_limit,
                  auto_threshold_quantile=auto_threshold_quantile,
                  auto_threshold_margin=auto_threshold_margin)
        if state:
            mon.state = MonitorState(state["state"])
            mon.baseline = BaselineProfile.from_state(
                len(mon.drift_features), min_baseline_samples, state.get("baseline"),
                min_span_seconds=min_baseline_span_seconds)
            if state.get("detector") and mon.baseline._frozen:
                mon._detector = DriftDetector(
                    mon.baseline, mon._drift_threshold,
                    alpha=ewma_alpha, patience=patience,
                    recent_limit=recent_exceedance_limit)
                mon._detector.set_state(state["detector"])
            mon._override_streak = state.get("override_streak", 0)
        return mon

    def process(self, feats, ts=None, recent_exceedances=None, recent_limit=None,
                warmup_samples=None):
        """Feed one reading's feature dict. Returns a decision record.

        ts: epoch seconds for this reading. Optional, but without it the
        baseline can't measure how much wall-clock time its warmup covers,
        so min_baseline_span_seconds has no effect.

        warmup_samples: this device's calibration feature vectors, used only
        on the reading that freezes the baseline, to derive the threshold
        empirically instead of from theory. Passed in rather than accumulated
        here because the poller is stateless between cycles -- the vectors
        live in filter_ml_readings.features, so they survive restarts and stay
        auditable after the fact. None falls back to the closed form."""
        label, p_dirty = self.classifier_fn(feats)
        clf_dirty = p_dirty >= 0.5
        record = {"state": self.state.value, "classifier_label": label,
                  "classifier_p_dirty": p_dirty, "flag": None}

        # ---- COLD_START / WARMING: filter assumed clean, baseline builds --
        if self.state in (MonitorState.COLD_START, MonitorState.WARMING):
            self.state = MonitorState.WARMING

            # A warmup only ever starts right after a filter install or an
            # explicit recalibration (see Devices.jsx's "Recalibrate"
            # action, which resets device_baselines specifically because
            # the user just fitted a known-clean filter) -- so every
            # warmup reading unconditionally feeds the baseline. No
            # attempt is made to guess "dirty at install" from the
            # classifier: it has no baseline to compare against yet, and
            # it's the same lab-trained model already confirmed to
            # misfire on new environments (AbrahamHouse: 100% of a real
            # house's genuinely clean readings called "dirty"). Trusting
            # that raw guess over the install itself would poison the
            # thing we're trying to protect -- an accurate "clean" baseline.
            self.baseline.update(self._vec(feats), ts=ts)

            if self.baseline.is_ready:
                self.baseline.freeze()

                # Derive this device's own threshold from the baseline that
                # just froze. Previously a threshold was only ever supplied by
                # the caller -- in practice one shared default for the whole
                # fleet, which then never changed, because the tool for
                # revising it read post-freeze clean history that a freshly
                # calibrated device does not have yet. The number is available
                # right here, from the calibration itself, with nothing to
                # wait for.
                if self._auto_threshold_quantile is not None:
                    thr, detail = threshold_from_baseline(
                        self.baseline,
                        quantile=self._auto_threshold_quantile,
                        margin=self._auto_threshold_margin,
                        warmup_samples=warmup_samples)
                    if thr is not None:
                        self._drift_threshold = thr
                        self.threshold_detail = detail
                elif self._drift_threshold is None:
                    # No calibrated threshold and auto-calibration disabled:
                    # fall back to the same closed form rather than a guess.
                    thr, detail = threshold_from_baseline(self.baseline)
                    self._drift_threshold = thr if thr is not None else 3.5
                    self.threshold_detail = detail

                self._detector = DriftDetector(
                    self.baseline, self._drift_threshold,
                    alpha=self._ewma_alpha, patience=self._patience,
                    recent_limit=self._recent_exceedance_limit)
                self.state = MonitorState.WARM

            # The classifier is trained on lab data and routinely
            # misfires on a new install's own acoustic environment (a
            # real device called 100% of a real house's clean readings
            # "dirty" -- see AbrahamHouse) -- exposing that raw guess as
            # a "clean"/"dirty" verdict during warmup is actively
            # misleading, not just noisy. "calibrating" is the honest
            # decision here; classifier_label/classifier_p_dirty are
            # still returned above for logging/analysis, just not
            # surfaced as a trustworthy result.
            record["decision"] = "calibrating"
            record["decided_by"] = "calibrating"
            return record

        # ---- WARM: drift is primary, classifier is sanity check -------
        drift = self._detector.update(self._vec(feats),
                                      recent_exceedances=recent_exceedances,
                                      recent_limit=recent_limit)
        record.update(drift)
        drift_dirty = drift["drift_dirty"]

        record["decision"] = "dirty" if drift_dirty else "clean"
        # Distinguish the two drift rules. A "drift_recent" verdict means the
        # EWMA never held above threshold for `patience` readings in a row but
        # crossed it in enough of the recent window to matter -- the gradual
        # clogging signature, which reads very differently from a step change
        # when reviewing history.
        record["decided_by"] = "drift_recent" if drift.get("drift_rule") == "recent" else "drift"

        # Disagreement handling: strong classifier "dirty" against a
        # calm baseline is informative, not noise -- either the baseline
        # is bad or the classifier found something absolute. Escalate --
        # but only once it's *sustained*, same reasoning as drift's own
        # patience counter. A single confident-but-wrong classifier
        # reading (the demonstrated failure mode on new environments)
        # must not flip a decision a calibrated baseline disagrees with;
        # `patience` consecutive confident-dirty-while-calm readings is
        # a materially different, much less likely, signal.
        if (p_dirty >= self.classifier_dirty_conf) and not drift_dirty:
            self._override_streak += 1
            if self._override_streak >= self._patience:
                record["flag"] = ("disagreement: classifier confident dirty "
                                  f"for {self._override_streak} straight "
                                  "readings, drift calm -> review / re-baseline?")
                record["decision"] = "dirty"        # FP-biased tiebreak
                record["decided_by"] = "classifier_override"
            else:
                record["flag"] = (f"watching: classifier confident dirty "
                                  f"({self._override_streak}/{self._patience}), "
                                  "drift still calm")
        else:
            self._override_streak = 0
            if drift_dirty and p_dirty < 0.2:
                record["flag"] = ("disagreement: drift alarmed, classifier "
                                  "confident clean -> possible environment "
                                  "change, not filter")

        return record
