"""
analyze_clip.py
Run the whole pipeline against ONE stored .wav and show what it concludes:
the frequency spectrum chart, whether the blower was running, and whether the
filter reads clean or dirty.

  python analyze_clip.py clip.wav
  python analyze_clip.py clip.wav --device E4B3238E6B10
  python analyze_clip.py clip.wav --device E4B3238E6B10 --out spectrum.png
  python analyze_clip.py clip.wav --json          # machine-readable, no chart

WHY --device MATTERS, and what you get without it
-------------------------------------------------
Neither question has a fleet-wide answer, so without --device this tool
reports measurements and withholds verdicts:

  blower on/off   The gate is per device. A register-mounted unit hears
                  airflow (P5: +12.8 dB at 500 Hz-1 kHz, nothing below
                  60 Hz); one strapped to the air handler hears motor rumble
                  (P3: +24 dB below 60 Hz). The fleet-wide 2.5 dB cut is what
                  discarded 82 of P3's wakes as silent while its blower ran.
  clean/dirty     Drift is measured against THIS device's own baseline. The
                  classifier alone has a raw opinion, but it is demonstrably
                  unreliable on an environment it was not trained on, which
                  is why the app shows "calibrating" rather than that opinion
                  until a device is warm. This tool labels it the same way.

With --device it loads that device's real baseline from Supabase and runs the
identical code path service/poll_and_infer.py uses -- load_monitor() and
blower_is_on() are imported from it, not reimplemented, so this cannot drift
from production.

READ-ONLY. The monitor object is mutated in memory by process() (that is how
the EWMA advances), but nothing is ever written back: no baseline update, no
filter_ml_readings row, no last_processed_at. Running this never changes what
the fleet thinks. The flip side is that the verdict shown is "how this clip
WOULD be scored against the device's current state", not a verdict that was
or will be recorded.
"""
import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "service"))

import librosa                                              # noqa: E402
from extract_features_v2 import extract_features, SR, SILENCE_RMS_THRESHOLD  # noqa: E402
from spectrum_viz import compute_spectrum, BINS, F_MIN      # noqa: E402
import blower_signature                                     # noqa: E402

# AcousticPanel.jsx's ISOLATED_BAND_MAX_HZ. Mirrored by hand, as spectrum_viz
# mirrors audioRecordings.js -- there is no shared constant across the
# Python/JS boundary.
CROP_HZ = 2000

# Both naming schemes the downloader has used: "recording_20261004_203111.wav"
# and the label-first "BLOWER ON - 20261004_203111.wav". The timestamp is what
# makes a stored clip analysable as history rather than as an anonymous file.
TS_IN_NAME = re.compile(r"(\d{8})_(\d{6})")

# Where a bare filename is looked for, so "analyze_clip.py 'unknown 504 - ...'"
# works without typing a path. Searched in order; first match wins.
SEARCH_ROOTS = [
    ROOT / "data",
    Path.home() / "Desktop" / "ML" / "data",
]

# Which device recorded the clips in a given folder. Without this a clip found
# by name would be analysed with no device, and silently get measurements
# instead of verdicts -- the one failure mode this tool exists to avoid.
# Folder name (lowercased) -> MAC. Anything not listed is treated as having no
# device, which is correct for the lab training sets (FPR12, FPR5, MERV1_CLEAN,
# IdleHVAC) -- those are rig recordings, not a deployed unit's history.
FOLDER_DEVICE = {
    "kannan_house_p5": "E4B3238E6B10",
    "abraham_house_p3": "58E6C5204B28",
    "abrahamhouse": "58E6C5204B28",
}


def device_for(path):
    """MAC for the device whose clips live here, or None.

    Walks UP the directory chain rather than looking only at the immediate
    parent. Downloads get filed into dated subfolders
    (data/AbrahamHouse/October15_data/...), and matching only the leaf meant
    every such folder silently fell back to "no device" -- which downgrades
    verdicts to bare measurements without ever saying why. Nearest match wins,
    so a per-device subfolder still overrides its parent.
    """
    here = path if path.is_dir() else path.parent
    for folder in [here, *here.parents]:
        mac = FOLDER_DEVICE.get(folder.name.lower())
        if mac:
            return mac, folder.name
    return None, None

# Charts go in a subfolder of wherever the clip lives, not beside it. A clip
# folder is the device's recording history and holds nothing but .wav files;
# dropping a PNG per analysed clip in there buries the recordings among
# generated output, and a folder run with --graphs would double its file count.
# Nothing globs *.wav out of here, so the subfolder is inert to the rest of the
# tool and safe to delete at any time.
ANALYSIS_DIRNAME = "analysis"


def analysis_dir(clip_path):
    """<clip's folder>/analysis/, created on demand."""
    d = clip_path.parent / ANALYSIS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def clip_timestamp(path):
    """(epoch_seconds, source). When the clip was RECORDED, as best as can be
    known from a file on disk.

    This matters more than it looks. FilterHealthMonitor measures the baseline
    span from these timestamps, and the drift threshold only freezes once the
    warmup covers MIN_BASELINE_SPAN_SECONDS. Passing now() for a clip recorded
    three weeks ago tells the monitor the whole history happened in one
    instant, which is exactly the mistake 20260829000000_baseline_warmup_span
    .sql exists to prevent.
    """
    m = TS_IN_NAME.search(path.name)
    if m:
        try:
            return (datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
                    .timestamp()), "filename"
        except ValueError:
            pass
    try:
        return path.stat().st_mtime, "file mtime"
    except OSError:
        return datetime.now(timezone.utc).timestamp(), "now (no timestamp found)"


# -- device context ----------------------------------------------------------

def load_device(mac):
    """(monitor, baseline_spectrum, spectrum_n, gate_hist, gate_boundary,
    classifier_fn) for this device, straight from production's own loader.

    Imported lazily so the tool still runs the measurement-only path on a
    machine with no Supabase credentials."""
    import poll_and_infer as P
    from supabase import create_client

    if not (P.SUPABASE_URL and P.SUPABASE_SERVICE_ROLE_KEY):
        raise SystemExit("--device needs SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
                         "in ML/service/.env")
    sb = create_client(P.SUPABASE_URL, P.SUPABASE_SERVICE_ROLE_KEY)
    clf = P.load_classifier()
    mon, spec_mean, spec_n, gate_hist, gate_boundary, had_row = P.load_monitor(sb, mac, clf)
    if not had_row:
        print(f"[!] no device_baselines row for {mac} -- it has never been scored, "
              f"so this is a cold-start monitor, not its real state.")
    return P, mon, spec_mean, spec_n, gate_hist, gate_boundary, clf


# -- the two questions -------------------------------------------------------

def blower_verdict(y, spec, mac, ctx):
    """(on_or_None, note, detail). None means 'measured, not judged' -- no
    device context, so no rule applies."""
    tilt, tone = blower_signature.features(y, SR)
    median_db = float(np.median(np.asarray(spec, dtype=float)))
    detail = {"gate_median_db": round(median_db, 2),
              "shape_tilt_db": None if tilt is None else round(tilt, 2),
              "shape_tone_db": None if tone is None else round(tone, 2)}

    if ctx is None:
        shape = ("not measurable (clip shorter than one 2048-sample frame)"
                 if tilt is None else
                 f"tilt {tilt:+.1f} dB, tone {tone:+.1f} dB "
                 f"-> shape rule alone would say "
                 f"{'ON' if (tilt > blower_signature.TILT_ON_DB and tone > blower_signature.TONE_ON_DB) else 'off'}")
        return None, (f"no device given, so no gate applies. "
                      f"Gate median {median_db:+.1f} dB. {shape}"), detail

    P, mon, spec_mean, spec_n, gate_hist, gate_boundary, _ = ctx
    on, note, _fresh = P.blower_is_on(spec, spec_mean, spec_n, gate_hist,
                                      gate_boundary, y=y, sr=SR, device_mac=mac)
    return bool(on), note, detail


def filter_verdict(path, y, mac, ctx, blower_on, ts=None):
    """(verdict, note, detail). Mirrors the pipeline's own order of refusals:
    a silent clip and a blower-off clip both produce NO verdict, because a
    filter cannot be judged from air that is not moving."""
    feats = extract_features(str(path))
    if feats is None:
        rms = float(np.sqrt(np.mean(y ** 2)))
        return None, (f"clip is near-silent (RMS {rms:.5f} < "
                      f"{SILENCE_RMS_THRESHOLD}); the pipeline discards it "
                      f"without a verdict"), {"rms": round(rms, 6)}

    detail = {}
    if ctx is not None:
        _P, _mon, _sm, _sn, _gh, _gb, clf = ctx
        raw_label, p_dirty = clf(feats)
    else:
        import joblib
        bundle_path = ROOT / "models" / "model_bundle.pkl"
        if not bundle_path.exists():
            return None, f"no classifier at {bundle_path}", detail
        bundle = joblib.load(bundle_path)
        cols = bundle["selected_features"]
        missing = [k for k in cols if k not in feats]
        if missing:
            return None, (f"extractor is out of sync with the model: "
                          f"{len(missing)} missing feature(s)"), detail
        x = np.array([[feats[k] for k in cols]])
        di = int(np.where(bundle["label_encoder"].classes_ == "FPR12")[0][0])
        p_dirty = float(bundle["classifier"].predict_proba(x)[0][di])
        raw_label = "clogged" if p_dirty >= 0.5 else "clean"
    detail["classifier_p_dirty"] = round(p_dirty, 4)
    detail["classifier_label"] = raw_label

    if blower_on is False:
        return None, (f"blower was off, so no filter verdict -- the pipeline "
                      f"discards this clip. (Classifier's raw opinion, which it "
                      f"would never act on here: {raw_label}, p_dirty={p_dirty:.3f})"), detail

    if ctx is None:
        return None, (f"no device given, so no drift baseline to judge against. "
                      f"Classifier's raw opinion only: {raw_label}, "
                      f"p_dirty={p_dirty:.3f} -- NOT a verdict, the app would "
                      f"show 'calibrating' rather than this"), detail

    P, mon, _sm, _sn, _gh, _gb, _clf = ctx
    # warmup_samples=None: this tool never freezes a baseline. If this clip
    # happened to be the one that tips the device over its freeze gates, the
    # monitor falls back to its closed-form threshold instead of the empirical
    # one production would compute. Flagged rather than hidden.
    res = mon.process(feats, ts=ts if ts is not None else datetime.now(timezone.utc).timestamp(),
                      recent_exceedances=None, recent_limit=None,
                      warmup_samples=None)
    detail.update({"state": res["state"], "decided_by": res.get("decided_by"),
                   "distance": _round(res.get("distance")),
                   "ewma": _round(res.get("ewma")),
                   "threshold": _round(mon.get_state().get("drift_threshold"))})
    if res["decision"] == "calibrating":
        n = mon.baseline.n
        return "calibrating", (f"device is {res['state']} ({n}/{P.MIN_BASELINE_SAMPLES} "
                               f"warmup samples) -- no verdict until it is warm"), detail
    return res["decision"], (f"{res['decision']} (decided_by={res.get('decided_by')}, "
                             f"distance {_fmt(res.get('distance'))}, "
                             f"EWMA {_fmt(res.get('ewma'))} vs threshold "
                             f"{_fmt(mon.get_state().get('drift_threshold'))})"), detail


def _round(v, n=4):
    return None if v is None else round(float(v), n)


def _fmt(v):
    return "n/a" if v is None else f"{float(v):.3f}"


# -- chart -------------------------------------------------------------------

ACCENT, POS, NEG = "#007BFF", "#e8590c", "#0f8a6a"
TEXT, SUBTEXT, DIVIDER, INK, MUTED = "#1a1a2e", "#9ca3af", "#eceef3", "#1a1a2e", "#6b7280"


def _bars(ax, vals, base, colour_of):
    """Bars anchored at `base`, carrying the app's top-to-bottom opacity fade."""
    from matplotlib.patches import Rectangle
    for xi, v in enumerate(vals):
        h, n = v - base, 20
        for k in range(n):
            a = 0.95 + (0.45 - 0.95) * (k / (n - 1))
            y0 = base + h * k / n if h >= 0 else base + h * (k + 1) / n
            ax.add_patch(Rectangle((xi - 0.44, y0), 0.88, abs(h) / n + 0.01,
                                   facecolor=colour_of(v), alpha=a, lw=0, zorder=3))


def _frame(ax, f_b, lo, hi, ystep, title, ylabel):
    ax.set_xlim(-0.9, len(f_b) - 0.1); ax.set_ylim(lo, hi)
    ax.set_yticks(np.arange(lo, hi + 1, ystep))
    ax.grid(True, axis="y", color=DIVIDER, ls=(0, (4, 4)), lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(DIVIDER)
    step = max(1, len(f_b) // 8)
    ax.set_xticks(np.arange(len(f_b))[::step])
    ax.set_xticklabels([f"{f/1000:.1f}k" if f >= 1000 else str(f) for f in f_b[::step]])
    ax.tick_params(colors=SUBTEXT, labelsize=9, length=0)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontweight(600)
    ax.set_xlabel("Frequency (Hz)", color=SUBTEXT, fontsize=10, fontweight=600, labelpad=7)
    ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=10, fontweight=600)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=8)
    ax.set_facecolor("white")
    if lo < 0 < hi:
        ax.axhline(0, color=INK, lw=2.2, zorder=8, solid_capstyle="butt")
        ax.text(len(f_b) - 1.2, 0, "0 dB", color=INK, fontsize=9.2, fontweight=700,
                va="bottom", ha="right", zorder=9,
                bbox=dict(fc="white", ec="none", pad=1.2))


def draw(spec, baseline_spec, out_path, title, subtitle):
    """Left: the web app's Acoustic view -- 72 log-spaced bands cropped to
    CROP_HZ, accent bars, the device baseline overlaid as a dashed line.

    Right, only when the device HAS a baseline: the gain, this clip minus that
    baseline, bars anchored at zero. The app draws both curves and leaves the
    reader to subtract them by eye; at 55 bars that is the part worth drawing,
    because the drift verdict is a statement about this difference and nothing
    else. Positive = louder than this device's normal, negative = quieter.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.logspace(np.log10(F_MIN), np.log10(SR / 2), BINS + 1)
    freq = np.round((edges[:-1] + edges[1:]) / 2).astype(int)
    band = freq <= CROP_HZ
    f_b, vals = freq[band], np.asarray(spec, dtype=float)[band]
    ref = None if baseline_spec is None else np.asarray(baseline_spec, dtype=float)[band]

    two = ref is not None
    fig, axes = plt.subplots(1, 2 if two else 1,
                             figsize=(15.2 if two else 11.6, 4.9))
    fig.patch.set_facecolor("white")
    ax = axes[0] if two else axes

    src = list(vals) + (list(ref) if two else [])
    lo = np.floor(min(src) / 5) * 5 - 5
    hi = np.ceil(max(src) / 5) * 5 + 5
    _frame(ax, f_b, lo, hi, 10, "Spectrum", "Level (dB)")
    _bars(ax, vals, lo, lambda v: ACCENT)
    if two:
        ax.plot(np.arange(len(ref)), ref, color=MUTED, lw=1.6, ls=(0, (4, 3)),
                zorder=6, label="device baseline")
        ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=MUTED)
    if lo < 0 < hi:
        ax.axhline(0, color=INK, lw=2.2, zorder=8, solid_capstyle="butt")

    if two:
        gain = vals - ref
        glo = np.floor(min(gain.min(), -2) / 5) * 5
        ghi = np.ceil(max(gain.max(), 2) / 5) * 5
        _frame(axes[1], f_b, glo, ghi, 5,
               "Gain vs this device's baseline", "Gain (dB)")
        _bars(axes[1], gain, 0.0, lambda v: POS if v >= 0 else NEG)
        axes[1].axhline(0, color=INK, lw=2.2, zorder=8, solid_capstyle="butt")

    fig.suptitle(title, color=TEXT, fontsize=13, x=0.032, ha="left", y=0.975)
    fig.text(0.032, 0.9, subtitle, color=SUBTEXT, fontsize=9.3)
    fig.tight_layout(rect=(0, 0, 1, 0.845))
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


# -- one clip ----------------------------------------------------------------

def analyse(path, ctx, mac, want_graph, out=None):
    """Everything known about one clip. Shared by the single-file and folder
    paths so they cannot disagree."""
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    if y.size == 0:
        return {"file": str(path), "error": "contains no samples"}
    spec = compute_spectrum(y, SR)
    if not spec:
        return {"file": str(path),
                "error": f"shorter than one 2048-sample frame ({y.size} samples)"}

    ts, ts_src = clip_timestamp(path)
    when = datetime.fromtimestamp(ts)
    on, blower_note, blower_detail = blower_verdict(y, spec, mac, ctx)
    verdict, filter_note, filter_detail = filter_verdict(path, y, mac, ctx, on, ts=ts)

    png = None
    if want_graph:
        # Named by when the clip was RECORDED, so the analysis folder sorts
        # chronologically and a chart is identifiable without opening it.
        # ':' is illegal in a Windows filename, hence '-' between H, M and S.
        png = Path(out) if out else analysis_dir(path) / (
            when.strftime("%Y-%m-%d %H-%M-%S") + ".png")
        label = "blower ON" if on else ("blower off" if on is False else "blower not judged")
        draw(spec, ctx[2] if ctx else None, png,
             f"Recorded {when:%Y-%m-%d %H:%M:%S}",
             f"{path.name} · {label} · filter: {verdict or 'no verdict'} · "
             f"20 Hz–{CROP_HZ//1000} kHz, as the web app draws it")

    return {"file": str(path), "name": path.name, "device": mac,
            "recorded_at": datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
            "timestamp_from": ts_src,
            "duration_s": round(y.size / SR, 3),
            "blower_on": on, "blower_note": blower_note, "blower": blower_detail,
            "filter_verdict": verdict, "filter_note": filter_note,
            "filter": filter_detail,
            "spectrum_db": [round(v, 2) for v in spec],
            "graph": None if png is None else str(png)}


def resolve_input(path):
    """The file or folder the user meant, or exit with something useful.

    Windows Explorer hides known extensions, so the name you copy off a clip
    is "unknown 504 - 20261006_163505" with no .wav on the end. Pasting that
    is the normal case, not user error -- so append the suffix rather than
    refusing. When that still misses, show near matches: a wrong clip index
    with a right timestamp (or the reverse) is otherwise a silent dead end in
    a folder of 500 near-identical names.
    """
    if path.exists():
        return path

    if not path.suffix:
        with_ext = path.with_name(path.name + ".wav")
        if with_ext.exists():
            return with_ext

    # A bare name, with no directory part: go and find it.
    if path.parent in (Path("."), Path("")):
        wanted = path.name if path.suffix else path.name + ".wav"
        hits = []
        for root in SEARCH_ROOTS:
            if root.exists():
                hits += [p for p in root.rglob(wanted)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            listing = "\n  ".join(str(h) for h in hits[:8])
            raise SystemExit(f"{wanted} exists in more than one place; "
                             f"give the path you mean:\n  {listing}")
        near = []
        for root in SEARCH_ROOTS:
            if not root.exists():
                continue
            m = TS_IN_NAME.search(path.name)
            if m:
                near += [str(p) for p in root.rglob(f"*{m.group(0)}*.wav")]
        if near:
            raise SystemExit(f"no clip named {wanted}.\n\nDid you mean:\n  "
                             + "\n  ".join(near[:6]))
        roots = "\n  ".join(str(r) for r in SEARCH_ROOTS if r.exists())
        raise SystemExit(f"no clip named {wanted} under:\n  {roots}")

    parent = path.parent if path.parent.exists() else None
    hint = ""
    if parent:
        stem = path.stem.lower()
        keys = [k for k in (TS_IN_NAME.search(path.name), None) if k]
        ts = keys[0].group(0) if keys else None
        near = [p.name for p in parent.glob("*.wav")
                if (ts and ts in p.name) or stem[:14] in p.name.lower()][:6]
        if near:
            hint = "\n\nDid you mean:\n  " + "\n  ".join(near)
        else:
            hint = f"\n\n{len(list(parent.glob('*.wav')))} .wav file(s) in {parent}"
    raise SystemExit(f"no such file or folder: {path}{hint}")


# -- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", help="a .wav file, or a FOLDER of them")
    ap.add_argument("--device", default=None, metavar="MAC",
                    help="device MAC, e.g. E4B3238E6B10. Without it you get "
                         "measurements but no verdicts -- see the module docstring.")
    ap.add_argument("--out", default=None,
                    help="PNG path (default: <clip folder>/analysis/<wav>_spectrum.png)")
    ap.add_argument("--no-graph", action="store_true", help="skip the chart")
    ap.add_argument("--graphs", action="store_true",
                    help="folder mode: write a PNG per clip (off by default -- "
                         "a folder of 500 clips means 500 files)")
    ap.add_argument("--csv", default=None, metavar="PATH",
                    help="write one row per clip here (works for a single "
                         "file as well as a folder)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--limit", type=int, default=None,
                    help="folder mode: stop after N clips")
    args = ap.parse_args()

    path = resolve_input(Path(args.wav).expanduser())

    # Infer the device from the folder when not told, so a bare filename still
    # gets verdicts rather than silently degrading to measurements. Announced,
    # never silent: an inferred MAC that is wrong would produce a confident
    # answer against the wrong device's baseline.
    device = args.device
    if device is None:
        device, matched = device_for(path)
        if device:
            print(f"device {device} inferred from folder '{matched}' "
                  f"(override with --device)")

    # Device context is loaded ONCE, not per clip: it is a Supabase round trip
    # plus the model bundle, and a per-clip shell loop would pay both every
    # time. Loading it once also means the monitor advances across the folder
    # exactly as production advanced it across the real uploads.
    ctx = load_device(device) if device else None
    args.device = device

    if path.is_dir():
        return run_folder(path, ctx, args)

    res = analyse(path, ctx, args.device, not args.no_graph, args.out)
    if "error" in res:
        raise SystemExit(f"{path.name}: {res['error']}")
    if args.csv:
        write_csv(args.csv, [res])
    if args.json:
        print(json.dumps(res, indent=2))
        return
    print_one(res, args.device)
    if args.csv:
        print(f"         csv   -> {args.csv}")


def print_one(r, mac):
    print(f"\n{r['name']}   {r['duration_s']:.2f} s @ {SR} Hz"
          + (f"   device {mac}" if mac else "   (no device given)"))
    print("-" * 72)
    print(f"BLOWER   {'ON' if r['blower_on'] else 'off' if r['blower_on'] is False else 'not judged'}")
    print(f"         {r['blower_note']}")
    print(f"FILTER   {r['filter_verdict'] or 'no verdict'}")
    print(f"         {r['filter_note']}")
    print(f"SPECTRUM gate median {r['blower']['gate_median_db']:+.1f} dB over "
          f"{len(r['spectrum_db'])} bands")
    print(f"RECORDED {r['recorded_at']}  (from {r['timestamp_from']})")
    if r["graph"]:
        print(f"         chart -> {r['graph']}")
    print()


def write_csv(dest, rows):
    """One row per clip. Same columns whether it came from one file or 646,
    so a single-clip check and a folder replay concatenate cleanly."""
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["recorded_at", "file", "device", "duration_s", "blower_on",
                    "gate_median_db", "shape_tilt_db", "shape_tone_db",
                    "filter_verdict", "classifier_p_dirty", "state",
                    "distance", "ewma", "threshold", "blower_note"])
        for r in rows:
            b, f = r["blower"], r["filter"]
            w.writerow([r["recorded_at"], r["name"], r["device"], r["duration_s"],
                        r["blower_on"], b["gate_median_db"], b["shape_tilt_db"],
                        b["shape_tone_db"], r["filter_verdict"],
                        f.get("classifier_p_dirty"), f.get("state"),
                        f.get("distance"), f.get("ewma"), f.get("threshold"),
                        r["blower_note"]])


def run_folder(folder, ctx, args):
    """Replay a folder of stored clips IN RECORDING ORDER.

    Order is the whole point. A warm monitor's verdict depends on its EWMA,
    which carries over from the previous reading, so clips fed in the sequence
    they were recorded reproduce what the pipeline saw. Shuffled, or one
    process per clip, every reading is judged from the device's current EWMA
    instead and the drift history is lost.

    Still read-only: nothing is written back, so this replays history without
    rewriting it.
    """
    clips = sorted((p for p in folder.glob("*.wav")
                    if p.parent.name != ANALYSIS_DIRNAME),
                   key=lambda p: (clip_timestamp(p)[0], p.name))
    if not clips:
        raise SystemExit(f"no .wav files in {folder}")
    if args.limit:
        clips = clips[:args.limit]
    print(f"{len(clips)} clip(s) in {folder.name}, oldest first"
          + (f"   device {args.device}" if args.device else "   (no device given)"))
    if ctx is None:
        print("no --device: measurements only, no verdicts. See --help.")
    print()

    rows, errors = [], 0
    counts = {"on": 0, "off": 0, "unjudged": 0}
    verdicts = {}
    print(f"{'recorded':<20}{'clip':<34}{'blower':<9}{'filter':<13}{'median dB':>10}")
    for p in clips:
        r = analyse(p, ctx, args.device, args.graphs)
        if "error" in r:
            print(f"{'':<20}{p.name[:33]:<34}{'--':<9}{r['error'][:30]}")
            errors += 1
            continue
        rows.append(r)
        b = "ON" if r["blower_on"] else ("off" if r["blower_on"] is False else "-")
        counts["on" if r["blower_on"] else "off" if r["blower_on"] is False else "unjudged"] += 1
        v = r["filter_verdict"] or "-"
        verdicts[v] = verdicts.get(v, 0) + 1
        print(f"{r['recorded_at'][:19].replace('T', ' '):<20}{r['name'][:33]:<34}"
              f"{b:<9}{v:<13}{r['blower']['gate_median_db']:>10.1f}")

    print(f"\n{len(rows)} analysed, {errors} unreadable")
    if args.graphs and rows:
        print(f"  charts -> {folder / ANALYSIS_DIRNAME}")
    print(f"  blower: {counts['on']} on, {counts['off']} off"
          + (f", {counts['unjudged']} not judged" if counts["unjudged"] else ""))
    print("  filter: " + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())))

    if args.csv:
        write_csv(args.csv, rows)
        print(f"  csv -> {args.csv}")

    if args.json:
        for r in rows:
            r.pop("spectrum_db", None)      # 72 values x N clips drowns the output
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
