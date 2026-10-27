"""blower_signature.py -- is the blower running, judged by SHAPE not level.

Why this exists
---------------
The spectral gate in service/poll_and_infer.py asks "is this clip loud?" --
the median of 72 log-spaced bands. That works at Abraham House (P3), whose
device is mounted against the air handler and hears motor rumble: blower-on
lifts everything below 500 Hz by ~25 dB, which moves the median a long way.

It fails at Kannan House (P5), mounted near a register, where the device hears
AIRFLOW instead. Measured against three clips labelled by hand on 2026-10-04:

    octave band      20-60   60-125  125-250  250-500  500-1k   1-2k   2-8k
    P5 on vs off      -0.7     +4.0     +4.3     +8.4   +12.8   +7.9    +0.7
    P3 on vs off     +24.4    +24.4    +25.7    +29.2   +27.9  +18.3    +9.6

P5's lift sits where the broadband median barely notices it, so household
noise and a running blower land on the same side of any level threshold. The
two features below are differences taken WITHIN one clip, so overall gain,
mic distance and how loud the house happens to be all cancel.

Validation (516 clips, 2026-09-22..10-06)
-----------------------------------------
Both features order all five hand-labelled clips correctly, with a clear gap:
on-clips tilt -24.5..-21.0 / tone +7.3..+15.2, references tilt -36.4..-33.3 /
tone +0.1..+0.7. The joint rule yields runs averaging 4.71 clips where
randomly scattering the same number of positives gives 1.20 -- i.e. it tracks
something that persists in time, which is the signature of a machine and not
of a threshold crossing noise.

Caveat, stated plainly: five labels from one afternoon. The thresholds below
are set from that handful plus the run-coherence test, NOT from a labelled
corpus, and they are specific to a register-mounted device. Do not ship them
fleet-wide without labels from each mounting style.
"""
import numpy as np

FRAME = 2048          # must match spectrum_viz.FRAME_SIZE
TILT_ON_DB = -27.0    # mid-band shelf relative to the clip's own 20-60 Hz
TONE_ON_DB = 3.0      # prominence of the ~445 Hz blade-pass tone

TILT_HI = (400.0, 1200.0)
TILT_LO = (20.0, 60.0)
TONE_BAND = (430.0, 465.0)
TONE_BACKDROP = (300.0, 700.0)


def fine_spectrum(y, sr, frame=FRAME):
    """Mean magnitude spectrum in dB, framed exactly as compute_spectrum does
    (Hann, 50% hop, no centring) so the two can be compared bin for bin.

    Clips are 1.0 s, which is ~14 frames at 7.8 Hz resolution -- enough for a
    steady tone, thin for anything transient. Returns (freqs, db) or None when
    the clip is shorter than a single frame.
    """
    import librosa
    if y.size < frame:
        return None
    S = np.abs(librosa.stft(y, n_fft=frame, hop_length=frame // 2,
                            window="hann", center=False))
    if S.shape[1] == 0:
        return None
    half = frame // 2
    db = 20 * np.log10(S[:half].mean(axis=1) + 1e-9)
    return np.fft.rfftfreq(frame, 1 / sr)[:half], db


def features(y, sr):
    """(tilt_db, tone_db), or (None, None) for an unusable clip.

    tilt: mean(400-1200 Hz) - mean(20-60 Hz). The reference clips fall away
          monotonically from 20 Hz; a running blower carries a mid-band shelf.
    tone: max(430-465 Hz) - median(300-700 Hz). Prominence of the narrowband
          blade-pass peak over its own backdrop -- measured at ~445 Hz and 8 Hz
          wide, consistent with a 24-blade squirrel cage near 1110 rpm.
    """
    got = fine_spectrum(y, sr)
    if got is None:
        return None, None
    f, db = got

    def band(lo, hi):
        return db[(f >= lo) & (f < hi)]

    tilt = float(band(*TILT_HI).mean() - band(*TILT_LO).mean())
    tone = float(band(*TONE_BAND).max() - np.median(band(*TONE_BACKDROP)))
    return tilt, tone


def blower_is_on(y, sr, tilt_thr=TILT_ON_DB, tone_thr=TONE_ON_DB):
    """(is_on, note). Both conditions must hold: a mid-band shelf AND the tone.

    Requiring both is what separates a running blower from a loud room -- a
    television lifts the mid band too, but carries no 445 Hz line.
    """
    tilt, tone = features(y, sr)
    if tilt is None:
        return False, "clip too short to analyse"
    on = tilt > tilt_thr and tone > tone_thr
    return on, (f"tilt {tilt:+.1f} dB (>{tilt_thr:+.0f}), "
                f"tone {tone:+.1f} dB (>{tone_thr:+.0f}) -> "
                f"{'ON' if on else 'off'}")
