"""
spectrum_viz.py
Server-side port of web/src/lib/audioRecordings.js's computeSpectrum() --
NOT the same thing as extract_features_v2.py's fft_band_00..63 (a linear,
64-bin, RMS-normalized-energy-fraction feature for the classifier). This
is purely for the UI's frequency-spectrum chart: log-spaced dB magnitude
bins matching exactly what the browser computes client-side for the
*current* reading, so a server-computed *baseline* (averaged across a
device's warmup readings, in service/poll_and_infer.py) can be overlaid
on the same chart without a binning mismatch.

Keep the two implementations' defaults in sync by hand -- there's no
shared source of truth across the JS/Python boundary. Current defaults
(frame_size=2048, bins=72, f_min=20) match audioRecordings.js exactly.
"""
import numpy as np
import librosa

FRAME_SIZE = 2048
HOP = FRAME_SIZE // 2
BINS = 72
F_MIN = 20.0


def compute_spectrum(y, sr, frame_size=FRAME_SIZE, bins=BINS, f_min=F_MIN):
    """Returns bins values in dB (list of float), log-spaced from f_min to
    sr/2 (Nyquist) -- same order/spacing as the JS computeSpectrum(), so
    index i here corresponds to the same frequency band as index i there.

    Mirrors the JS implementation's framing exactly: a Hann-windowed STFT
    with 50% hop, no centering/padding (only fully-contained frames, same
    as JS's `for (start = 0; start + n <= data.length; start += hop)`),
    magnitude-averaged across frames, then log-binned and converted to dB
    with the same -100 dB floor.
    """
    nyquist = sr / 2
    S = np.abs(librosa.stft(y, n_fft=frame_size, hop_length=frame_size // 2,
                            window="hann", center=False))
    S = S[: frame_size // 2]  # drop the Nyquist bin, matching JS's `k < n/2`
    frames = S.shape[1]
    if frames == 0:
        return []
    accum = S.sum(axis=1)

    log_min = np.log10(max(f_min, 1))
    log_max = np.log10(nyquist)
    n_half = frame_size // 2

    out = []
    for b in range(bins):
        f0 = 10 ** (log_min + (log_max - log_min) * b / bins)
        f1 = 10 ** (log_min + (log_max - log_min) * (b + 1) / bins)
        k0 = max(0, int(np.floor((f0 / nyquist) * n_half)))
        k1 = min(n_half, int(np.ceil((f1 / nyquist) * n_half)))
        if k1 > k0:
            mag = accum[k0:k1].mean() / frames
        else:
            mag = 0.0
        db = max(20 * np.log10(mag + 1e-9), -100.0)
        out.append(float(db))
    return out
