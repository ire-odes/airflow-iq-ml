"""
firmware_mfcc.py
Python port of the LoRa node's on-device MFCC (computeMFCC() in
firmware/filtsure_lora_slot_sync/filtsure_lora_slot_sync.ino), so a wav
recorded by a WiFi node can be turned into the same 13 numbers a LoRa node
uplinks.

NOT librosa's MFCC, and not interchangeable with extract_features_v2's
mfcc*_mean/std. The firmware uses 26 HTK-formula mel bands with unnormalised
triangles, a 400-sample Hann frame zero-padded to a 512 FFT, hop 160,
log10 (not dB, no top_db clip), an unnormalised DCT-II, and mean-pools over
frames. Every one of those differs from librosa's defaults.

Ported line for line, float32 throughout like the chip, including the +1e-6
guards and the x100 -> int16 truncation. The only intended difference is the
FFT: numpy's instead of the sketch's radix-2, which agrees to float32
rounding -- far below the 0.01 step the x100 int16 quantisation imposes.

One property worth knowing: c1..c12 do not change with microphone gain.
A gain g adds 2*log10(g) to every band's log energy, and for c >= 1 the DCT
weights sum to zero across the 26 bands, so the shift cancels. Only c0 moves.
So c1..c12 compare across devices and gain settings; c0 does not.
"""
import numpy as np

SR = 16000
TOTAL_SAMPLES = SR          # CLIP_DURATION_S = 1
FRAME_SIZE = 400
HOP_SIZE = 160
FFT_SIZE = 512
NUM_MEL_FILTERS = 26
NUM_MFCC = 13
NUM_FRAMES = (TOTAL_SAMPLES - FRAME_SIZE) // HOP_SIZE + 1


def _hz_to_mel(hz):
    return np.float32(2595.0) * np.log10(np.float32(1.0) + np.float32(hz) / np.float32(700.0))


def _mel_to_hz(mel):
    return np.float32(700.0) * (np.power(np.float32(10.0), np.float32(mel) / np.float32(2595.0))
                                - np.float32(1.0))


def _mel_filterbank():
    low, high = _hz_to_mel(0.0), _hz_to_mel(SR / 2.0)
    bins = []
    for i in range(NUM_MEL_FILTERS + 2):
        mel = low + (high - low) * np.float32(i) / np.float32(NUM_MEL_FILTERS + 1)
        bins.append(int(np.floor(np.float32(FFT_SIZE + 1) * _mel_to_hz(mel) / np.float32(SR))))

    fb = np.zeros((NUM_MEL_FILTERS, FFT_SIZE // 2 + 1), dtype=np.float32)
    for m in range(1, NUM_MEL_FILTERS + 1):
        f_minus, f_ctr, f_plus = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_minus, f_ctr):
            if 0 <= k <= FFT_SIZE // 2:
                fb[m - 1, k] = np.float32(k - f_minus) / np.float32(f_ctr - f_minus + 1e-6)
        for k in range(f_ctr, f_plus):
            if 0 <= k <= FFT_SIZE // 2:
                fb[m - 1, k] = np.float32(f_plus - k) / np.float32(f_plus - f_ctr + 1e-6)
    return fb


MEL_FB = _mel_filterbank()
HANN = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(FRAME_SIZE) / (FRAME_SIZE - 1))).astype(np.float32)
DCT = np.cos((np.pi / NUM_MEL_FILTERS)
             * (np.arange(NUM_MEL_FILTERS)[None, :] + 0.5)
             * np.arange(NUM_MFCC)[:, None]).astype(np.float32)


def firmware_mfcc(samples_int16):
    """13 MFCCs exactly as the node uplinks them, already divided by 100 the
    way ingest_ttn stores them. `samples_int16` is the clip as int16 counts
    (the firmware keeps the top 16 bits of each 32-bit I2S word)."""
    x = np.asarray(samples_int16)[:TOTAL_SAMPLES].astype(np.float32) / np.float32(32768.0)
    if x.size < TOTAL_SAMPLES:
        # The node always captures a full second unless the 10 s capture
        # timeout fires; then the unfilled tail is whatever was in the buffer.
        # Zeros are the honest stand-in here.
        x = np.pad(x, (0, TOTAL_SAMPLES - x.size))

    acc = np.zeros(NUM_MFCC, dtype=np.float32)
    for f in range(NUM_FRAMES):
        start = f * HOP_SIZE
        frame = np.zeros(FFT_SIZE, dtype=np.float32)
        frame[:FRAME_SIZE] = x[start:start + FRAME_SIZE] * HANN
        spec = np.fft.rfft(frame)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        mel_energy = np.log10(MEL_FB @ power + np.float32(1e-6)).astype(np.float32)
        acc += DCT @ mel_energy

    scaled = np.clip(acc / np.float32(NUM_FRAMES) * np.float32(100.0), -32768.0, 32767.0)
    return np.trunc(scaled).astype(np.int16).astype(float) / 100.0   # C cast truncates


def from_wav(path):
    """Firmware MFCCs for a wav on disk. WiFi uploads are 16-bit PCM at 16 kHz,
    which is the node's int16 sample format already."""
    import soundfile as sf
    y, sr = sf.read(path, dtype="int16", always_2d=False)
    if y.ndim > 1:
        y = y[:, 0]
    if sr != SR:
        import librosa
        y = (librosa.resample(y.astype(np.float32) / 32768.0, orig_sr=sr, target_sr=SR)
             * 32768.0).astype(np.int16)
    return firmware_mfcc(y)
