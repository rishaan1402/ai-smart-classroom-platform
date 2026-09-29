"""
Room-mic acoustic features: a per-second noise curve, not a transcript.

CCTV room mics (with AGC) can't be transcribed (see docs/footage_findings.md),
but their level still tells us when the class gets loud. Per second:
  rms_db          loudness (dBFS)
  speech_band     share of energy in 300–3400 Hz (voices vs. fans, scraping)
  above_baseline  dB above the rolling 60 s median, which normalizes away mic gain
  spike           a sudden loud event (≥ SPIKE_DB above baseline)
  noisy           sustained voice-band noise above baseline (chatter)
"""

import numpy as np
import pandas as pd

from .io import SAMPLE_RATE

FRAME = 400          # 25 ms
SPIKE_DB = 12.0
CHATTER_DB = 6.0
BASELINE_WINDOW_S = 60


def noise_curve(audio: np.ndarray) -> pd.DataFrame:
    usable = audio[: audio.size // FRAME * FRAME].reshape(-1, FRAME)
    rms_db = 20 * np.log10(np.sqrt((usable ** 2).mean(axis=1)) + 1e-9)

    spec = np.abs(np.fft.rfft(usable * np.hanning(FRAME), axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1 / SAMPLE_RATE)
    band = (freqs >= 300) & (freqs <= 3400)
    speech_band = spec[:, band].sum(axis=1) / (spec.sum(axis=1) + 1e-12)

    frames_per_s = SAMPLE_RATE // FRAME
    seconds = len(rms_db) // frames_per_s
    per_s = pd.DataFrame({
        "second": np.arange(seconds),
        # Energy-average dB within each second, and peak for spikes
        "rms_db": 10 * np.log10(
            (10 ** (rms_db[: seconds * frames_per_s] / 10)).reshape(seconds, -1).mean(axis=1) + 1e-12
        ),
        "peak_db": rms_db[: seconds * frames_per_s].reshape(seconds, -1).max(axis=1),
        "speech_band": speech_band[: seconds * frames_per_s].reshape(seconds, -1).mean(axis=1),
    })
    baseline = per_s["rms_db"].rolling(BASELINE_WINDOW_S, center=True, min_periods=5).median()
    per_s["above_baseline"] = per_s["rms_db"] - baseline
    per_s["spike"] = (per_s["peak_db"] - baseline) >= SPIKE_DB
    per_s["noisy"] = (per_s["above_baseline"] >= CHATTER_DB) & (per_s["speech_band"] >= 0.5)
    return per_s.round(3)
