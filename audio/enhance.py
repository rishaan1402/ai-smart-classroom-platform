"""
Optional speech enhancement for noisy teacher audio.

Spectral gating (noisereduce). DeepFilterNet was the plan's choice but its
native build fails on Python 3.12 here; tools/audio_noise_robustness.py
measures whether enhancement helps ASR before we depend on it.
"""

import numpy as np
import noisereduce

from .io import SAMPLE_RATE


def enhance(audio: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """Non-stationary spectral gating; strength = prop_decrease (0 = off, 1 = full)."""
    out = noisereduce.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=False, prop_decrease=strength)
    return out.astype(np.float32)
