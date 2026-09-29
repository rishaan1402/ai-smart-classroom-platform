"""Audio I/O — ffmpeg decode to 16 kHz mono float32 (the rate Whisper and Silero expect)."""

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


def load_audio(path: Path, start_s: float = 0.0, duration_s: float | None = None) -> np.ndarray:
    """
    Decode any audio/video file to a mono float32 array in [-1, 1] at 16 kHz.

    Same ffmpeg settings as opensource-clipping-autopilot's extract_audio,
    piped to memory instead of a WAV file.
    """
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start_s), "-i", str(path)]
    if duration_s is not None:
        cmd += ["-t", str(duration_s)]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def save_wav(path: Path, audio: np.ndarray) -> None:
    """Write 16 kHz mono 16-bit PCM (for listening checks)."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "-", str(path)],
        input=pcm.tobytes(), check=True,
    )
