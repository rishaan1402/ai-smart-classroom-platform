"""Speech activity — Silero VAD as bundled with faster-whisper (no extra model download)."""

from dataclasses import dataclass

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

from .io import SAMPLE_RATE


@dataclass
class SpeechSegment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def speech_segments(
    audio: np.ndarray,
    threshold: float = 0.5,
    min_silence_ms: int = 400,
    min_speech_ms: int = 250,
) -> list[SpeechSegment]:
    """Speech regions in seconds. min_silence_ms sets how long a pause splits two turns."""
    opts = VadOptions(
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=min_speech_ms,
        speech_pad_ms=100,
    )
    stamps = get_speech_timestamps(audio, opts)
    return [SpeechSegment(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in stamps]
