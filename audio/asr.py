"""
Speech recognition — faster-whisper with word timestamps.

Decoding settings come from opensource-clipping-autopilot's transcribe_video:
condition_on_previous_text=False + vad_filter=True, because on long
recordings one hallucinated stretch otherwise poisons the context and the
model skips real speech for the rest of the file. That matters for
45–75 minute lectures.
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    language_probability: float
    duration_s: float
    model: str
    decode_seconds: float
    segments: list[Segment]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        segments = [
            Segment(**{**s, "words": [Word(**w) for w in s.get("words", [])]}) for s in d["segments"]
        ]
        return cls(**{**d, "segments": segments})


def default_device() -> tuple[str, str]:
    """faster-whisper (CTranslate2) runs on CUDA or CPU; there is no MPS backend."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


def load_model(model_size: str = "large-v3") -> WhisperModel:
    device, compute_type = default_device()
    logger.info(f"Loading Whisper {model_size} on {device} ({compute_type})")
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe(
    model: WhisperModel,
    audio: np.ndarray,
    model_name: str,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> Transcript:
    """
    Transcribe 16 kHz mono audio.

    Args:
        language: ISO code ("en", "hi") or None to auto-detect.
        initial_prompt: Optional vocabulary hint (subject, course name) that
            helps with technical terms in lectures.
    """
    started = time.time()
    segments_iter, info = model.transcribe(
        audio,
        beam_size=5,
        language=language,
        word_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=True,
        initial_prompt=initial_prompt,
    )
    segments = []
    for s in segments_iter:
        segments.append(Segment(
            start=s.start,
            end=s.end,
            text=s.text.strip(),
            avg_logprob=s.avg_logprob,
            no_speech_prob=s.no_speech_prob,
            words=[Word(w.word.strip(), w.start, w.end, w.probability) for w in (s.words or [])],
        ))
        if len(segments) % 100 == 0:
            logger.info(f"  asr {s.end:7.1f}s / {info.duration:.0f}s")
    return Transcript(
        language=info.language,
        language_probability=info.language_probability,
        duration_s=info.duration,
        model=model_name,
        decode_seconds=time.time() - started,
        segments=segments,
    )
