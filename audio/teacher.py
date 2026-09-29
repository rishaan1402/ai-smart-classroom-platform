"""
Teaching metrics from the teacher's speech.

Inputs: VAD speech segments (timing) + the ASR transcript (words, questions).
All metrics describe the lesson, not a verdict on the teacher; they feed the
teacher's own coaching report first.

Questions are detected from Whisper's punctuation ("?"), which is reliable
in English, plus unambiguous cue words at the start or end of a sentence in
English and Hindi (Devanagari and romanized), since Whisper rarely
punctuates Hindi. English auxiliaries (is, can, did…) and relative openers
(which, when, where) are not cues: they start ordinary clauses too.
"""

import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .asr import Segment, Transcript
from .vad import SpeechSegment

# Explicit word ends, not \b: Python's regex treats Devanagari vowel signs
# (matras, anusvara) as non-word characters, so \b fires *inside* words:
# "क्यों" would match the start of "क्योंकि" (because).
_END = r"(?=[\s,.?!।]|$)"
_START = r"(?:^|(?<=\s))"
QUESTION_START = re.compile(
    r"^(what|why|how|who|"
    r"kya|kyu|kyun|kyon|kaise|kaun|kitna|kitne|kab|kahan|"
    r"क्या|क्यों|क्यूँ|कैसे|कौन|कितना|कितने|कब|कहाँ|कहां)" + _END,
    re.IGNORECASE,
)
QUESTION_END = re.compile(
    r"(\?|" + _START + r"(right|okay|ok|isn't it|no|na|haina|hai na|samjhe|samajh aaya|"
    r"ना|है ना|समझे|समझ आया)\s*[.?!।]?$)",
    re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.?!।])\s+")

LONG_PAUSE_S = 3.0
PAUSE_SPLIT_S = 0.8
MAX_SENTENCE_WORDS = 40


@dataclass
class Question:
    time_s: float
    text: str
    wait_time_s: float    # pause after the question before the next word (student think/answer time)


@dataclass
class TeacherMetrics:
    duration_s: float
    speech_s: float
    talk_ratio: float                 # teacher speech / lesson time
    words: int
    words_per_minute: float           # over speaking time only
    questions: int
    questions_per_10min: float
    median_wait_time_s: float | None  # after questions
    long_pauses: int                  # silences ≥ LONG_PAUSE_S
    longest_monologue_s: float        # longest stretch with no pause ≥ LONG_PAUSE_S
    mean_asr_confidence: float        # mean word probability; low = noisy or unclear audio

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(segments: list[Segment]) -> list[tuple[float, float, str]]:
    """
    Sentence-level (start, end, text) rebuilt from the word stream across segments.

    Whisper segments often start mid-sentence, and a fragment like "which
    defines…" would look like a question opener. Sentences end at terminal
    punctuation or, for unpunctuated speech (Whisper rarely punctuates Hindi),
    at a pause ≥ PAUSE_SPLIT_S or after MAX_SENTENCE_WORDS words.
    """
    words = [w for s in segments for w in s.words]
    if not words:
        return [(s.start, s.end, s.text) for s in segments if s.text.strip()]
    sentences, buf = [], []
    for i, w in enumerate(words):
        buf.append(w)
        next_gap = words[i + 1].start - w.end if i + 1 < len(words) else None
        if (
            re.search(r"[.?!।]$", w.word)
            or next_gap is None
            or next_gap >= PAUSE_SPLIT_S
            or len(buf) >= MAX_SENTENCE_WORDS
        ):
            sentences.append((buf[0].start, buf[-1].end, " ".join(x.word for x in buf)))
            buf = []
    return sentences


def is_question(text: str) -> bool:
    t = text.strip()
    return bool(t) and (bool(QUESTION_END.search(t)) or bool(QUESTION_START.search(t)))


def find_questions(transcript: Transcript) -> list[Question]:
    """
    Questions with their wait time: the gap from the question's end to the
    next transcribed word. Word timestamps, not VAD segments: VAD merges a
    continuous lecture into one long segment, hiding short pauses.
    """
    word_starts = np.array(sorted(w.start for s in transcript.segments for w in s.words))
    questions = []
    for start, end, text in split_sentences(transcript.segments):
        if not is_question(text):
            continue
        later = word_starts[word_starts >= end - 0.05]
        wait = float(later[0] - end) if len(later) else 0.0
        questions.append(Question(time_s=round(start, 2), text=text, wait_time_s=round(max(0.0, wait), 2)))
    return questions


def compute_metrics(transcript: Transcript, speech: list[SpeechSegment], duration_s: float) -> TeacherMetrics:
    speech_s = sum(s.duration for s in speech)
    words = [w for s in transcript.segments for w in s.words]
    questions = find_questions(transcript)

    gaps = [b.start - a.end for a, b in zip(speech[:-1], speech[1:])]
    longest, run_start = 0.0, speech[0].start if speech else 0.0
    for a, b in zip(speech[:-1], speech[1:]):
        if b.start - a.end >= LONG_PAUSE_S:
            longest = max(longest, a.end - run_start)
            run_start = b.start
    if speech:
        longest = max(longest, speech[-1].end - run_start)

    waits = [q.wait_time_s for q in questions]
    return TeacherMetrics(
        duration_s=round(duration_s, 1),
        speech_s=round(speech_s, 1),
        talk_ratio=round(speech_s / duration_s, 3) if duration_s else 0.0,
        words=len(words),
        words_per_minute=round(len(words) / (speech_s / 60), 1) if speech_s else 0.0,
        questions=len(questions),
        questions_per_10min=round(len(questions) / (duration_s / 600), 2) if duration_s else 0.0,
        median_wait_time_s=round(float(np.median(waits)), 2) if waits else None,
        long_pauses=sum(g >= LONG_PAUSE_S for g in gaps),
        longest_monologue_s=round(longest, 1),
        mean_asr_confidence=round(float(np.mean([w.probability for w in words])), 3) if words else 0.0,
    )


def minute_timeline(transcript: Transcript, speech: list[SpeechSegment], duration_s: float) -> pd.DataFrame:
    """Per-minute speech seconds, words, pace and questions, joinable with the engagement curve."""
    minutes = int(np.ceil(duration_s / 60)) or 1
    talk = np.zeros(minutes)
    for s in speech:
        for m in range(int(s.start // 60), min(minutes, int(s.end // 60) + 1)):
            talk[m] += max(0.0, min(s.end, (m + 1) * 60) - max(s.start, m * 60))
    words = np.zeros(minutes, dtype=int)
    for seg in transcript.segments:
        for w in seg.words:
            words[min(minutes - 1, int(w.start // 60))] += 1
    qs = np.zeros(minutes, dtype=int)
    for q in find_questions(transcript):
        qs[min(minutes - 1, int(q.time_s // 60))] += 1
    return pd.DataFrame({
        "minute": np.arange(minutes),
        "speech_s": talk.round(1),
        "words": words,
        "wpm": np.where(talk > 5, words / np.maximum(talk, 1e-6) * 60, 0).round(1),
        "questions": qs,
    })
