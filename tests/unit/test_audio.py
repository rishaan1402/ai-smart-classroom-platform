"""Unit tests for audio metrics and room noise — no Whisper model needed."""

import numpy as np

from audio.asr import Segment, Transcript, Word
from audio.io import SAMPLE_RATE
from audio.room import noise_curve
from audio.teacher import compute_metrics, find_questions, is_question, minute_timeline, split_sentences
from audio.vad import SpeechSegment


def seg(start, end, text, words=None) -> Segment:
    if words is None:
        toks = text.split()
        step = (end - start) / max(1, len(toks))
        words = [Word(t, start + i * step, start + (i + 1) * step, 0.9) for i, t in enumerate(toks)]
    return Segment(start, end, text, -0.2, 0.01, words)


def transcript(*segments) -> Transcript:
    return Transcript("en", 0.99, 120.0, "test", 1.0, list(segments))


class TestQuestions:
    def test_punctuation_and_cue_words(self):
        assert is_question("What is photosynthesis?")
        assert is_question("why does the leaf turn yellow")
        assert is_question("So this is the answer, right?")
        assert is_question("क्या आप समझे")
        assert is_question("yeh formula yaad hai na")
        assert not is_question("Today we will study the cell.")
        assert not is_question("")

    def test_relative_clause_fragments_are_not_questions(self):
        assert not is_question("which defines pharmacognosy as a study of drugs.")
        assert not is_question("was generally to improve the bioavailability of the drug.")

    def test_sentences_rebuilt_across_segments(self):
        a = seg(0.0, 2.0, "This course is one")
        b = seg(2.0, 4.0, "which defines drugs. Is it clear?")
        texts = [x[2] for x in split_sentences([a, b])]
        assert texts == ["This course is one which defines drugs.", "Is it clear?"]

    def test_unpunctuated_speech_splits_on_pauses(self):
        a = seg(0.0, 2.0, "यह पहला भाग है")
        b = seg(3.5, 5.0, "क्या आप समझे")
        texts = [x[2] for x in split_sentences([a, b])]
        assert texts == ["यह पहला भाग है", "क्या आप समझे"]

    def test_sentence_split_uses_word_times(self):
        s = seg(0.0, 6.0, "This is a cell. What does it contain?")
        parts = split_sentences([s])
        assert [p[2] for p in parts] == ["This is a cell.", "What does it contain?"]
        assert parts[1][0] > parts[0][0]

    def test_hindi_cues_do_not_fire_inside_words(self):
        assert not is_question("क्योंकि जब भी हम काम करते हैं")     # "because", starts with क्यों
        assert not is_question("बच्चे को दूध पिलाना")                  # ends with ना inside a word
        assert is_question("दूध पिलाना जरूरी है ना")

    def test_wait_time_is_gap_to_next_word(self):
        t = transcript(seg(0.0, 3.0, "What is a cell?"), seg(8.0, 10.0, "A cell is the unit of life."))
        (q,) = find_questions(t)
        assert q.wait_time_s == 5.0


class TestTeacherMetrics:
    def test_core_metrics(self):
        t = transcript(
            seg(0.0, 30.0, " ".join(["word"] * 60) + "."),
            seg(35.0, 40.0, "Any questions?"),
            seg(50.0, 110.0, " ".join(["word"] * 120) + "."),
        )
        speech = [SpeechSegment(0.0, 30.0), SpeechSegment(35.0, 40.0), SpeechSegment(50.0, 110.0)]
        m = compute_metrics(t, speech, duration_s=120.0)
        assert m.speech_s == 95.0
        assert m.talk_ratio == round(95 / 120, 3)
        assert m.questions == 1 and m.median_wait_time_s == 10.0
        assert m.long_pauses == 2                  # 5 s and 10 s gaps
        assert m.longest_monologue_s == 60.0
        assert m.words_per_minute == 114.9         # 182 words / 95 s

    def test_minute_timeline(self):
        t = transcript(seg(0.0, 30.0, "one two three"), seg(70.0, 80.0, "Is that clear?"))
        speech = [SpeechSegment(0.0, 30.0), SpeechSegment(70.0, 80.0)]
        df = minute_timeline(t, speech, duration_s=120.0)
        assert list(df["speech_s"]) == [30.0, 10.0]
        assert list(df["questions"]) == [0, 1]

    def test_no_speech(self):
        m = compute_metrics(transcript(), [], duration_s=60.0)
        assert m.speech_s == 0 and m.words_per_minute == 0 and m.median_wait_time_s is None


class TestRoomNoise:
    def test_detects_loud_chatter_and_spike(self):
        rng = np.random.default_rng(0)
        secs = 120
        audio = 0.003 * rng.standard_normal(secs * SAMPLE_RATE).astype(np.float32)   # quiet room
        t = np.arange(SAMPLE_RATE * 20) / SAMPLE_RATE
        voices = sum(np.sin(2 * np.pi * f * t) for f in (400, 900, 1700)) * 0.05     # voice-band noise
        audio[60 * SAMPLE_RATE: 80 * SAMPLE_RATE] += voices.astype(np.float32)
        audio[100 * SAMPLE_RATE: 100 * SAMPLE_RATE + 1600] += 0.8                     # bang
        curve = noise_curve(audio)
        assert len(curve) == secs
        assert curve.loc[62:78, "noisy"].mean() > 0.8
        assert curve.loc[10:50, "noisy"].mean() == 0
        assert bool(curve.loc[100, "spike"])
