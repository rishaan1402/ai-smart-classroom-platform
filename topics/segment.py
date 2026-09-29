"""
Topic segmentation + question classification of a lesson transcript.

The transcript is split into sentences (audio.teacher.split_sentences), numbered,
and sent to the LLM in chunks of CHUNK_LINES, in two focused passes:
  topics     contiguous segments: start line, English title, one-line summary,
             teaching method, keywords. Each chunk gets the previous chunk's
             last topic, so a topic spanning a chunk boundary is continued.
  questions  candidate lines, each with the line that follows, judged as
             not_a_question / open / closed / rhetorical / check_understanding.
             Candidates are the "?"/cue-word lines for English (Whisper
             punctuates English reliably) and every line otherwise, since
             Whisper rarely punctuates Hindi.
A single combined pass let an 8B model label most statements "open questions";
a forced per-line verdict with a not_a_question option fixes that.

The model's open/closed/rhetorical call is still biased towards "open", so each
question also gets an objective wait time (gap to the teacher's next word) and
an effective_type: any question followed within MIN_WAIT_S counts as
rhetorical in practice, since students had no window to answer.

Line numbers are mapped back to timestamps, so topics join the per-second
engagement curve and the per-minute audio timeline.

Usage:
  .venv/bin/python -m topics.segment \\
      --transcript outputs/nptel_iitb_pharmacognosy_l1/audio/transcript.json \\
      --context "B.Pharm, Pharmacognosy and Phytochemistry, Week 1 Lecture 1"
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from audio.asr import Transcript
from audio.teacher import is_question, split_sentences

from .llm import DEFAULT_MODEL, OllamaClient

logger = logging.getLogger(__name__)

CHUNK_LINES = 90
MIN_TOPIC_S = 60.0   # shorter segments are merged into a neighbour
MIN_WAIT_S = 1.5   # shorter than this after a question → no real chance for students to answer

METHODS = [
    "explanation", "example", "demonstration", "questioning", "discussion",
    "recap", "activity", "administration", "other",
]
QUESTION_TYPES = ["open", "closed", "rhetorical", "check_understanding"]

_TRANSCRIPT_INTRO = """\
You analyse transcripts of school and college lessons in India. Transcripts may be English, \
Hindi (Devanagari) or code-mixed, and come from automatic speech recognition, so expect some \
garbled words. Each line is "L<number> [mm:ss] <text>" spoken by the teacher."""

TOPIC_PROMPT = _TRANSCRIPT_INTRO + """

Split the lines into contiguous topic segments, in order, together covering every line. \
A new segment starts where the subject matter or teaching activity clearly changes; do not \
split on small digressions. Typical segments last 2-10 minutes.
For each: start_line (the L number where it begins), title (English, max 8 words), \
summary (one English sentence), method (the dominant teaching method), keywords (max 5, English).
Set continues_previous=true on the first segment only if it continues the previous topic given below.
Use only line numbers that appear in the input. Return JSON only."""

QUESTION_PROMPT = _TRANSCRIPT_INTRO + """

You get candidate lines, each followed by "next:" (what the teacher says right after). \
For every candidate decide what it is:
- not_a_question: a statement, instruction or fragment (most lines are this)
- rhetorical: a question the teacher answers themselves, usually in the next line. \
Most questions in lectures are rhetorical.
- check_understanding: "right?", "okay?", "samjhe?", "any doubts?" style checks
- closed: a genuine question to students with one short factual answer
- open: a genuine question to students inviting explanation or ideas
Give text_en, a short English rendering. Judge every candidate line. Return JSON only."""


def _topic_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_line": {"type": "integer"},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "method": {"type": "string", "enum": METHODS},
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "continues_previous": {"type": "boolean"},
                    },
                    "required": ["start_line", "title", "summary", "method", "keywords", "continues_previous"],
                },
            },
        },
        "required": ["topics"],
    }


def _question_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "line": {"type": "integer"},
                        "type": {"type": "string", "enum": ["not_a_question", *QUESTION_TYPES]},
                        "text_en": {"type": "string"},
                    },
                    "required": ["line", "type", "text_en"],
                },
            },
        },
        "required": ["verdicts"],
    }


@dataclass
class Line:
    number: int
    start: float
    end: float
    text: str


@dataclass
class Topic:
    start_s: float
    end_s: float
    title: str
    summary: str
    method: str
    keywords: list[str] = field(default_factory=list)
    start_line: int = 0


@dataclass
class ClassifiedQuestion:
    time_s: float
    line: int
    type: str            # the LLM's reading of intent
    text: str
    text_en: str
    wait_time_s: float = 0.0
    effective_type: str = ""   # type, downgraded to rhetorical when the wait is < MIN_WAIT_S


def build_lines(transcript: Transcript) -> list[Line]:
    return [
        Line(i, round(s, 2), round(e, 2), t.strip())
        for i, (s, e, t) in enumerate(split_sentences(transcript.segments), start=1)
        if t.strip()
    ]


def format_lines(lines: list[Line]) -> str:
    return "\n".join(f"L{ln.number} [{int(ln.start // 60):02d}:{int(ln.start % 60):02d}] {ln.text}" for ln in lines)


def merge_chunk(topics: list[Topic], chunk: list[Line], raw: dict, lesson_end_s: float) -> list[Topic]:
    """
    Validate one chunk's LLM output and append it to the running topic list.

    Enforces: segments start inside the chunk, in order, the first at the
    chunk's first line; continues_previous extends the last topic instead of
    opening a new one.
    """
    by_number = {ln.number: ln for ln in chunk}
    first, last = chunk[0].number, chunk[-1].number

    segs = sorted(
        (t for t in raw.get("topics", []) if first <= int(t.get("start_line", -1)) <= last),
        key=lambda t: int(t["start_line"]),
    )
    deduped = []
    for t in segs:
        if deduped and int(t["start_line"]) == int(deduped[-1]["start_line"]):
            continue
        deduped.append(t)
    if not deduped:
        deduped = [{"start_line": first, "title": "Unsegmented", "summary": "", "method": "other",
                    "keywords": [], "continues_previous": False}]
    deduped[0]["start_line"] = first

    for i, t in enumerate(deduped):
        if i == 0 and topics and (t.get("continues_previous") or _same_title(t["title"], topics[-1].title)):
            continue    # extends the previous topic; its end is set below
        method = t.get("method") if t.get("method") in METHODS else "other"
        topics.append(Topic(
            start_s=by_number[int(t["start_line"])].start,
            end_s=lesson_end_s,
            title=t.get("title", "").strip() or "Untitled",
            summary=t.get("summary", "").strip(),
            method=method,
            keywords=[k.strip() for k in t.get("keywords", [])][:5],
            start_line=int(t["start_line"]),
        ))
    for a, b in zip(topics[:-1], topics[1:]):
        a.end_s = b.start_s

    return topics


def question_candidates(chunk: list[Line], language: str) -> list[Line]:
    """Lines worth asking the LLM about: punctuation/cue hits in English, everything otherwise."""
    if language == "en":
        return [ln for ln in chunk if is_question(ln.text)]
    return list(chunk)


def format_candidates(candidates: list[Line], lines_by_number: dict[int, Line]) -> str:
    out = []
    for ln in candidates:
        nxt = lines_by_number.get(ln.number + 1)
        out.append(f"L{ln.number} [{int(ln.start // 60):02d}:{int(ln.start % 60):02d}] {ln.text}\n"
                   f"   next: {nxt.text if nxt else '(end of lesson)'}")
    return "\n".join(out)


def parse_verdicts(candidates: list[Line], raw: dict) -> list[ClassifiedQuestion]:
    by_number = {ln.number: ln for ln in candidates}
    questions, seen = [], set()
    for v in raw.get("verdicts", []):
        n = int(v.get("line", -1))
        if n in by_number and n not in seen and v.get("type") in QUESTION_TYPES:
            seen.add(n)
            ln = by_number[n]
            questions.append(ClassifiedQuestion(ln.start, n, v["type"], ln.text, v.get("text_en", "").strip()))
    return questions


def _same_title(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def segment_transcript(
    transcript: Transcript,
    client: OllamaClient,
    context: Optional[str] = None,
    chunk_lines: int = CHUNK_LINES,
    min_topic_s: float = MIN_TOPIC_S,
) -> tuple[list[Topic], list[ClassifiedQuestion], list[Line]]:
    lines = build_lines(transcript)
    lesson_end = lines[-1].end if lines else transcript.duration_s
    topics: list[Topic] = []
    questions: list[ClassifiedQuestion] = []

    lines_by_number = {ln.number: ln for ln in lines}
    ctx = f"Lesson context: {context or 'not provided'}\n"

    for i in range(0, len(lines), chunk_lines):
        chunk = lines[i: i + chunk_lines]
        span = f"L{chunk[0].number}–L{chunk[-1].number}"
        started = time.time()

        previous = f'"{topics[-1].title}": {topics[-1].summary}' if topics else "none (start of lesson)"
        user = f"{ctx}Previous topic: {previous}\n\nTranscript lines {span}:\n{format_lines(chunk)}"
        raw = _safe_call(client, TOPIC_PROMPT, user, _topic_schema(), f"topics {span}")
        topics = merge_chunk(topics, chunk, raw, lesson_end)

        candidates = question_candidates(chunk, transcript.language)
        chunk_questions: list[ClassifiedQuestion] = []
        if candidates:
            user = f"{ctx}\nCandidate lines:\n{format_candidates(candidates, lines_by_number)}"
            raw_q = _safe_call(client, QUESTION_PROMPT, user, _question_schema(), f"questions {span}")
            chunk_questions = parse_verdicts(candidates, raw_q)
        questions.extend(chunk_questions)
        logger.info(
            f"  chunk {span}: {len(raw.get('topics', []))} segments, {len(candidates)} candidates → "
            f"{len(chunk_questions)} questions ({time.time() - started:.0f}s)"
        )
    topics = merge_short_topics(topics, min_topic_s)
    add_wait_times(questions, lines, transcript)
    return topics, questions, lines


def merge_short_topics(topics: list[Topic], min_s: float = MIN_TOPIC_S) -> list[Topic]:
    """
    Fold segments shorter than min_s into the previous topic (the next one for
    the first topic). The LLM over-splits when lines are long and few, e.g.
    unpunctuated Hindi where each "sentence" is ~30 s of speech.
    """
    merged: list[Topic] = []
    for t in topics:
        if merged and (t.end_s - t.start_s) < min_s:
            prev = merged[-1]
            prev.end_s = t.end_s
            prev.keywords = list(dict.fromkeys(prev.keywords + t.keywords))[:5]
        elif merged and (merged[-1].end_s - merged[-1].start_s) < min_s and len(merged) == 1:
            # A short opening topic is absorbed by the one that follows it
            first = merged.pop()
            t.start_s, t.start_line = first.start_s, first.start_line
            t.keywords = list(dict.fromkeys(first.keywords + t.keywords))[:5]
            merged.append(t)
        else:
            merged.append(t)
    return merged


def add_wait_times(questions: list[ClassifiedQuestion], lines: list[Line], transcript: Transcript) -> None:
    """Gap from the question's end to the next spoken word, and the effective type it implies."""
    word_starts = sorted(w.start for s in transcript.segments for w in s.words)
    ends = {ln.number: ln.end for ln in lines}
    for q in questions:
        end = ends[q.line]
        nxt = next((t for t in word_starts if t >= end - 0.05), None)
        q.wait_time_s = round(max(0.0, nxt - end), 2) if nxt is not None else 0.0
        q.effective_type = (
            "rhetorical" if q.type in ("open", "closed") and q.wait_time_s < MIN_WAIT_S else q.type
        )


def _safe_call(client: OllamaClient, system: str, user: str, schema: dict, what: str) -> dict:
    """One LLM call; an empty result on failure keeps the rest of the lesson usable."""
    try:
        return client.chat_json(system, user, schema)
    except Exception as e:
        logger.warning(f"LLM failed on {what}: {e}")
        return {}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--context", default=None, help="Class / subject / lesson plan text")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=None, help="Default: <transcript dir>/../topics")
    parser.add_argument("--chunk-lines", type=int, default=CHUNK_LINES)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    transcript = Transcript.from_dict(json.loads(args.transcript.read_text()))
    out = args.out or args.transcript.parent.parent / "topics"
    out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    topics, questions, lines = segment_transcript(transcript, OllamaClient(args.model), args.context, args.chunk_lines)
    (out / "topics.json").write_text(json.dumps([asdict(t) for t in topics], ensure_ascii=False, indent=1))
    (out / "questions_llm.json").write_text(json.dumps([asdict(q) for q in questions], ensure_ascii=False, indent=1))
    logger.info(f"{len(topics)} topics, {len(questions)} questions from {len(lines)} lines "
                f"in {time.time() - started:.0f}s ({args.model})")
    for t in topics:
        logger.info(f"  {t.start_s / 60:5.1f}–{t.end_s / 60:5.1f} min  [{t.method}] {t.title}")


if __name__ == "__main__":
    main()
