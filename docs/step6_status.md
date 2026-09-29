# Step 6 status: topics, questions, lesson join (2026-09-29)

## What exists
- `topics/llm.py`: Ollama chat client with JSON-schema structured output (`think: false` for qwen3), plus SentinelCore's raw_decode fallback. The model is **qwen3:8b** running locally, so transcripts stay on the machine / edge server.
- `topics/segment.py`: two focused LLM passes per 90-line chunk:
  - **Topics**: contiguous segments with English title, summary, teaching method (explanation / example / demonstration / questioning / discussion / recap / activity / administration / other) and keywords. The previous chunk's topic is passed forward so topics continue across chunks. Segments < 60 s fold into a neighbour.
  - **Questions**: candidate lines (English: "?"/cue lines; Hindi: every line) shown with the following line; the model must choose not_a_question / open / closed / rhetorical / check_understanding. Each question then gets an objective **wait time** (gap to the teacher's next word) and an **effective type**: open/closed questions followed within 1.5 s are counted as rhetorical in practice.
- `analytics/lesson.py`: joins topics × transcript × questions × (optional) `class_engagement.csv` into `lesson_topics.csv`, `lesson_methods.csv` and `lesson_report.md`. It reports per-topic pace, questions by type, engaged share, difference from the lesson average and the main disengagement state, plus time-weighted engagement by teaching method. `--engagement-offset` aligns video and audio clocks.
- Tests: `tests/unit/test_topics.py`, 13 tests with a fake LLM (76 total, all passing).

Run:
```
.venv/bin/python -m topics.segment --transcript outputs/<lesson>/audio/transcript.json --context "<class, subject, lesson plan>"
.venv/bin/python -m analytics.lesson --lesson outputs/<lesson> [--engagement outputs/<video>/class_engagement.csv]
```

## Results (qwen3:8b on M4)
| Lecture | Time | Topics | Questions |
|---|---|---|---|
| Pharmacognosy L1, English, 26 min | 2.7 min | 10, following the lecture's arc (history → scope → examples → validation → nutraceuticals → excipients → cosmetics) | 16, all rhetorical in practice (waits 0.0–0.8 s) |
| Nutrition intro, Hindi, 9 min | 4.7 min | 2 (background; audience and goals) | 0 (monologue course intro) |

## What went wrong and was fixed
1. **One combined prompt** → qwen3:8b labelled 58 lines as questions, most of them statements. Splitting into a topic pass and a forced per-candidate verdict gave 16.
2. **Model bias to "open"**: it called the lecturer's self-answered questions "open". The wait-time rule overrides this with an objective signal (teacher wait time, a standard teaching-quality measure).
3. **Over-split Hindi**: 28 long unpunctuated lines → 9 topics of 20–30 s. Fixed by the 60 s minimum.

## Limits
- **No real paired lesson yet.** The NPTEL lectures have no students on camera, and the Securus classroom clip has no usable audio, so the topic × engagement join is verified by unit tests only. Its first real output needs a pilot lesson recorded with a student-facing camera and a teacher lapel mic.
- Hindi: 4.7 min of LLM time for 9 min of lesson (Devanagari is token-heavy). That's fine offline after class; tune chunk size on the edge GPU.
- Board/slide OCR from the rear camera (planned as a supporting topic source) isn't built. Useful when the teacher is quiet (board work, tests).
- Topic titles are in English even for Hindi lessons, by design, for cross-school aggregation. The summaries could also be kept in the lesson language for teachers.
