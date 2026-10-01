# AI Smart Classroom

Camera and audio analytics for Indian schools: class engagement over a lesson, which topics and teaching methods hold attention, teacher coaching metrics, and an exam-proctoring review queue. Built on the SentinelCore multi-camera engine.

> **Status: research prototype, not pilot-ready.** Each part's measured accuracy is documented honestly in `docs/`. In short: the class-level engagement curve tracks labels well on a normal classroom camera; per-student states and exam flags do not yet meet the bar for use with real students.

## What it does

| Part | Code | Status |
|---|---|---|
| Classroom vision: seat map, head pose, per-student states, class engagement curve | `classroom/` | Class curve within 1–8 points of labels; per-student ~65% (`docs/step8_status.md`) |
| Teacher audio: ASR (faster-whisper), talk ratio, pace, questions, wait time | `audio/` | Works with a lapel mic; CCTV room mics can't be transcribed (`docs/step5_status.md`) |
| Topics: local LLM (Ollama qwen3) segmentation + question classification | `topics/` | `docs/step6_status.md` |
| Consent-aware identity: consent registry, encrypted face gallery, seat voting, blurring, purge | `classroom/consent.py`, `identity.py`, `privacy.py` | Recognition untested on real consented people (`docs/step4_status.md`) |
| Dashboard: lessons, seat map, teacher report, student view, labelling, exam review | `api/`, `dashboard/` | `docs/step7_status.md` |
| Exam proctoring: baseline-relative flags, blurred review clips, review queue | `classroom/exam.py` | All flags on a calm test exam were false alarms (`docs/exam_mode_status.md`) |
| Ground truth: labelling tool + evaluation | `labeling/` | `docs/labeling_guide.md` |

## Privacy by design
- **Consent first** (DPDP Act 2023 §9): students are identifiable only with verified guardian consent for the specific purpose (engagement, exam). Without it they're anonymous seats and, in any export, blurred.
- **Video and audio stay on the school's server**; only metadata leaves it. LLM/VLM models run locally (Ollama).
- **Exports are blurred by default** (heads + whole-frame face detection); there's no switch to turn that off.
- Every identity operation, view and review is audit-logged; `classroom.privacy purge` removes a student everywhere.
- Face models are commercially licensed (OpenCV YuNet MIT, SFace Apache-2.0); SentinelCore's InsightFace default is non-commercial and not used by the classroom path.
- Flags and per-student states are prompts for a teacher's judgement, never automated verdicts or rankings.

No footage, outputs, model weights, face data or secrets are in this repository (see `.gitignore`).

## Run locally
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # fill in secrets
# Lesson analysis on a recorded video
.venv/bin/python -m classroom.pipeline --video footage/<lesson>.mp4 --preview
# Teacher audio, then topics (needs Ollama with qwen3:8b)
.venv/bin/python -m audio.pipeline --input <lecture>.mp4 --language hi
.venv/bin/python -m topics.segment --transcript outputs/<lecture>/audio/transcript.json
# Exam mode
.venv/bin/python -m classroom.pipeline --video footage/<exam>.mp4 --mode exam
# Tests
.venv/bin/python -m pytest tests/unit
```
Dashboard: Postgres + Redis (see `docs/step7_status.md`), `uvicorn api.main:app`, `npm --prefix dashboard run dev`.

## Background
The engine (`sentinel_engine/`, `api/`, `dashboard/` base) comes from SentinelCore (Smartvision1.0.0). Audio settings were adapted from opensource-clipping-autopilot. Test footage came from public YouTube CCTV demos and Creative Commons NPTEL lectures, used for internal R&D only and not redistributed.
