# Exam proctoring mode (2026-10-01)

## What exists
- `classroom/exam.py`: `ExamMonitor` turns per-frame, per-student signals into reviewable flags.
  - Signals: `gaze_neighbour`, `talking` (mutual facing), `turned_around`, `reaching` (hand moving towards a neighbour), `device` (phone-class object near a wrist), `out_of_seat`.
  - **Every signal is relative to the student's own normal** (running median of their head direction and wrist position). Absolute rules flagged ordinary posture: students at row ends sit angled, and perspective and handedness put a writing wrist "towards" the next row.
  - Only students matched to their seat *in this frame*, at least 80 px tall, and with no invigilator within 1.5 box-widths are assessed. The rest are counted as not assessable, not flagged on noise.
  - Minimum durations (gaze 3 s, reaching/device 2 s, talking/turned 2 s, out of seat 3 s), episodes merged across ≤ 2 s gaps, score = duration vs. minimum × signal strength.
  - Room mic: class-level `room_talking` flags from `audio.room.noise_curve`, when the track isn't silent.
- `classroom/pipeline.py --mode exam`: identity uses the `exam` consent purpose; writes `exam_flags.json`, `exam_summary.json` and `flags/<id>.mp4` review clips (top `--max-clips`, ±3 s, H.264). Clips box the flagged student in yellow and ring the neighbour's seat in orange. **Only the flagged student can appear unblurred, and only if identified under exam consent**; everyone else stays blurred. `--allow-calculators` disables device flags (COCO "cell phone" fires on calculators).
- API (`api/routes/lessons.py`): `GET /lessons/{id}/exam`, `GET …/exam/clips/{flag}`, `POST …/exam/reviews` (confirmed / dismissed / unclear + note). Roles: admin, principal, teacher, invigilator. Invigilators never see student names; teachers and above see names only under exam consent. Every view and review is audit-logged.
- Dashboard: **Exam review** page (`/lessons/:id/exam`, linked from the lesson page and list): "prompts, not evidence" banner, summary tiles, type filters, queue sorted by score, clip, details, Confirm / Dismiss / Unclear with keyboard shortcuts (c / d / u) and a note.
- Tests: `tests/unit/test_exam.py`, 11 tests (133 total, all passing). Verified in the browser: queue, clip playback, keyboard review, audit entry.

## Results on the Altima exam (72 s, 24 students, calm exam)
| Iteration | Flags | Per student-hour | What was wrong |
|---|---|---|---|
| First rules | 110 | 228 | "reaching" fired on ordinary writing (front/back rows counted as neighbours) |
| Same-row reaching | 85 | 176 | perspective-diagonal rows: "same row" still spans several physical rows |
| Wrist baseline per student | 57 | 118 | gaze: row-end students angled all exam; invigilator inherits a seat while walking past |
| + yaw baseline, at-seat, size gate | 31 | 64 | invigilator overlap scrambles keypoints; 1–2 s jitter |
| + invigilator suppression, longer minimums | **21** (5 high priority) | **43** | |

**I reviewed the top-scored and all 5 high-priority flags by eye: all are false alarms** (students writing normally, a calculator on the desk, or tiny back-row students). This was a calm exam, so precision here is effectively 0 and recall can't be measured: there are no real incidents to find.

### What this means
- **The workflow is ready; the detector is not.** At ~43 flags per student-hour, a 3-hour exam with 30 students would produce thousands of prompts. That's unusable, and it would erode invigilators' trust.
- The main limit is **camera geometry**: near-vertical cameras show the tops of heads, so head direction and hands are guesswork. Exam mode needs a **front-facing camera at a lower angle** (or two angles) to be judged fairly.
- Next layer: **VLM verification** of each candidate clip (SentinelCore already has an Ollama VLM worker; a local `qwen2.5vl:7b` would need a ~6 GB download), and a learned classifier once confirmed/dismissed reviews accumulate. **The review decisions are the training labels**, so the review page is also the data engine.
- Success bar before any school pilot: **< 1 flag per student-hour on calm exams**, measured, plus recall measured on **staged incidents** (volunteers, with consent) on the pilot's camera setup.

## Limits
- No labelled cheating footage; recall unmeasured.
- Device flags can't separate phones from calculators; use `--allow-calculators` where calculators are permitted.
- Top-down faces looking straight down aren't detected by the face blurrer, so partial faces can show in clips; clips are for authorized staff only.
