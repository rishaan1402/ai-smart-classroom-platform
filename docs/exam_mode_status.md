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

**I reviewed the top-scored and all 5 high-priority flags by eye and judged them false alarms (but see the F0015 correction below)** (students writing normally, a calculator on the desk, or tiny back-row students). This was a calm exam, so precision here is effectively 0 and recall can't be measured: there are no real incidents to find.

### What this means
- **The workflow is ready; the detector is not.** At ~43 flags per student-hour, a 3-hour exam with 30 students would produce thousands of prompts. That's unusable, and it would erode invigilators' trust.
- The main limit is **camera geometry**: near-vertical cameras show the tops of heads, so head direction and hands are guesswork. Exam mode needs a **front-facing camera at a lower angle** (or two angles) to be judged fairly.
- Next layer: **VLM verification** of each candidate clip (SentinelCore already has an Ollama VLM worker; a local `qwen2.5vl:7b` would need a ~6 GB download), and a learned classifier once confirmed/dismissed reviews accumulate. **The review decisions are the training labels**, so the review page is also the data engine.
- Success bar before any school pilot: **< 1 flag per student-hour on calm exams**, measured, plus recall measured on **staged incidents** (volunteers, with consent) on the pilot's camera setup.

## VLM verification (local qwen2.5vl:7b + qwen3:8b check)
`--vlm-verify` (or `python -m classroom.vlm_verify --lesson outputs/<exam> --video …` on an existing run) sends each flag to a local VLM as one image of three marked crops. The crops are built in memory and never leave the machine. The VLM never deletes a flag; "normal" only drops it out of the default queue.

| Prompt design | Calm exam (Altima): VLM verdicts | Queue |
|---|---|---|
| Type-specific yes/no ("is the student looking at the neighbour's paper?") | 16 suspicious / 5 normal, descriptions echo the question | 16 |
| Neutral: describe, then pick from an activity list | 7 suspicious / 14 normal; 3 of the 7 contradict their own description | 7 |
| + text check of the description on every flag | 18 "cannot tell", including clear "holding a pen, writing" | 21 |
| **+ text check only re-examines suspicious labels** | **2 suspicious / 1 unclear / 18 normal** | **3** (6.2 per student-hour, from 43) |

Securus classroom run as an exam (real turning and talking, a lesson not an exam): 29 flags → **13 in queue** (10 suspicious, 3 unclear). On the blurred clips, cleared flags were a small back-row child facing forward or a child leaning over their own desk; kept ones show a child turned to the row behind and a child reaching sideways towards a neighbour's desk. That's plausible, but recall is unmeasured: nobody has labelled what really happened.

**Correction to the earlier review**: flag F0015 on Altima (the maroon-shirt student) shows, in the unblurred VLM crop, a student turned towards the neighbour holding a dark rectangular object in both hands. That could be a phone or a calculator. "All Altima flags are false alarms" was too strong; that judgement was made from small blurred images. The final VLM pass **cleared this flag as normal** ("holding the edge of a paper"). That's exactly the kind of miss a second-opinion filter can make, and why "normal" never deletes a flag.

Cost: ~40 s per flag on an M4 (qwen2.5vl:7b), so a 21-flag exam takes ~14 min after the run. That's fine overnight, too slow for live alerts.

## Limits
- No labelled cheating footage; recall unmeasured.
- Device flags can't separate phones from calculators; use `--allow-calculators` where calculators are permitted.
- Top-down faces looking straight down aren't detected by the face blurrer, so partial faces can show in clips; clips are for authorized staff only.
