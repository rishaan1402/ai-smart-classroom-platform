# Step 3 status: seat map, head pose, classroom states (2026-09-29)

## What exists
- `classroom/seatmap.py`: learns seats from where people stay (scale-aware density peaks), Hungarian track↔seat matching, per-seat radii, JSON persistence, and `SeatTracker` for role (student vs. other) and seat smoothing.
- `classroom/headpose.py`: head yaw / pitch / ear-lift / back-turned from COCO pose keypoints. No crop-based gaze (L2CS): faces are ~20–30 px in classroom CCTV.
- `classroom/states.py`: rule classifier (out of seat, hand raised, phone, turned away, writing, head down, interacting, attentive) with a **local** facing consensus (nearest classmates) and a **camera-wide** upright baseline, plus a temporal majority smoother.
- `classroom/pipeline.py`: offline runner → `tokens.parquet`, `seat_timeline.parquet`, `class_engagement.csv`, `seatmap.json`, `summary.json`, `preview.mp4`; caches pass-1 detections.
- `tests/unit/test_classroom.py`: 21 tests (52 total, all passing).

Run: `.venv/bin/python -m classroom.pipeline --video footage/<clip>.mp4 --preview`

## Results on the test footage (unlabelled, judged by eye from previews)
| Clip | Seats | State mix | Verdict |
|---|---|---|---|
| Securus (Indian primary class, corner CCTV) | 23 (~25 kids) | 59% attentive, 23% writing, 6% head down, 6% out of seat, 6% turned away | Plausible. The teacher is correctly excluded; the child lying on the desk is caught |
| Altima (exam hall, near top-down) | 26 | 29% writing, 36% head down, 24% attentive | Most "head down" and "attentive" should be writing |
| Lolo 5 min (silent study, desk-level) | 23 (of ~45 students) | 29% writing, 27% head down, 14% attentive, 12% turned away | Detection covers only ~1/3 of the room; writing is under-called |

## Known limits of the keypoint rules
1. **Writing vs. resting** can't be separated reliably from keypoints at steep angles: wrists on paper look like "head on arms", and a nose over paper looks like a head on the desk.
2. **Raised hands at face height** (small children) are missed.
3. **Phones vs. calculators**: COCO phone detections fire on exam calculators. Wrist proximity and conf 0.45 cut this, but it isn't solved.
4. **Dense/desk-level scenes**: YOLO11m-pose at 1280 px finds ~10–14 of ~45 students. Needs tiled inference or a higher-mounted camera.
5. **Track churn**: 82 tracker IDs for ~25 kids on Securus. Seat-keyed timelines absorb most of it.

## Next
- **Labels (plan step 8, pulled forward)**: 2–3 lessons, ~1 label per seat per 5 s, to measure the rules and tune thresholds against numbers, not previews.
- **Learned classifier on seat crops** (fine-tuned small video/VLM model) for writing vs. resting, phone vs. calculator and hand raises: the cases the rules can't separate.
- Tiled inference for back rows.
