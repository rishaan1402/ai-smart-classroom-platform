# Step 4 status: consent-aware identity (2026-09-30)

## What exists
- `classroom/consent.py`: consent registry (CSV: status granted / pending / withdrawn, purposes engagement / exam, how guardian consent was verified) and an append-only JSONL audit log. Only `granted` + the requested purpose makes a student identifiable; pending and withdrawn are treated exactly like unknown.
- `classroom/faces.py`: face detection + embedding with **OpenCV Zoo YuNet (MIT) + SFace (Apache-2.0)**, swappable via the `FaceEmbedder` protocol. The models live in `models/` (git-ignored). Download:
  ```
  curl -sSL -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  curl -sSL -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
  ```
- `classroom/identity.py`:
  - `Gallery`: consented students' embeddings, **Fernet-encrypted at rest** (`FACE_ENCRYPTION_KEY`, as in SentinelCore), searched via SentinelCore's `FaceIndex`, and re-filtered against current consent on every run.
  - `SeatIdentifier`: per-seat face votes across the lesson, resolved with the seating chart into `face+chart` / `face` / `chart` / `conflict` (flagged, stays anonymous) / `none`. Only (student, similarity) votes are kept; **embeddings of faces are discarded immediately**, matched or not.
  - `blur_heads` + `FaceBlurrer`: exports blur every head not identified as a consented student, both tracked people and any face found by whole-frame face detection (catches people the person detector missed).
- `classroom/privacy.py`: CLI `enroll` (refused and audit-logged without consent), `purge` (gallery, every output's student_id, seat identities; consent marked withdrawn), `list`.
- `classroom/pipeline.py`: identity runs automatically after reasoning. With no consented enrollments, **no face is detected or embedded for identification**. Writes `seat_identity.json`, adds `student_id` to tokens and seat timelines for identified seats only.
- `sentinel_engine/workers/face.py`: fixed `FaceIndex.add()`, which stored un-normalized vectors while `search()` assumes unit vectors.
- Templates: `docs/templates/consent_registry.example.csv`, `docs/templates/seating_chart.example.csv`.
- Tests: `tests/unit/test_identity.py`, 20 tests with a fake embedder (96 total, all passing).

## Verified on real footage
- Securus classroom with no registry/gallery: 23/23 seats anonymous, audit entry written, and in the preview every child's head is blurred, including back-row children the person detector missed (after adding whole-frame face blurring; the first version leaked those).

## Not yet verified
- **Recognition accuracy on real faces.** Deliberately not tested on the YouTube footage: those children haven't consented, so they're never enrolled or embedded. It needs consenting people, e.g. your own team: enroll 3–5 photos each, record a short clip seated in rows, run the pipeline with a seating chart, and check `seat_identity.json`.
- SFace is a lighter model than ArcFace-R100. At classroom CCTV face sizes (~20–30 px) expect face evidence only for the front and middle rows; the seating chart carries the back rows (`source: chart`). Measure this in the pilot before promising per-student accuracy.

## Decisions to raise
- **InsightFace licence**: SentinelCore's FaceWorker defaults to InsightFace `buffalo_l`, whose pretrained weights are for non-commercial research only. The classroom path doesn't use it; the SentinelCore default should be switched or licensed before any commercial deployment.
- Exports are blurred by default, and there's intentionally no switch to turn that off. Unblurred video should only be viewable inside the school, by authorized staff, and audit-logged (dashboard, step 7).

## Run
```
export FACE_ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
cp docs/templates/consent_registry.example.csv data/consent/registry.csv      # then edit
.venv/bin/python -m classroom.privacy enroll --student S0001 --photos roster/S0001/
.venv/bin/python -m classroom.pipeline --video footage/<lesson>.mp4 --seating-chart data/seating/8A.csv --preview
.venv/bin/python -m classroom.privacy purge --student S0001
```
