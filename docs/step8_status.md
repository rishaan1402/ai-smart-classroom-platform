# Step 8 status: labelling + evaluation (2026-10-01)

## What exists
- `labeling/tasks.py`: uniform sampling of (seat, time) moments (≤ 1 per seat per 5 s), exported as 3-frame strips (t−1 s, t, t+1 s) of the student's region. **Every face in the frame is pixelated before cropping** (coarse 7-block mosaic, detected on the full frame); a Gaussian blur was tried first but hid head orientation, which labellers need. Append-only label store; latest label per (task, labeller) wins. The model's prediction never leaves the server.
- `api/routes/labeling.py` + **Label** page in the dashboard: task sets with per-labeller progress, the strip, 10 labels with one-key shortcuts (1–9, 0) and inline definitions. Roles: admin / principal / teacher / labeler.
- `labeling/evaluate.py`: accuracy, macro-F1, per-state P/R/F1, confusion matrix, engaged-vs-not accuracy, engaged share by labels vs. predictions, detection-error rate (`not_a_student`), Cohen's κ between labellers, and a breakdown by person size (resolution bias check).
- `docs/labeling_guide.md`: label definitions and rules of thumb.
- Tests: `tests/unit/test_labeling.py` (111 total, all passing). Verified in the browser: task list → labelling page → keyboard label → next task.

## First results: provisional
**Labeller: `claude-provisional`, a single non-human labeller.** These numbers give a first read on where the pipeline stands; they are not ground truth. They need two human labellers (with κ measured) before anyone quotes them.

| Lesson | Tasks (scored) | State accuracy | Engaged-vs-not accuracy | Class engaged share: labels vs. predicted |
|---|---|---|---|---|
| Securus, Indian primary class | 96 (91) | **58%** | **73%** | **78% vs. 79%** |
| Altima, exam hall (top-down) | 45 (37) | **27%** | **49%** | **100% vs. 49%** |

Full tables: `docs/eval_cctv_securus_2mp_audio.md`, `docs/eval_cctv_altima.md`.

### What this says
1. **The class-level curve on a normal classroom camera is usable**: on Securus the engaged share matches the labels within 1 point, because per-student errors largely cancel out. That's what the lesson and topic views are built on.
2. **Per-student states are not reliable yet** (58%). Writing vs. head-down vs. attentive are confused both ways, and every raised hand was missed (0 of 3). **Per-student reports should not ship to parents or teachers until a learned classifier beats this.**
3. **Top-down exam cameras break the keypoint rules**: on Altima all 37 scored students were writing, yet 17 were called head down and 8 attentive. This matches the step 3 suspicion; exam mode needs a different model (and a different camera position).
4. **`out_of_seat` is a false alarm** on Securus (5 predicted, 0 true): seat-assignment jitter, not real movement. Needs a minimum duration and a displacement check.
5. Resolution: on Securus, back rows scored *higher* (71%) than front rows (50%). The back rows here are small children sitting still, front rows move more, so this is about behaviour mix, not just pixels. Not a fairness result.

## Privacy notes found while labelling
- Top-down views: faces looking straight down are not detected, so some partial faces in Altima strips are not pixelated. Strips are for authorized internal labellers only. Before using pilot footage, add a top-of-head blur for steep cameras or review strips before release.
- Labelling students' footage for model improvement is a **separate consent purpose**. Add it to the consent form; only export lessons whose students consented to it.

## Next
- **Human labels**: two labellers on these 141 tasks (≥ 20% overlap) → κ, and real accuracy numbers.
- **More data**: a learned per-student classifier needs on the order of 2–5k labelled strips across several classrooms. The VLM pre-labelling idea from the plan fits here: have a VLM propose labels, humans confirm, which cuts labelling time.
- Quick wins in the rules: a duration/displacement gate for `out_of_seat`; hand-raise at face level for small children.
