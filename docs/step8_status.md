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

## Rule fixes (same day), re-scored on the same labels
Scored with `labeling.evaluate --repredict` against the same 141 provisional labels. These labels were also used to diagnose the problems, so this is a sanity check, not a held-out result.

| Securus | Before | After |
|---|---|---|
| State accuracy | 58% | **64%** |
| Engaged-vs-not accuracy | 73% | **79%** |
| `out_of_seat` false alarms (in sample) | 5 | **0** |
| `out_of_seat` share of all observations | 6.1% | 0.1% |
| Raised hands found | 0 / 3 | 0 / 3 |
| Class engaged share, labels vs. predicted | 78% vs. 79% | 78% vs. **86%** |

1. **Out of seat** now needs ≥ 3 s continuously without a seat match **and** an anchor ≥ 2 seat-radii from the student's usual seat (`SeatTracker.is_out_of_seat`). Seat-assignment jitter no longer fires it.
2. **Face-level hand raise** (wrist above the shoulder line, elbow lifted, hand beside the face, not on the chin) is in, with tests, but **found none of the 3 labelled raises**. Keypoint inspection shows why: at 20–45 px shoulder width the pose model either loses the raised arm (confidence 0.2) or puts the wrist at or below the shoulder while the hand is at the face. Loosening the threshold to "wrist level with the shoulder" added a wrong prediction and still found none, so it was reverted. Fixing this needs pose on upscaled per-student crops or a learned classifier, not thresholds.
3. **Side effect**: the false out-of-seat calls had been offsetting the opposite error (writing and head-down students called attentive), so the class-level share now overestimates engagement by 8 points on this sample (≈ 2 standard errors at n = 91). Per-moment accuracy is better; the aggregate is less calibrated. Watch this when human labels arrive. The real fix is the writing/head-down confusion, not reintroducing a false alarm.
4. Also fixed: the detection cache was pickled from `__main__` when the pipeline ran via `-m`, so other modules couldn't load it. The entry point now runs through the importable module; stale caches are recomputed instead of crashing; existing caches were migrated.

Altima is unchanged (27%); its errors are the top-down writing confusion, which these fixes don't touch.

## Privacy notes found while labelling
- Top-down views: faces looking straight down are not detected, so some partial faces in Altima strips are not pixelated. Strips are for authorized internal labellers only. Before using pilot footage, add a top-of-head blur for steep cameras or review strips before release.
- Labelling students' footage for model improvement is a **separate consent purpose**. Add it to the consent form; only export lessons whose students consented to it.

## Next
- **Human labels**: two labellers on these 141 tasks (≥ 20% overlap) → κ, and real accuracy numbers.
- **More data**: a learned per-student classifier needs on the order of 2–5k labelled strips across several classrooms. The VLM pre-labelling idea from the plan fits here: have a VLM propose labels, humans confirm, which cuts labelling time.
- Quick wins in the rules: a duration/displacement gate for `out_of_seat`; hand-raise at face level for small children.
