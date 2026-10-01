"""
Labelling tasks: sample moments from a lesson and export blurred crops.

Sampling is **uniform** over (student token) rows, so measured accuracy
reflects what the pipeline sees in practice; rare states will have few
samples, and the report says so via per-class support. Each task gets a
3-frame strip (t − STRIP_GAP_S, t, t + STRIP_GAP_S) of the student's region,
so motion such as writing or turning is visible.

Every face in the frame is pixelated (coarse mosaic, detected on the full
frame so faces at the crop edge are caught) before cropping: states are judged
from posture and head direction, never identity. A Gaussian blur was too
strong: it hid head orientation, which labellers need. The model's prediction is kept in the manifest for
evaluation but is never sent to the labelling UI.

Layout under data/labels/<lesson>/:
  tasks.json      task manifest (id, seat, track, time, bbox, prediction)
  crops/<id>.jpg  blurred 3-frame strip
  labels.jsonl    append-only: {task_id, label, labeller, ts}

Usage:
  .venv/bin/python -m labeling.tasks --lesson cctv_securus_2mp_audio --n 120
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

LABELS = [
    "attentive", "writing", "hand_raised", "head_down", "turned_away",
    "interacting", "phone", "out_of_seat", "cant_tell", "not_a_student",
]
STRIP_GAP_S = 1.0
CROP_MARGIN = 0.35
STRIP_HEIGHT = 220
DATA_ROOT = Path("data/labels")


def lesson_dir(lesson: str, root: Path = DATA_ROOT) -> Path:
    return root / lesson


def sample_tasks(tokens: pd.DataFrame, n: int, seed: int = 7, min_gap_s: float = 5.0) -> pd.DataFrame:
    """
    Uniform sample of student tokens, at most one per seat per min_gap_s
    window, so the set isn't dominated by near-duplicate moments.
    """
    df = tokens[tokens["seat_id"].notna()].copy()
    df["window"] = (df["timestamp_s"] // min_gap_s).astype(int)
    one_per_window = df.sample(frac=1.0, random_state=seed).drop_duplicates(["seat_id", "window"])
    return one_per_window.sample(min(n, len(one_per_window)), random_state=seed).sort_values("timestamp_s")


def export_tasks(
    lesson: str, video: Path, tokens: pd.DataFrame, n: int, root: Path = DATA_ROOT, seed: int = 7,
    blurrer=None,
) -> list[dict]:
    """Write tasks.json and crops for a lesson; returns the task list."""
    from classroom.identity import FaceBlurrer, pixelate_regions

    out = lesson_dir(lesson, root)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    blurrer = blurrer or FaceBlurrer()
    chosen = sample_tasks(tokens, n, seed)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tasks = []
    for i, row in enumerate(chosen.itertuples(), start=1):
        task_id = f"{lesson[:24]}-{i:04d}"
        bbox = [row.x1, row.y1, row.x2, row.y2]
        frames = []
        for dt in (-STRIP_GAP_S, 0.0, STRIP_GAP_S):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round((row.timestamp_s + dt) * fps))))
            ok, frame = cap.read()
            if not ok:
                break
            pixelate_regions(frame, blurrer.face_boxes(frame, margin=0.15))
            frames.append(_crop(frame, bbox))
        if len(frames) < 3:
            continue
        cv2.imwrite(str(out / "crops" / f"{task_id}.jpg"), np.hstack(frames), [cv2.IMWRITE_JPEG_QUALITY, 88])
        tasks.append({
            "id": task_id,
            "lesson": lesson,
            "seat_id": row.seat_id,
            "track_id": int(row.track_id),
            "t": round(float(row.timestamp_s), 2),
            "bbox": [round(float(v), 1) for v in bbox],
            "box_height_px": round(float(row.y2 - row.y1), 1),
            "predicted": row.state,
        })
    cap.release()
    (out / "tasks.json").write_text(json.dumps({
        "lesson": lesson, "video": str(video), "created": time.strftime("%Y-%m-%d"), "sampling": "uniform",
        "seed": seed, "labels": LABELS, "tasks": tasks,
    }, indent=1))
    return tasks


def _crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
    """Student region with margin (faces already pixelated), resized to STRIP_HEIGHT."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
    cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    cx2, cy2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    crop = frame[cy1:cy2, cx1:cx2].copy()
    scale = STRIP_HEIGHT / crop.shape[0]
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), STRIP_HEIGHT))
    # Mark the target student's box so labellers know whom to judge
    bx1, by1 = int((x1 - cx1) * scale), int((y1 - cy1) * scale)
    bx2, by2 = int((x2 - cx1) * scale), int((y2 - cy1) * scale)
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
    return cv2.copyMakeBorder(crop, 0, 0, 0, 6, cv2.BORDER_CONSTANT, value=(20, 20, 20))


# ----------------------------------------------------------------------
# Label store
# ----------------------------------------------------------------------

def load_tasks(lesson: str, root: Path = DATA_ROOT) -> Optional[dict]:
    path = lesson_dir(lesson, root) / "tasks.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_labels(lesson: str, root: Path = DATA_ROOT) -> pd.DataFrame:
    """All labels, latest per (task, labeller) wins (re-labelling corrects mistakes)."""
    path = lesson_dir(lesson, root) / "labels.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["task_id", "label", "labeller", "ts"])
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    return df.sort_values("ts").groupby(["task_id", "labeller"], as_index=False).last()


def add_label(lesson: str, task_id: str, label: str, labeller: str, root: Path = DATA_ROOT) -> dict:
    if label not in LABELS:
        raise ValueError(f"unknown label '{label}'")
    tasks = load_tasks(lesson, root)
    if tasks is None or task_id not in {t["id"] for t in tasks["tasks"]}:
        raise KeyError(task_id)
    entry = {"task_id": task_id, "label": label, "labeller": labeller, "ts": time.time()}
    with (lesson_dir(lesson, root) / "labels.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def next_task(lesson: str, labeller: str, root: Path = DATA_ROOT) -> Optional[dict]:
    """Next task this labeller hasn't labelled, without the prediction."""
    tasks = load_tasks(lesson, root)
    if tasks is None:
        return None
    labels = load_labels(lesson, root)
    done = set(labels[labels["labeller"] == labeller]["task_id"])
    for i, t in enumerate(tasks["tasks"]):
        if t["id"] not in done:
            return {"id": t["id"], "seat_id": t["seat_id"], "t": t["t"], "index": i, "total": len(tasks["tasks"]),
                    "done": len(done)}
    return {"id": None, "total": len(tasks["tasks"]), "done": len(done)}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lesson", required=True, help="outputs/<lesson> produced by classroom.pipeline")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    lesson_out = args.outputs / args.lesson
    summary = json.loads((lesson_out / "summary.json").read_text())
    tokens = pd.read_parquet(lesson_out / "tokens.parquet")
    tasks = export_tasks(args.lesson, Path(summary["video"]), tokens, args.n, seed=args.seed)
    print(f"{len(tasks)} tasks → {lesson_dir(args.lesson)}")


if __name__ == "__main__":
    main()
