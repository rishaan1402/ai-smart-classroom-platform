"""
VLM verification of exam flags — a second opinion before the review queue.

Rule-based exam flags are cheap but noisy (see docs/exam_mode_status.md). For
each flag, a local vision-language model (Ollama, default qwen2.5vl:7b) sees
one image: three moments across the flag (start, middle, end), cropped
around the flagged student (yellow box) and the neighbour's seat (orange
ring), the same marks the invigilator's clip uses.

The question is deliberately neutral and the same for every flag type: the
model first describes the student in its own words, then picks one activity
from a fixed list with the ordinary exam activities first, and the verdict
is derived here from that choice. A first version asked a type-specific
yes/no question ("is the student looking at the neighbour's paper?"), and
qwen2.5vl:7b said yes to 16 of 21 flags on a calm exam, echoing the question
back as its description (acquiescence bias).

Even with the neutral prompt, the chosen activity sometimes contradicts the
model's own description ("looking down at their table" labelled
looking_towards_neighbour). The description, written before the label, is
more grounded, so a text LLM (check_client, default qwen3:8b) re-reads the
description of every flag the VLM labelled suspicious and picks the activity
the description supports; the verdict follows that reading. The check only
ever re-examines suspicious labels: run on every flag, it answered
"cannot tell" even for clear "holding a pen, writing" descriptions and
cleared nothing. Both labels are kept, and disagreements are marked.

The VLM never removes a flag. "normal" lowers its priority so it drops out
of the default queue, but invigilators can still see and review it, and the
VLM's description is shown next to the clip. Crops are built in memory and
sent only to the local model; nothing is written to disk or leaves the machine.
"""

import base64
import logging
import time
from typing import Optional

import cv2
import numpy as np

from topics.llm import OllamaClient

logger = logging.getLogger(__name__)

DEFAULT_VLM = "qwen2.5vl:7b"
CROP_SIDE = 448
CONTEXT_MARGIN = 0.6
VERDICTS = ("suspicious", "normal", "unclear")
PRIORITY_FACTOR = {"suspicious": 1.0, "unclear": 0.8, "normal": 0.25}

SYSTEM = """\
You help an exam invigilator in an Indian school or college. You see ONE image made of three \
panels, left to right, showing the same place about a second apart. Look at the person inside the \
YELLOW box. An ORANGE ring, if present, marks a neighbouring seat. Describe only what is visible; \
most of the time students are simply doing their own work. Return JSON only."""

QUESTION = (
    "First describe, in your own words, what the person in the yellow box is doing in the MIDDLE panel "
    "(where they look, what their hands do, what they hold). Then choose the one activity that fits best."
)

ACTIVITIES = [
    "writing_on_own_paper", "reading_own_paper", "thinking_or_looking_up", "using_calculator",
    "resting_or_stretching", "handling_own_items",
    "looking_towards_neighbour", "talking_with_neighbour", "passing_or_receiving_object",
    "using_phone", "turned_around_to_back", "standing_or_out_of_seat",
    "cannot_tell",
]
SUSPICIOUS_ACTIVITIES = {
    "looking_towards_neighbour", "talking_with_neighbour", "passing_or_receiving_object",
    "using_phone", "turned_around_to_back", "standing_or_out_of_seat",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "activity": {"type": "string", "enum": ACTIVITIES},
    },
    "required": ["description", "activity"],
}


CHECK_SYSTEM = """\
You read an observer's description of one student during an exam and choose the activity the \
description supports. Choose a suspicious activity (looking towards a neighbour, talking, passing \
an object, using a phone, turned around, standing) only if the description says so, for example \
the student looks at another person, at someone else's paper or object, or to the side. If it says \
the student looks down at their own desk, paper, book, hands or lap, choose the matching ordinary \
activity (writing_on_own_paper, reading_own_paper or handling_own_items). Choose cannot_tell only if \
the description gives no usable information. Return JSON only."""

CHECK_SCHEMA = {
    "type": "object",
    "properties": {"activity": {"type": "string", "enum": ACTIVITIES}},
    "required": ["activity"],
}


def verdict_for(activity: str) -> str:
    if activity in SUSPICIOUS_ACTIVITIES:
        return "suspicious"
    if activity == "cannot_tell" or activity not in ACTIVITIES:
        return "unclear"
    return "normal"


def flag_strip(video, flag, frames, seatmap) -> Optional[np.ndarray]:
    """Three marked crops (start, middle, end of the flag) side by side, in memory."""
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    times = np.array([f.timestamp_s for f in frames])
    panels = []
    for t in (flag.start_s, (flag.start_s + flag.end_s) / 2, flag.end_s):
        det = frames[int(np.argmin(np.abs(times - t)))]
        if flag.track_id not in det.track_ids:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * src_fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = det.bboxes[det.track_ids.index(flag.track_id)]
        region = [x1, y1, x2, y2]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
        seat = next((s for s in seatmap.seats if s.seat_id == flag.neighbour_seat), None) if flag.neighbour_seat else None
        if seat is not None:
            sx, sy, sr = seat.x * det.width, seat.y * det.height, max(12, seat.radius * det.width * 1.5)
            cv2.circle(frame, (int(sx), int(sy)), int(sr), (0, 140, 255), 3)
            region = [min(region[0], sx - sr), min(region[1], sy - sr), max(region[2], sx + sr), max(region[3], sy + sr)]
        panels.append(_square_crop(frame, region))
    cap.release()
    if not panels:
        return None
    return np.hstack(panels)


def _square_crop(frame: np.ndarray, region: list[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region
    side = max(x2 - x1, y2 - y1) * (1 + CONTEXT_MARGIN)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = frame[b:d, a:c]
    scale = CROP_SIDE / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    return cv2.copyMakeBorder(crop, 0, CROP_SIDE - crop.shape[0], 0, CROP_SIDE - crop.shape[1] + 8,
                              cv2.BORDER_CONSTANT, value=(30, 30, 30))


def verify_flags(video, flags: list, frames, seatmap, client: Optional[OllamaClient] = None,
                 limit: Optional[int] = None, check_client: Optional[OllamaClient] = None,
                 reuse_vlm: bool = False) -> dict:
    """
    Attach flag.vlm and flag.priority; returns stats. Flags beyond `limit` keep their rule score.

    VLM calls run first for every flag, then the description check (if
    check_client is given) for all of them, so Ollama doesn't swap models
    between each call. reuse_vlm keeps existing VLM descriptions/labels and
    only re-runs the description check.
    """
    client = client or OllamaClient(DEFAULT_VLM, think=None, timeout_s=180)
    stats = {v: 0 for v in VERDICTS} | {"errors": 0, "seconds": 0.0, "label_description_disagree": 0}
    for i, flag in enumerate(flags):
        flag.priority = flag.score
        if limit is not None and i >= limit:
            continue
        if reuse_vlm and flag.vlm:
            flag.vlm = {k: v for k, v in flag.vlm.items() if k not in ("description_activity", "disagree", "check_model")}
            flag.vlm["activity"] = flag.vlm["vlm_activity"]
            flag.vlm["verdict"] = verdict_for(flag.vlm["vlm_activity"])
            continue
        strip = flag_strip(video, flag, frames, seatmap)
        if strip is None:
            stats["errors"] += 1
            continue
        ok, jpg = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 90])
        started = time.time()
        try:
            raw = client.chat_json(SYSTEM, QUESTION, SCHEMA, temperature=0.0,
                                   images=[base64.b64encode(jpg.tobytes()).decode()])
        except Exception as e:
            logger.warning(f"VLM failed on {flag.id}: {e}")
            stats["errors"] += 1
            continue
        elapsed = time.time() - started
        stats["seconds"] += elapsed
        activity = raw.get("activity", "cannot_tell")
        flag.vlm = {
            "verdict": verdict_for(activity),
            "activity": activity,
            "vlm_activity": activity,
            "description": str(raw.get("description", ""))[:400],
            "model": client.model,
            "seconds": round(elapsed, 1),
        }
        logger.info(f"  VLM {flag.id} {flag.type}: [{activity}] {flag.vlm['description'][:120]} ({elapsed:.0f}s)")

    if check_client is not None:
        for flag in flags:
            if not flag.vlm or not flag.vlm["description"] or verdict_for(flag.vlm["vlm_activity"]) != "suspicious":
                continue
            try:
                checked = check_client.chat_json(CHECK_SYSTEM, flag.vlm["description"], CHECK_SCHEMA, temperature=0.0)
            except Exception as e:
                logger.warning(f"Description check failed on {flag.id}: {e}")
                continue
            text_activity = checked.get("activity", "cannot_tell")
            flag.vlm["description_activity"] = text_activity
            flag.vlm["disagree"] = verdict_for(text_activity) != verdict_for(flag.vlm["vlm_activity"])
            stats["label_description_disagree"] += flag.vlm["disagree"]
            flag.vlm["activity"] = text_activity
            flag.vlm["verdict"] = verdict_for(text_activity)
            flag.vlm["check_model"] = check_client.model

    for flag in flags:
        if flag.vlm:
            stats[flag.vlm["verdict"]] += 1
            flag.priority = round(flag.score * PRIORITY_FACTOR[flag.vlm["verdict"]], 3)
    stats["seconds"] = round(stats["seconds"], 1)
    return stats


def main(argv: Optional[list[str]] = None) -> None:
    """Re-run verification on an existing exam run (outputs/<lesson>) without redoing perception or clips."""
    import argparse
    import json
    import pickle
    from pathlib import Path

    from .exam import ExamFlag
    from .seatmap import SeatMap

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--lesson", type=Path, required=True, help="outputs/<lesson> from --mode exam")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_VLM)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--check-model", default="qwen3:8b", help="Text LLM that re-reads descriptions ('' to skip)")
    parser.add_argument("--check-only", action="store_true", help="Keep stored VLM outputs; re-run only the check")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    doc = json.loads((args.lesson / "exam_flags.json").read_text())
    flags = [ExamFlag(**{k: v for k, v in f.items() if k in ExamFlag.__dataclass_fields__}) for f in doc["flags"]]
    if not args.check_only:
        for f in flags:
            f.vlm = None
    frames = pickle.load(open(args.lesson / "detections.pkl", "rb"))["frames"]
    seatmap = SeatMap.from_json(args.lesson / "seatmap.json")
    flags.sort(key=lambda f: (-f.score, f.start_s))
    check = OllamaClient(args.check_model, think=False, timeout_s=120) if args.check_model else None
    stats = verify_flags(args.video, flags, frames, seatmap, OllamaClient(args.model, think=None, timeout_s=180),
                         limit=args.limit, check_client=check, reuse_vlm=args.check_only)
    flags.sort(key=lambda f: (-(f.priority or 0), f.start_s))

    doc["flags"] = [f.to_dict() for f in flags]
    (args.lesson / "exam_flags.json").write_text(json.dumps(doc, indent=1))
    summary_path = args.lesson / "exam_summary.json"
    summary = json.loads(summary_path.read_text())
    queue = [f for f in flags if not (f.vlm and f.vlm["verdict"] == "normal")]
    hours = summary.get("student_hours") or 0
    summary.update(vlm={**stats, "model": args.model}, queue_after_vlm=len(queue),
                   queue_per_student_hour=round(len(queue) / hours, 2) if hours else None)
    summary_path.write_text(json.dumps(summary, indent=2))
    logging.info(json.dumps(summary))


if __name__ == "__main__":
    from classroom.vlm_verify import main as _main
    _main()
