"""
Offline classroom pipeline — recorded video → per-seat state timeline.

Pass 1 (perception): sample frames at --fps, run YOLO pose + BoT-SORT
tracking, and optionally a COCO detector for phones. Detections are kept
in memory.
Calibrate: learn the seat map from the first --calib-seconds (or load
--seatmap).
Pass 2 (reasoning): seat assignment → role → StateClassifier →
TemporalSmoother.

Outputs in --out:
  tokens.parquet            one row per student per sampled frame
  seat_timeline.parquet     per seat per second: majority state
  class_engagement.csv      per second: students seen, engaged share, state shares
  seatmap.json              learned or loaded seat map
  summary.json              run stats and state distribution
  seat_identity.json        seat → consented student (or anonymous), with evidence
  preview.mp4 (--preview)   annotated video: seat ids + state colours; every head
                            not identified as a consented student is blurred
  detections.pkl            pass-1 cache, reused when video + perception settings match

Identity is consent-gated (classroom/consent.py, classroom/identity.py): seats
are linked to students only when the consent registry allows the purpose and
the student is enrolled in the encrypted gallery or listed on the seating
chart. With no consented enrollments, the run stays fully anonymous and no face
is ever embedded.

Usage:
  .venv/bin/python -m classroom.pipeline --video footage/cctv_securus_2mp_audio.mp4 --preview
"""

import argparse
import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from sentinel_engine.device import detect_device

from .consent import AuditLog, ConsentRegistry
from .headpose import shoulder_anchor, shoulder_width
from .identity import (
    FaceBlurrer, Gallery, SeatIdentifier, blur_heads, head_box, load_seating_chart, outside_all, save_identities,
)
from .seatmap import SeatMap, SeatTracker
from .states import ClassifierConfig, PersonObs, StateClassifier, TemporalSmoother
from .tokens import ENGAGED_STATES, ClassroomToken, Role, StudentState

logger = logging.getLogger(__name__)

COCO_CELL_PHONE = 67
PHONE_CONF = 0.45  # COCO phones confuse with calculators below this

STATE_COLOURS = {  # BGR
    StudentState.ATTENTIVE: (80, 200, 80),
    StudentState.WRITING: (200, 160, 40),
    StudentState.HAND_RAISED: (255, 255, 0),
    StudentState.HEAD_DOWN: (0, 140, 255),
    StudentState.TURNED_AWAY: (0, 0, 230),
    StudentState.INTERACTING: (180, 0, 200),
    StudentState.PHONE: (0, 0, 130),
    StudentState.OUT_OF_SEAT: (120, 120, 120),
    StudentState.UNKNOWN: (200, 200, 200),
}


@dataclass
class FrameDetections:
    frame_idx: int
    timestamp_s: float
    width: int
    height: int
    track_ids: list[int] = field(default_factory=list)
    bboxes: list[list[float]] = field(default_factory=list)
    keypoints: list[np.ndarray] = field(default_factory=list)
    phones: list[list[float]] = field(default_factory=list)


# ----------------------------------------------------------------------
# Pass 1 — perception
# ----------------------------------------------------------------------

def run_perception(
    video: Path,
    pose_model_path: str,
    phone_model_path: Optional[str],
    fps: float,
    imgsz: int,
    max_seconds: Optional[float],
    device: str,
) -> list[FrameDetections]:
    from ultralytics import YOLO

    pose_model = YOLO(pose_model_path)
    phone_model = YOLO(phone_model_path) if phone_model_path else None

    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / fps))
    last = total if max_seconds is None else min(total, int(max_seconds * src_fps))

    results: list[FrameDetections] = []
    started = time.time()
    for frame_idx in range(0, last, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        det = FrameDetections(frame_idx, frame_idx / src_fps, w, h)

        res = pose_model.track(
            frame, persist=True, tracker="botsort.yaml", imgsz=imgsz, conf=0.2, device=device, verbose=False,
        )[0]
        if res.boxes is not None and res.boxes.id is not None:
            det.track_ids = res.boxes.id.int().cpu().tolist()
            det.bboxes = res.boxes.xyxy.cpu().numpy().tolist()
            kp_xy = res.keypoints.xy.cpu().numpy()
            kp_conf = res.keypoints.conf.cpu().numpy()
            det.keypoints = [np.concatenate([kp_xy[i], kp_conf[i][:, None]], axis=1) for i in range(len(kp_xy))]

        if phone_model is not None:
            pres = phone_model.predict(
                frame, classes=[COCO_CELL_PHONE], imgsz=imgsz, conf=PHONE_CONF, device=device, verbose=False,
            )[0]
            if pres.boxes is not None and len(pres.boxes):
                det.phones = pres.boxes.xyxy.cpu().numpy().tolist()

        results.append(det)
        if len(results) % 50 == 0:
            rate = len(results) / (time.time() - started)
            logger.info(f"  perception {det.timestamp_s:7.1f}s  ({rate:.1f} frames/s)")
    cap.release()
    return results


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

def calibrate_seatmap(frames: list[FrameDetections], calib_seconds: float, camera_id: str) -> SeatMap:
    anchors_by_frame = []
    for det in frames:
        if det.timestamp_s > frames[0].timestamp_s + calib_seconds:
            break
        rows = [
            [x / det.width, y / det.height, shoulder_width(k, b) / det.width]
            for k, b in zip(det.keypoints, det.bboxes)
            for x, y in [shoulder_anchor(k, b)]
        ]
        anchors_by_frame.append(np.array(rows, dtype=np.float32).reshape(-1, 3))
    return SeatMap.calibrate(anchors_by_frame, camera_id=camera_id)


# ----------------------------------------------------------------------
# Pass 2 — reasoning
# ----------------------------------------------------------------------

def phone_in_hands(bbox: list[float], kps: np.ndarray, phones: list[list[float]]) -> bool:
    """
    A phone box close to one of the student's wrists.

    Wrist proximity, not just "inside the person box": calculators and
    phones lying on desks (common in exams) sit inside the box too.
    """
    wrists = [kps[i, :2] for i in (9, 10) if kps[i, 2] >= 0.4]
    if not phones or not wrists:
        return False
    reach = 0.35 * (bbox[2] - bbox[0])
    for px1, py1, px2, py2 in phones:
        cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
        if any(np.hypot(cx - wx, cy - wy) <= reach for wx, wy in wrists):
            return True
    return False


def run_reasoning(
    frames: list[FrameDetections],
    seatmap: SeatMap,
    camera_id: str,
    smooth_frames: int,
) -> list[ClassroomToken]:
    seat_tracker = SeatTracker()
    classifier = StateClassifier(seatmap, ClassifierConfig())
    smoother = TemporalSmoother(window=smooth_frames)
    tokens: list[ClassroomToken] = []

    for det in frames:
        anchors = {
            tid: (x / det.width, y / det.height)
            for tid, (x, y) in zip(det.track_ids, (shoulder_anchor(k, b) for k, b in zip(det.keypoints, det.bboxes)))
        }
        seats = seat_tracker.update(seatmap.assign(anchors))

        people = [
            PersonObs(
                track_id=tid,
                bbox=bbox,
                keypoints=kps,
                seat_id=seats[tid],
                is_student=seat_tracker.is_student(tid),
                seated_now=seat_tracker.currently_seated(tid),
                has_phone=phone_in_hands(bbox, kps, det.phones),
            )
            for tid, bbox, kps in zip(det.track_ids, det.bboxes, det.keypoints)
        ]
        results = classifier.classify_frame(people)

        for p in people:
            if p.track_id not in results:
                continue
            c = results[p.track_id]
            state, share = smoother.update(p.track_id, c.state)
            tokens.append(ClassroomToken(
                frame_idx=det.frame_idx,
                timestamp_s=det.timestamp_s,
                camera_id=camera_id,
                track_id=p.track_id,
                seat_id=p.seat_id,
                role=Role.STUDENT,
                bbox=p.bbox,
                head=c.head,
                state=state,
                state_confidence=share * c.confidence if state == c.state else share * 0.5,
                phone=p.has_phone,
                cues={**c.cues, f"raw_{c.state.value}": True},
            ))

        active = set(det.track_ids)
        classifier.forget(active)
        smoother.forget(active)
    return tokens


# ----------------------------------------------------------------------
# Aggregation & outputs
# ----------------------------------------------------------------------

def aggregate(tokens_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-seat-per-second timeline and per-second class engagement."""
    df = tokens_df.copy()
    df["second"] = df["timestamp_s"].astype(int)
    df["seat_key"] = df["seat_id"].fillna("track_" + df["track_id"].astype(str))

    seat_timeline = (
        df.groupby(["seat_key", "second"])["state"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
        .rename(columns={"seat_key": "seat_id"})
    )

    known = seat_timeline[seat_timeline["state"] != StudentState.UNKNOWN.value]
    engaged_values = {s.value for s in ENGAGED_STATES}
    per_second = known.groupby("second").agg(
        students=("seat_id", "nunique"),
        engaged_share=("state", lambda s: float(s.isin(engaged_values).mean())),
    )
    shares = pd.crosstab(known["second"], known["state"], normalize="index").add_prefix("share_")
    class_engagement = per_second.join(shares).reset_index()
    return seat_timeline, class_engagement


def run_identification(
    video: Path, frames: list[FrameDetections], tokens: list[ClassroomToken], identifier: SeatIdentifier,
    every_n: int,
) -> None:
    """Feed each seated student's head region to the identifier, every `every_n` sampled frames."""
    seat_of = {(t.frame_idx, t.track_id): t.seat_id for t in tokens if t.seat_id}
    cap = cv2.VideoCapture(str(video))
    for det in frames[::every_n]:
        targets = [
            (seat_of[(det.frame_idx, tid)], head_box(kps, bbox))
            for tid, bbox, kps in zip(det.track_ids, det.bboxes, det.keypoints)
            if (det.frame_idx, tid) in seat_of
        ]
        if not targets:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, det.frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        for seat_id, head in targets:
            identifier.observe(frame, seat_id, head)
    cap.release()


def write_preview(
    video: Path, out_path: Path, frames: list[FrameDetections], tokens: list[ClassroomToken], fps: float,
    seat_students: dict[str, str],
) -> None:
    """
    Annotated video. Every head not identified as a consented student is blurred:
    tracked people via their pose head box, plus any face the whole-frame face
    detector finds (catches people the person detector missed).
    """
    blurrer = FaceBlurrer()
    by_frame: dict[int, list[ClassroomToken]] = {}
    for t in tokens:
        by_frame.setdefault(t.frame_idx, []).append(t)

    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].width, frames[0].height))
    for det in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, det.frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        identified = {t.track_id for t in by_frame.get(det.frame_idx, []) if seat_students.get(t.seat_id)}
        people = list(zip(det.track_ids, det.bboxes, det.keypoints))
        keep = [head_box(kps, bbox) for tid, bbox, kps in people if tid in identified]
        blur_heads(frame, [head_box(kps, bbox) for tid, bbox, kps in people if tid not in identified])
        blur_heads(frame, [b for b in blurrer.face_boxes(frame) if outside_all(b, keep)])
        student_ids = set()
        for t in by_frame.get(det.frame_idx, []):
            student_ids.add(t.track_id)
            colour = STATE_COLOURS[t.state]
            x1, y1, x2, y2 = map(int, t.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            label = f"{seat_students.get(t.seat_id) or t.seat_id or '-'} {t.state.value}"
            cv2.putText(frame, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
        for tid, bbox in zip(det.track_ids, det.bboxes):
            if tid not in student_ids:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(frame, "other", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        for x1, y1, x2, y2 in det.phones:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        engaged = [t for t in by_frame.get(det.frame_idx, []) if t.state != StudentState.UNKNOWN]
        share = np.mean([t.state in ENGAGED_STATES for t in engaged]) if engaged else 0.0
        cv2.putText(frame, f"t={det.timestamp_s:6.1f}s  students={len(engaged)}  engaged={share:.0%}",
                    (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    cap.release()


def identify_seats(args, frames: list[FrameDetections], tokens: list[ClassroomToken], seatmap: SeatMap) -> dict[str, str]:
    """
    Consent-gated seat → student mapping. Returns only identified seats.

    Face matching runs only when the gallery holds at least one student whose
    consent allows this purpose; otherwise no face is detected or embedded.
    """
    out = args.out or Path("outputs") / args.video.stem
    registry = ConsentRegistry(args.consent)
    allowed = registry.allowed_students(args.purpose)
    chart = load_seating_chart(args.seating_chart, registry)

    gallery = Gallery(dim=128)
    if allowed and args.gallery.exists():
        gallery = Gallery.load(args.gallery)
        gallery.restrict_to(allowed)

    seat_ids = [s.seat_id for s in seatmap.seats]
    if len(gallery):
        from .faces import OpenCVFaceEmbedder
        identifier = SeatIdentifier(gallery, OpenCVFaceEmbedder())
        run_identification(args.video, frames, tokens, identifier, every_n=max(1, round(args.fps)))
    else:
        identifier = SeatIdentifier(gallery, embedder=None)   # chart-only / anonymous
    identities = identifier.resolve(seat_ids, chart, allowed)
    save_identities(out / "seat_identity.json", identities)

    seat_students = {s: i.student_id for s, i in identities.items() if i.student_id}
    AuditLog(args.audit).record(
        "identify_run", video=str(args.video), purpose=args.purpose, seats=len(seat_ids),
        identified=len(seat_students), conflicts=sum(i.source == "conflict" for i in identities.values()),
        gallery_students=len(gallery),
    )
    logger.info(f"Identity: {len(seat_students)}/{len(seat_ids)} seats linked to consented students "
                f"(gallery {len(gallery)}, chart {len(chart)}, purpose {args.purpose})")
    return seat_students


def load_cached_detections(cache: Path, key: dict) -> Optional[list[FrameDetections]]:
    """Pass-1 results from a previous run with identical perception settings (local, self-written file)."""
    if not cache.exists():
        return None
    with cache.open("rb") as f:
        data = pickle.load(f)
    return data["frames"] if data.get("key") == key else None


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=3.0, help="Sampling rate for analysis")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--pose-model", default="models/yolo11m-pose.pt")
    parser.add_argument("--phone-model", default="models/yolo11s.pt", help="'' to disable phone detection")
    parser.add_argument("--seatmap", type=Path, default=None, help="Load a saved seat map instead of calibrating")
    parser.add_argument("--calib-seconds", type=float, default=300.0, help="Seat-map learning window")
    parser.add_argument("--smooth-seconds", type=float, default=3.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--consent", type=Path, default=Path("data/consent/registry.csv"))
    parser.add_argument("--gallery", type=Path, default=Path("data/faces/gallery.enc"))
    parser.add_argument("--seating-chart", type=Path, default=None, help="CSV: seat_id,student_id")
    parser.add_argument("--purpose", choices=["engagement", "exam"], default="engagement")
    parser.add_argument("--audit", type=Path, default=Path("data/audit/audit.jsonl"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    camera_id = args.camera_id or args.video.stem
    out = args.out or Path("outputs") / args.video.stem
    out.mkdir(parents=True, exist_ok=True)
    device = detect_device()

    perception_key = {
        "video": str(args.video.resolve()), "size": args.video.stat().st_size, "pose_model": args.pose_model,
        "phone_model": args.phone_model, "fps": args.fps, "imgsz": args.imgsz, "max_seconds": args.max_seconds,
    }
    cache = out / "detections.pkl"
    t0 = time.time()
    frames = load_cached_detections(cache, perception_key)
    if frames is None:
        logger.info(f"Perception on {args.video.name} @ {args.fps} fps, {args.pose_model}, device={device}")
        frames = run_perception(
            args.video, args.pose_model, args.phone_model or None, args.fps, args.imgsz, args.max_seconds, device,
        )
        with cache.open("wb") as f:
            pickle.dump({"key": perception_key, "frames": frames}, f)
    else:
        logger.info(f"Reusing cached detections from {cache}")
    perception_s = time.time() - t0
    if not frames:
        sys.exit("No frames read")

    if args.seatmap:
        seatmap = SeatMap.from_json(args.seatmap)
    else:
        seatmap = calibrate_seatmap(frames, args.calib_seconds, camera_id)
    seatmap.to_json(out / "seatmap.json")
    logger.info(f"Seat map: {len(seatmap)} seats")

    tokens = run_reasoning(frames, seatmap, camera_id, smooth_frames=max(1, round(args.smooth_seconds * args.fps)))
    tokens_df = pd.DataFrame([t.to_dict() for t in tokens])
    if tokens_df.empty:
        sys.exit("No student tokens produced — check the seat map / detections")

    seat_students = identify_seats(args, frames, tokens, seatmap)
    tokens_df["student_id"] = tokens_df["seat_id"].map(seat_students)
    tokens_df.to_parquet(out / "tokens.parquet", index=False)

    seat_timeline, class_engagement = aggregate(tokens_df)
    seat_timeline["student_id"] = seat_timeline["seat_id"].map(seat_students)
    seat_timeline.to_parquet(out / "seat_timeline.parquet", index=False)
    class_engagement.to_csv(out / "class_engagement.csv", index=False)

    people_per_frame = [len(f.track_ids) for f in frames]
    summary = {
        "video": str(args.video),
        "camera_id": camera_id,
        "sampled_frames": len(frames),
        "analysis_fps": args.fps,
        "duration_analysed_s": round(frames[-1].timestamp_s - frames[0].timestamp_s, 1),
        "perception_seconds": round(perception_s, 1),
        "seats": len(seatmap),
        "people_per_frame_median": float(np.median(people_per_frame)),
        "unique_tracks": len({t for f in frames for t in f.track_ids}),
        "student_tracks": int(tokens_df["track_id"].nunique()),
        "state_distribution": tokens_df["state"].value_counts(normalize=True).round(3).to_dict(),
        "mean_engaged_share": round(float(class_engagement["engaged_share"].mean()), 3),
        "phone_frames": int(sum(bool(f.phones) for f in frames)),
        "identified_seats": len(seat_students),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(json.dumps(summary, indent=2))

    if args.preview:
        write_preview(args.video, out / "preview.mp4", frames, tokens, args.fps, seat_students)
        logger.info(f"Preview: {out / 'preview.mp4'}")


if __name__ == "__main__":
    main()
