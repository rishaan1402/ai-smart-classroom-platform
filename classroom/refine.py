"""
Crop-level pose refinement for small students.

Classroom CCTV students are 60–150 px tall, where full-frame pose loses
arms (raised-arm wrist confidence ~0.2) and misplaces joints. Re-running the
same pose model on each student's crop, upscaled to CROP_HEIGHT, gives it a
detailed single-person image. Keypoints are merged per joint: the refined
estimate replaces the original only where it is more confident, so a good
full-frame keypoint is never overwritten by a worse one.

Gates (which students get refined, to bound cost):
  arms   any elbow or wrist below ARM_CONF; targets lost / uncertain arms
  small  person box shorter than SMALL_PX
  all    every detected person
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CROP_HEIGHT = 320
MARGIN_X, MARGIN_Y = 0.3, 0.25
ARM_KPS = [7, 8, 9, 10]
ARM_CONF = 0.5
SMALL_PX = 150
GATES = ("arms", "small", "all")


def needs_refinement(kps: np.ndarray, bbox: list[float], gate: str) -> bool:
    if gate == "all":
        return True
    if gate == "small":
        return (bbox[3] - bbox[1]) < SMALL_PX
    if gate == "arms":
        return bool((np.asarray(kps)[ARM_KPS, 2] < ARM_CONF).any())
    raise ValueError(f"unknown gate '{gate}'")


def crop_box(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    mx, my = MARGIN_X * (x2 - x1), MARGIN_Y * (y2 - y1)
    return (int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(width, x2 + mx)), int(min(height, y2 + my)))


def pick_target(boxes: np.ndarray, expected: tuple[float, float, float, float]) -> Optional[int]:
    """Index of the detection that best overlaps the expected (original) box, or None if none overlaps well."""
    if not len(boxes):
        return None
    ex1, ey1, ex2, ey2 = expected
    ix1 = np.maximum(boxes[:, 0], ex1)
    iy1 = np.maximum(boxes[:, 1], ey1)
    ix2 = np.minimum(boxes[:, 2], ex2)
    iy2 = np.minimum(boxes[:, 3], ey2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + (ex2 - ex1) * (ey2 - ey1) - inter
    iou = inter / np.maximum(union, 1e-6)
    best = int(np.argmax(iou))
    return best if iou[best] >= 0.3 else None


def merge_keypoints(original: np.ndarray, refined: np.ndarray) -> np.ndarray:
    """Per joint, keep whichever estimate is more confident."""
    original = np.asarray(original, dtype=np.float32)
    refined = np.asarray(refined, dtype=np.float32)
    take = refined[:, 2] > original[:, 2]
    out = original.copy()
    out[take] = refined[take]
    return out


def refine_frames(video, frames: list, model, gate: str = "arms", device: str = "cpu", batch: int = 16) -> dict:
    """
    Refine keypoints in-place on `frames` (FrameDetections). Returns stats.

    Reads each frame that has at least one person passing the gate, crops
    and upscales those people, and runs pose in batches.
    """
    cap = cv2.VideoCapture(str(video))
    stats = {"frames": 0, "people": 0, "refined": 0, "joints_replaced": 0}
    for det in frames:
        targets = [i for i, (k, b) in enumerate(zip(det.keypoints, det.bboxes)) if needs_refinement(k, b, gate)]
        if not targets:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, det.frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        stats["frames"] += 1
        h, w = frame.shape[:2]

        crops, meta = [], []
        for i in targets:
            cx1, cy1, cx2, cy2 = crop_box(det.bboxes[i], w, h)
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            scale = CROP_HEIGHT / (cy2 - cy1)
            crops.append(cv2.resize(frame[cy1:cy2, cx1:cx2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
            meta.append((i, cx1, cy1, scale))

        for start in range(0, len(crops), batch):
            results = model.predict(crops[start:start + batch], imgsz=CROP_HEIGHT, conf=0.25, device=device, verbose=False)
            for res, (i, cx1, cy1, scale) in zip(results, meta[start:start + batch]):
                stats["people"] += 1
                if res.keypoints is None or res.boxes is None or not len(res.boxes):
                    continue
                x1, y1, x2, y2 = det.bboxes[i]
                expected = ((x1 - cx1) * scale, (y1 - cy1) * scale, (x2 - cx1) * scale, (y2 - cy1) * scale)
                j = pick_target(res.boxes.xyxy.cpu().numpy(), expected)
                if j is None:
                    continue
                kp = np.concatenate([res.keypoints.xy[j].cpu().numpy(), res.keypoints.conf[j].cpu().numpy()[:, None]], 1)
                kp[:, 0] = kp[:, 0] / scale + cx1
                kp[:, 1] = kp[:, 1] / scale + cy1
                merged = merge_keypoints(det.keypoints[i], kp)
                stats["joints_replaced"] += int((merged != np.asarray(det.keypoints[i])).any(axis=1).sum())
                det.keypoints[i] = merged
                stats["refined"] += 1
        if stats["frames"] % 100 == 0:
            logger.info(f"  refine {det.timestamp_s:7.1f}s  ({stats['refined']} people refined)")
    cap.release()
    return stats
