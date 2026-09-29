"""
Head orientation from COCO pose keypoints.

Classroom CCTV gives ~20–30 px faces, too small for crop-based gaze models
(L2CS) and marginal for 6DRepNet. Pose keypoints survive at that scale, so
this is the default estimator. Values are image-space and camera-relative;
StateClassifier compares them against the class consensus and each
student's own baseline instead of absolute thresholds.
"""

from typing import Optional

import numpy as np

from .tokens import HeadPose

KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR = 0, 1, 2, 3, 4
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6

KP_CONF = 0.4
MIN_SHOULDER_WIDTH_PX = 4.0
BACK_TURNED_EAR_LIFT = 0.3  # ears this many shoulder widths above the shoulder line


def _visible(kps: np.ndarray, idx: int) -> bool:
    return bool(kps[idx, 2] >= KP_CONF)


def shoulder_anchor(kps: np.ndarray, bbox: list[float]) -> tuple[float, float]:
    """Stable per-person anchor: shoulder midpoint, else upper-body point of the bbox."""
    if _visible(kps, KP_LEFT_SHOULDER) and _visible(kps, KP_RIGHT_SHOULDER):
        mid = (kps[KP_LEFT_SHOULDER, :2] + kps[KP_RIGHT_SHOULDER, :2]) / 2
        return float(mid[0]), float(mid[1])
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, y1 + 0.25 * (y2 - y1)


def shoulder_width(kps: np.ndarray, bbox: list[float]) -> float:
    """Shoulder width in px, falling back to a bbox-width proxy."""
    if _visible(kps, KP_LEFT_SHOULDER) and _visible(kps, KP_RIGHT_SHOULDER):
        w = float(np.linalg.norm(kps[KP_LEFT_SHOULDER, :2] - kps[KP_RIGHT_SHOULDER, :2]))
        if w >= MIN_SHOULDER_WIDTH_PX:
            return w
    return max(MIN_SHOULDER_WIDTH_PX, 0.6 * (bbox[2] - bbox[0]))


def estimate_head_pose(kps: np.ndarray, bbox: list[float]) -> HeadPose:
    """
    Estimate head yaw / pitch proxies from one person's keypoints.

    Args:
        kps: (17, 3) array of [x, y, confidence]
        bbox: [x1, y1, x2, y2]
    """
    kps = np.asarray(kps, dtype=np.float32)
    scale = shoulder_width(kps, bbox)
    face_visibility = float(np.mean(kps[[KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE], 2]))

    ears = [i for i in (KP_LEFT_EAR, KP_RIGHT_EAR) if _visible(kps, i)]
    eyes = [i for i in (KP_LEFT_EYE, KP_RIGHT_EYE) if _visible(kps, i)]
    nose_ok = _visible(kps, KP_NOSE)

    # Back of the head: no face, but ears sit clearly above the shoulder line.
    # A hidden face with ears at/below the shoulders (or no head points at
    # all) is a head lowered onto the desk, not a turned back.
    shoulders_ok = _visible(kps, KP_LEFT_SHOULDER) and _visible(kps, KP_RIGHT_SHOULDER)
    ear_lift: Optional[float] = None
    if ears and shoulders_ok:
        shoulder_y = (kps[KP_LEFT_SHOULDER, 1] + kps[KP_RIGHT_SHOULDER, 1]) / 2
        ear_lift = float((shoulder_y - np.mean(kps[ears, 1])) / scale)
    back_turned = not nose_ok and not eyes and ear_lift is not None and ear_lift > BACK_TURNED_EAR_LIFT

    yaw: Optional[float] = None
    if nose_ok and (ears or eyes):
        # Ears give the most rotation signal; eyes are the fallback centre
        centre_pts = kps[ears, :2] if ears else kps[eyes, :2]
        centre = centre_pts.mean(axis=0)
        yaw = float((kps[KP_NOSE, 0] - centre[0]) / scale)

    pitch_ratio: Optional[float] = None
    if nose_ok and shoulders_ok:
        shoulder_y = (kps[KP_LEFT_SHOULDER, 1] + kps[KP_RIGHT_SHOULDER, 1]) / 2
        pitch_ratio = float((shoulder_y - kps[KP_NOSE, 1]) / scale)

    used = [KP_NOSE] + ears + eyes + ([KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER] if shoulders_ok else [])
    confidence = float(np.mean(kps[used, 2])) if (yaw is not None or pitch_ratio is not None) else 0.0
    if back_turned:
        confidence = float(np.mean(kps[ears + [KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER], 2]))

    return HeadPose(
        yaw=yaw,
        pitch_ratio=pitch_ratio,
        ear_lift=ear_lift,
        face_visibility=face_visibility,
        back_turned=back_turned,
        confidence=confidence,
    )
