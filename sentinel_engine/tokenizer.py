"""
Tokenizer — converts raw YOLO detections into BehaviorTokens.

Handles:
- Posture inference from keypoint geometry
- Velocity classification from pixel displacement history
- Hands-raised detection from wrist/shoulder geometry
- Zone assignment via point-in-polygon
- Zone recalculation whenever frame dimensions change (bug fix from existing code)
- trace_id generation per new detection chain
"""

import logging
import time
from collections import deque
from typing import Optional
from uuid import uuid4

import numpy as np

from .tokens import BehaviorToken, HandsState, Posture, Velocity

logger = logging.getLogger(__name__)

# COCO keypoint indices
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

KEYPOINT_CONF_THRESHOLD = 0.3
VELOCITY_WALKING_PX = 3.0
VELOCITY_RUNNING_PX = 12.0


class _PersonState:
    """Per-person tracking state for the tokenizer."""
    __slots__ = ("bbox_history", "trace_id", "last_seen")

    def __init__(self) -> None:
        self.bbox_history: deque[list[float]] = deque(maxlen=10)
        self.trace_id = uuid4()
        self.last_seen: float = time.time()


class Tokenizer:
    """
    Converts YOLO pose detections into BehaviorTokens.

    One Tokenizer instance per camera. Maintains per-person
    state (velocity history, trace IDs) across frames.
    """

    def __init__(self, camera_id: str, tenant_id: str) -> None:
        self.camera_id = camera_id
        self.tenant_id = tenant_id

        # Zone definitions: list of {"id", "name", "polygon": [[x,y], ...]}
        # Stored in normalized (0-1) coordinates, scaled on set_zones()
        self._zones: list[dict] = []
        self._zone_polygons: list[np.ndarray] = []  # pixel-space polygons
        self._last_frame_w: int = 0
        self._last_frame_h: int = 0

        # Per-person state keyed by tracker ID
        self._person_states: dict[int, _PersonState] = {}

    def set_zones(
        self,
        zones: list[dict],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """
        Set or recalculate zone polygons.

        Called whenever zones config changes OR frame dimensions change.
        Bug fix: existing code only recalculated when _zones was empty —
        this version always recalculates when dimensions change.

        Args:
            zones: List of zone dicts with "id", "name", and "polygon"
                   (list of [x, y] in normalized 0-1 coordinates)
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels
        """
        if (
            self._zones == zones
            and self._last_frame_w == frame_w
            and self._last_frame_h == frame_h
        ):
            return  # Nothing changed

        self._zones = zones
        self._last_frame_w = frame_w
        self._last_frame_h = frame_h
        self._zone_polygons = []

        for zone in zones:
            pts = zone.get("polygon", [])
            if len(pts) < 3:
                logger.warning(f"Zone {zone.get('id')} has < 3 points, skipping")
                self._zone_polygons.append(np.empty((0, 2)))
                continue
            # Scale normalized coords to pixel space
            pixel_pts = np.array(
                [[p[0] * frame_w, p[1] * frame_h] for p in pts],
                dtype=np.float32,
            )
            self._zone_polygons.append(pixel_pts)

        logger.debug(f"Set {len(zones)} zones for camera {self.camera_id} ({frame_w}×{frame_h})")

    def tokenize(
        self,
        person_id: int,
        bbox: list[float],
        keypoints: list[list[float]],
    ) -> BehaviorToken:
        """
        Convert a single detection into a BehaviorToken.

        Args:
            person_id: Tracker-assigned ID
            bbox: [x1, y1, x2, y2] in pixels
            keypoints: 17×3 list of [x, y, confidence]

        Returns:
            BehaviorToken for this detection
        """
        kps = np.array(keypoints, dtype=np.float32)  # (17, 3)

        # Get or create per-person state
        if person_id not in self._person_states:
            self._person_states[person_id] = _PersonState()
        state = self._person_states[person_id]
        state.last_seen = time.time()

        # Update bbox history for velocity
        state.bbox_history.append(bbox)

        posture = self._infer_posture(kps, bbox)
        velocity_value = self._compute_velocity(state.bbox_history)
        velocity = self._classify_velocity(velocity_value)
        hands_state = self._infer_hands_state(kps)
        zone_id, zone_name = self._get_zone(bbox)

        return BehaviorToken(
            person_id=person_id,
            camera_id=self.camera_id,
            tenant_id=self.tenant_id,
            bbox=bbox,
            keypoints=keypoints,
            posture=posture,
            velocity=velocity,
            hands_state=hands_state,
            zone_id=zone_id,
            zone_name=zone_name,
            velocity_value=velocity_value,
            trace_id=state.trace_id,
        )

    def cleanup(self, active_ids: set[int]) -> None:
        """Remove state for persons no longer being tracked."""
        stale = set(self._person_states.keys()) - active_ids
        for pid in stale:
            del self._person_states[pid]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_posture(self, kps: np.ndarray, bbox: list[float]) -> Posture:
        """Infer posture from keypoint geometry.

        Aspect ratio is NOT used — a seated or close-up person can have a wide bbox
        without being fallen. Rely entirely on keypoint vertical relationships.
        """
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        if h <= 0:
            return Posture.UNKNOWN

        # Hip vs ankle height: if hips near bottom of frame, person is crouching/fallen
        left_hip = kps[KP_LEFT_HIP]
        right_hip = kps[KP_RIGHT_HIP]
        left_ankle = kps[KP_LEFT_ANKLE]
        right_ankle = kps[KP_RIGHT_ANKLE]

        hips_visible = (
            left_hip[2] > KEYPOINT_CONF_THRESHOLD
            or right_hip[2] > KEYPOINT_CONF_THRESHOLD
        )
        ankles_visible = (
            left_ankle[2] > KEYPOINT_CONF_THRESHOLD
            or right_ankle[2] > KEYPOINT_CONF_THRESHOLD
        )

        if hips_visible and ankles_visible:
            hip_y = (
                left_hip[1] if left_hip[2] > KEYPOINT_CONF_THRESHOLD else right_hip[1]
            )
            ankle_y = (
                left_ankle[1]
                if left_ankle[2] > KEYPOINT_CONF_THRESHOLD
                else right_ankle[1]
            )
            # Hips almost level with ankles → person is lying down (FALLEN)
            if abs(hip_y - ankle_y) < h * 0.15:
                return Posture.FALLEN
            # Hips moderately close to ankles → crouching/squatting
            if abs(hip_y - ankle_y) < h * 0.40:
                return Posture.CROUCHING

        # Shoulder height check
        left_shoulder = kps[KP_LEFT_SHOULDER]
        right_shoulder = kps[KP_RIGHT_SHOULDER]
        shoulders_visible = (
            left_shoulder[2] > KEYPOINT_CONF_THRESHOLD
            or right_shoulder[2] > KEYPOINT_CONF_THRESHOLD
        )

        if shoulders_visible and hips_visible:
            shoulder_y = (
                left_shoulder[1]
                if left_shoulder[2] > KEYPOINT_CONF_THRESHOLD
                else right_shoulder[1]
            )
            # Shoulders well below midpoint of bbox → crouching/bent over
            mid_y = (y1 + y2) / 2
            if shoulder_y > mid_y + h * 0.3:
                return Posture.CROUCHING

        return Posture.STANDING

    def _compute_velocity(self, bbox_history: deque) -> float:
        """Compute velocity as average pixel displacement of bbox center."""
        if len(bbox_history) < 2:
            return 0.0

        centers = [
            ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in bbox_history
        ]
        displacements = [
            ((centers[i][0] - centers[i - 1][0]) ** 2
             + (centers[i][1] - centers[i - 1][1]) ** 2) ** 0.5
            for i in range(1, len(centers))
        ]
        return float(np.mean(displacements))

    def _classify_velocity(self, velocity_value: float) -> Velocity:
        if velocity_value >= VELOCITY_RUNNING_PX:
            return Velocity.RUNNING
        if velocity_value >= VELOCITY_WALKING_PX:
            return Velocity.WALKING
        return Velocity.STATIONARY

    def _infer_hands_state(self, kps: np.ndarray) -> HandsState:
        """
        Detect raised hands: wrist above shoulder on either side.
        """
        left_wrist = kps[KP_LEFT_WRIST]
        right_wrist = kps[KP_RIGHT_WRIST]
        left_shoulder = kps[KP_LEFT_SHOULDER]
        right_shoulder = kps[KP_RIGHT_SHOULDER]

        left_raised = (
            left_wrist[2] > KEYPOINT_CONF_THRESHOLD
            and left_shoulder[2] > KEYPOINT_CONF_THRESHOLD
            and left_wrist[1] < left_shoulder[1]  # lower y = higher in frame
        )
        right_raised = (
            right_wrist[2] > KEYPOINT_CONF_THRESHOLD
            and right_shoulder[2] > KEYPOINT_CONF_THRESHOLD
            and right_wrist[1] < right_shoulder[1]
        )

        if left_raised or right_raised:
            return HandsState.RAISED

        # Can't determine if keypoints not visible
        all_visible = (
            left_wrist[2] > KEYPOINT_CONF_THRESHOLD
            or right_wrist[2] > KEYPOINT_CONF_THRESHOLD
        )
        return HandsState.LOWERED if all_visible else HandsState.UNKNOWN

    def _get_zone(self, bbox: list[float]) -> tuple[Optional[str], Optional[str]]:
        """Return (zone_id, zone_name) for the zone containing the bbox center."""
        if not self._zones:
            return None, None

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        point = np.array([cx, cy], dtype=np.float32)

        for zone, polygon in zip(self._zones, self._zone_polygons):
            if len(polygon) < 3:
                continue
            try:
                import cv2
                inside = cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False)
                if inside >= 0:
                    return zone.get("id"), zone.get("name")
            except Exception:
                # Fallback: simple bounding box check
                if (polygon[:, 0].min() <= point[0] <= polygon[:, 0].max()
                        and polygon[:, 1].min() <= point[1] <= polygon[:, 1].max()):
                    return zone.get("id"), zone.get("name")

        return None, None
