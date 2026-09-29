"""
Classroom tokens & enums — per-student semantic state for one sampled frame.

Mirrors sentinel_engine.tokens: enums are str-valued so they serialize
cleanly, and ClassroomToken.to_dict() produces a flat row for Parquet.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StudentState(str, Enum):
    ATTENTIVE = "attentive"          # head up, facing with the class
    WRITING = "writing"              # head down, writing hand moving
    HAND_RAISED = "hand_raised"
    HEAD_DOWN = "head_down"          # head down, hands idle or head resting on arms
    TURNED_AWAY = "turned_away"      # facing clearly away from the class direction
    INTERACTING = "interacting"      # facing a neighbour who faces back
    PHONE = "phone"
    OUT_OF_SEAT = "out_of_seat"
    UNKNOWN = "unknown"              # not enough keypoint evidence


ENGAGED_STATES = frozenset({StudentState.ATTENTIVE, StudentState.WRITING, StudentState.HAND_RAISED})
DISENGAGED_STATES = frozenset({
    StudentState.HEAD_DOWN,
    StudentState.TURNED_AWAY,
    StudentState.INTERACTING,
    StudentState.PHONE,
    StudentState.OUT_OF_SEAT,
})


class Role(str, Enum):
    STUDENT = "student"   # occupies a seat most of the time
    OTHER = "other"       # teacher, visitor — excluded from class metrics


@dataclass
class HeadPose:
    """
    Head orientation estimated from COCO face/shoulder keypoints.

    Attributes:
        yaw: Nose offset from head centre, in shoulder widths. Image space:
             positive = nose towards image right. ~0 when facing the camera.
        pitch_ratio: (shoulder line y − nose y) / shoulder width. Large when
             the head is up, small or negative when the head is down.
        ear_lift: (shoulder line y − mean ear y) / shoulder width; ~0 or
             negative when the head rests on the desk.
        face_visibility: Mean confidence of nose + eyes (0–1).
        back_turned: No face, ears well above the shoulders → back to camera.
        confidence: Evidence strength for the estimate (0–1).
    """
    yaw: Optional[float]
    pitch_ratio: Optional[float]
    ear_lift: Optional[float]
    face_visibility: float
    back_turned: bool
    confidence: float


@dataclass
class ClassroomToken:
    """One student (track) in one sampled frame."""
    frame_idx: int
    timestamp_s: float
    camera_id: str
    track_id: int
    seat_id: Optional[str]
    role: Role
    bbox: list[float]
    head: HeadPose
    state: StudentState
    state_confidence: float
    phone: bool = False
    cues: dict = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_s": round(self.timestamp_s, 3),
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "seat_id": self.seat_id,
            "role": self.role.value,
            "x1": self.bbox[0], "y1": self.bbox[1], "x2": self.bbox[2], "y2": self.bbox[3],
            "yaw": self.head.yaw,
            "pitch_ratio": self.head.pitch_ratio,
            "ear_lift": self.head.ear_lift,
            "face_visibility": self.head.face_visibility,
            "back_turned": self.head.back_turned,
            "head_confidence": self.head.confidence,
            "state": self.state.value,
            "state_confidence": round(self.state_confidence, 3),
            "phone": self.phone,
            "cues": ",".join(sorted(k for k, v in self.cues.items() if v)),
            "schema_version": self.schema_version,
        }
