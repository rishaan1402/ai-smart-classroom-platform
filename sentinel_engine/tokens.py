"""
Tokens & Enums — core data types for the Sentinel pipeline.

BehaviorToken carries all semantic information about a detected person
through the entire pipeline (CameraWorker → Gatekeeper → Redis Streams
→ workers → EventLogger).

Every token includes:
  - schema_version: bump on any schema change
  - trace_id: UUID generated at CameraWorker, carried end-to-end for debugging
  - severity: set by Gatekeeper based on EventType
"""

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class Posture(str, Enum):
    STANDING = "standing"
    CROUCHING = "crouching"
    FALLEN = "fallen"
    UNKNOWN = "unknown"


class Velocity(str, Enum):
    STATIONARY = "stationary"
    WALKING = "walking"
    RUNNING = "running"


class HandsState(str, Enum):
    LOWERED = "lowered"
    RAISED = "raised"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    FALL_DETECTED = "FALL_DETECTED"
    RUNNING_IN_ZONE = "RUNNING_IN_ZONE"
    HANDS_RAISED = "HANDS_RAISED"
    AGGRESSIVE_BEHAVIOR = "AGGRESSIVE_BEHAVIOR"
    UNKNOWN_PERSON = "UNKNOWN_PERSON"
    LOITERING = "LOITERING"
    ZONE_INTRUSION = "ZONE_INTRUSION"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Gatekeeper severity mapping — authoritative source
EVENT_SEVERITY: dict[EventType, Severity] = {
    EventType.FALL_DETECTED: Severity.CRITICAL,
    EventType.ZONE_INTRUSION: Severity.HIGH,
    EventType.HANDS_RAISED: Severity.HIGH,
    EventType.AGGRESSIVE_BEHAVIOR: Severity.HIGH,
    EventType.UNKNOWN_PERSON: Severity.MEDIUM,
    EventType.RUNNING_IN_ZONE: Severity.MEDIUM,
    EventType.LOITERING: Severity.LOW,
}


@dataclass
class BehaviorToken:
    """
    Semantic description of a detected person in a single frame.

    Attributes:
        person_id: Tracker-assigned persistent ID
        camera_id: Source camera identifier
        tenant_id: Tenant this detection belongs to
        bbox: Bounding box [x1, y1, x2, y2]
        keypoints: YOLO pose keypoints array (17×3: x, y, conf)
        posture: Inferred posture
        velocity: Inferred movement speed
        hands_state: Whether hands appear raised
        zone_id: ID of zone person is in (None if no zone)
        zone_name: Human-readable zone name
        velocity_value: Raw velocity in pixels/frame
        severity: Set by Gatekeeper when an event fires (None if no event)
        timestamp: Unix timestamp of detection
        trace_id: UUID for end-to-end log correlation
        schema_version: Bump on any schema change
    """
    person_id: int
    camera_id: str
    tenant_id: str
    bbox: list[float]
    keypoints: list[list[float]]
    posture: Posture
    velocity: Velocity
    hands_state: HandsState
    zone_id: Optional[str]
    zone_name: Optional[str]
    velocity_value: float
    severity: Optional[Severity] = None
    timestamp: float = field(default_factory=time.time)
    trace_id: UUID = field(default_factory=uuid4)
    schema_version: int = 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["posture"] = self.posture.value
        d["velocity"] = self.velocity.value
        d["hands_state"] = self.hands_state.value
        d["severity"] = self.severity.value if self.severity else None
        d["trace_id"] = str(self.trace_id)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "BehaviorToken":
        d = d.copy()
        d["posture"] = Posture(d["posture"])
        d["velocity"] = Velocity(d["velocity"])
        d["hands_state"] = HandsState(d["hands_state"])
        d["severity"] = Severity(d["severity"]) if d.get("severity") else None
        d["trace_id"] = UUID(d["trace_id"]) if isinstance(d.get("trace_id"), str) else d.get("trace_id", uuid4())
        return cls(**d)


@dataclass
class SentinelEvent:
    """
    A semantic event emitted by the Gatekeeper.

    Enriched progressively as it moves through the pipeline:
      CameraWorker → Gatekeeper → LLMWorker → VLMWorker → FaceWorker → EventLogger
    """
    event_type: EventType
    severity: Severity
    person_id: int
    camera_id: str
    tenant_id: str
    token: BehaviorToken
    trace_id: UUID
    timestamp: float = field(default_factory=time.time)
    zone_id: Optional[str] = None
    requires_vlm: bool = False
    requires_face_id: bool = False
    schema_version: int = 1

    # Enrichment fields — filled by downstream workers
    threat_level: Optional[str] = None           # LLMWorker / VLMWorker
    threat_description: Optional[str] = None     # LLMWorker / VLMWorker
    face_name: Optional[str] = None              # FaceWorker
    face_confidence: float = 0.0                 # FaceWorker
    face_recognition_was_active: bool = False    # FaceWorker
    snapshot_path: Optional[str] = None          # EventLogger
    clip_path: Optional[str] = None              # ClipEncoder (backfilled)
    enrichment_skipped: bool = False             # StreamTrimmer forced persistence
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        d["severity"] = self.severity.value
        d["trace_id"] = str(self.trace_id)
        d["token"] = self.token.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "SentinelEvent":
        d = d.copy()
        d["event_type"] = EventType(d["event_type"])
        d["severity"] = Severity(d["severity"])
        d["trace_id"] = UUID(d["trace_id"]) if isinstance(d.get("trace_id"), str) else d.get("trace_id", uuid4())
        if isinstance(d.get("token"), dict):
            d["token"] = BehaviorToken.from_dict(d["token"])
        return cls(**d)
