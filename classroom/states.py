"""
StateClassifier — per-student classroom state from pose + head cues.

Rules, in priority order (first match wins):
  1. OUT_OF_SEAT   student track currently not on any seat
  2. HAND_RAISED   a wrist above the nose, or (small children) a wrist above
                   the shoulder line beside the face with the elbow lifted.
                   A hand on the chin or cheek doesn't count.
  3. PHONE         a detected phone inside the student's hand/lap region
  4. TURNED_AWAY   back to camera while the class faces it
  5. HEAD_DOWN / WRITING
                   head lowered vs. the upright baseline: the higher of the
                   student's own and the whole camera's (a student who only
                   ever works heads-down would otherwise look "upright");
                   HEAD_DOWN if resting (head at shoulder level or on the
                   arms), else WRITING, i.e. working at the desk. Wrist motion
                   raises WRITING confidence, but isn't required: at 3 fps
                   with desks hiding hands it is rarely observable.
  6. INTERACTING / TURNED_AWAY
                   yaw deviates from the local consensus; INTERACTING when
                   an adjacent seat faces back towards this student
  7. ATTENTIVE     head up, facing with the class
  8. UNKNOWN       not enough keypoint evidence

"Facing with the class" is relative: the consensus for each student is the
median yaw of their nearest seated classmates in the frame. Local, not global,
because a wide-angle camera sees students on the left and right of the room
at different angles even when all face the board. This keeps the logic
camera-angle agnostic, with no board position to calibrate. It assumes most
of a neighbourhood faces the lesson most of the time.

TemporalSmoother then takes a majority vote over a short window per track,
so one noisy frame can't flip a student's state.
"""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .headpose import (
    KP_CONF,
    KP_LEFT_SHOULDER,
    KP_NOSE,
    KP_RIGHT_SHOULDER,
    estimate_head_pose,
    shoulder_width,
)
from .seatmap import SeatMap
from .tokens import HeadPose, StudentState

KP_LEFT_ELBOW, KP_RIGHT_ELBOW = 7, 8
KP_LEFT_WRIST, KP_RIGHT_WRIST = 9, 10


@dataclass
class ClassifierConfig:
    yaw_turn_threshold: float = 0.35      # shoulder widths away from consensus
    head_down_ratio: float = 0.55         # pitch below this share of own baseline
    baseline_percentile: float = 80.0
    baseline_min_samples: int = 6
    camera_baseline_min_samples: int = 50
    face_hidden_visibility: float = 0.25  # face keypoints this weak + shoulders → head down
    hand_raise_margin: float = 0.0        # wrist above nose by this many shoulder widths
    face_level_wrist_lift: float = 0.3    # face-level raise: wrist this far above the shoulder line
    face_level_elbow_drop: float = 0.6    # …elbow no lower than this below the shoulder line
    face_level_side_offset: float = 0.5   # …wrist this far sideways from the nose (not on the chin)
    writing_motion: float = 0.06          # wrist std (shoulder widths) that counts as writing
    motion_window: int = 6                # sampled frames of wrist history
    min_head_confidence: float = 0.35
    resting_pitch: float = -0.1           # nose/ears below the shoulder line → head resting on the desk
    consensus_neighbours: int = 6         # nearest classmates forming the local facing consensus


@dataclass
class PersonObs:
    """One tracked person in one frame, as fed to the classifier."""
    track_id: int
    bbox: list[float]
    keypoints: np.ndarray          # (17, 3)
    seat_id: Optional[str]
    is_student: bool
    seated_now: bool
    has_phone: bool = False


@dataclass
class Classification:
    state: StudentState
    confidence: float
    head: HeadPose
    cues: dict


class StateClassifier:
    def __init__(self, seatmap: SeatMap, cfg: Optional[ClassifierConfig] = None) -> None:
        self.seatmap = seatmap
        self.cfg = cfg or ClassifierConfig()
        self._pitch_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=300))
        self._camera_pitch: deque = deque(maxlen=20_000)
        self._wrist_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.cfg.motion_window))

    def classify_frame(self, people: list[PersonObs]) -> dict[int, Classification]:
        """Classify every student in one frame. Non-students are skipped."""
        heads = {p.track_id: estimate_head_pose(p.keypoints, p.bbox) for p in people}
        students = [p for p in people if p.is_student]

        consensus = self._local_consensus(students, heads)

        # Seat → relative yaw, for neighbour interaction checks
        seat_rel_yaw = {
            p.seat_id: heads[p.track_id].yaw - consensus[p.track_id]
            for p in students
            if p.seat_id and heads[p.track_id].yaw is not None
        }

        for p in students:
            h = heads[p.track_id]
            if h.pitch_ratio is not None and h.confidence >= self.cfg.min_head_confidence:
                self._pitch_history[p.track_id].append(h.pitch_ratio)
                self._camera_pitch.append(h.pitch_ratio)

        camera_baseline = (
            float(np.percentile(self._camera_pitch, self.cfg.baseline_percentile))
            if len(self._camera_pitch) >= self.cfg.camera_baseline_min_samples else None
        )
        return {
            p.track_id: self._classify(p, heads[p.track_id], consensus[p.track_id], seat_rel_yaw, camera_baseline)
            for p in students
        }

    def _local_consensus(self, students: list[PersonObs], heads: dict[int, HeadPose]) -> dict[int, float]:
        """Median yaw of each student's nearest valid seated classmates (0.0 if too few)."""
        valid = [
            p for p in students
            if p.seated_now and heads[p.track_id].yaw is not None
            and heads[p.track_id].confidence >= self.cfg.min_head_confidence
        ]
        if len(valid) < 3:
            return {p.track_id: 0.0 for p in students}
        centres = np.array([[(p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2] for p in valid])
        yaws = np.array([heads[p.track_id].yaw for p in valid])
        k = min(self.cfg.consensus_neighbours, len(valid) - 1)

        consensus = {}
        for p in students:
            me = np.array([(p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2])
            dist = np.linalg.norm(centres - me, axis=1)
            dist[[q.track_id == p.track_id for q in valid]] = np.inf   # exclude self
            nearest = np.argsort(dist)[:k]
            consensus[p.track_id] = float(np.median(yaws[nearest]))
        return consensus

    def forget(self, active_ids: set[int]) -> None:
        for store in (self._pitch_history, self._wrist_history):
            for tid in set(store) - active_ids:
                del store[tid]

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _classify(
        self,
        p: PersonObs,
        head: HeadPose,
        consensus_yaw: float,
        seat_rel_yaw: dict[str, float],
        camera_baseline: Optional[float],
    ) -> Classification:
        cfg = self.cfg
        kps = np.asarray(p.keypoints, dtype=np.float32)
        scale = shoulder_width(kps, p.bbox)
        wrist_motion = self._update_wrist_motion(p.track_id, kps, scale)
        cues = {"wrist_moving": wrist_motion is not None and wrist_motion >= cfg.writing_motion}

        def result(state: StudentState, conf: float) -> Classification:
            return Classification(state, float(np.clip(conf, 0.0, 1.0)), head, cues)

        if not p.seated_now:
            return result(StudentState.OUT_OF_SEAT, 0.6)

        if self._hand_raised(kps, scale):
            cues["hand_above_head"] = True
            return result(StudentState.HAND_RAISED, 0.8)

        if p.has_phone:
            cues["phone_in_hands"] = True
            return result(StudentState.PHONE, 0.7)

        if head.back_turned:
            cues["back_turned"] = True
            return result(StudentState.TURNED_AWAY, 0.5 + 0.3 * head.confidence)

        head_down = self._head_down(p.track_id, head, camera_baseline, kps)
        if head_down:
            cues["head_down"] = True
            if self._head_on_arms(kps) or self._head_resting(head):
                cues["head_resting"] = True
                return result(StudentState.HEAD_DOWN, 0.65)
            if cues["wrist_moving"]:
                return result(StudentState.WRITING, 0.6 + 0.3 * min(1.0, wrist_motion / (3 * cfg.writing_motion)))
            return result(StudentState.WRITING, 0.45)

        if head.yaw is None or head.confidence < cfg.min_head_confidence:
            return result(StudentState.UNKNOWN, 0.0)

        rel_yaw = head.yaw - consensus_yaw
        if abs(rel_yaw) >= cfg.yaw_turn_threshold:
            cues["yaw_deviates"] = True
            if p.seat_id and self._neighbour_faces_back(p.seat_id, rel_yaw, seat_rel_yaw):
                cues["neighbour_faces_back"] = True
                return result(StudentState.INTERACTING, 0.6)
            return result(StudentState.TURNED_AWAY, 0.45 + 0.4 * min(1.0, abs(rel_yaw)))

        return result(StudentState.ATTENTIVE, 0.4 + 0.5 * head.confidence)

    def _hand_raised(self, kps: np.ndarray, scale: float) -> bool:
        if kps[KP_NOSE, 2] < KP_CONF:
            return False
        cfg = self.cfg
        nose_x, nose_y = kps[KP_NOSE, 0], kps[KP_NOSE, 1]
        shoulders = [s for s in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER) if kps[s, 2] >= KP_CONF]
        shoulder_y = float(np.mean(kps[shoulders, 1])) if shoulders else None

        for wrist, elbow in ((KP_LEFT_WRIST, KP_LEFT_ELBOW), (KP_RIGHT_WRIST, KP_RIGHT_ELBOW)):
            if kps[wrist, 2] < KP_CONF:
                continue
            wx, wy = kps[wrist, 0], kps[wrist, 1]
            if wy < nose_y - cfg.hand_raise_margin * scale:
                return True
            # Face-level raise (small children): hand up beside the face, elbow lifted
            if (
                shoulder_y is not None
                and kps[elbow, 2] >= KP_CONF
                and wy < shoulder_y - cfg.face_level_wrist_lift * scale
                and kps[elbow, 1] < shoulder_y + cfg.face_level_elbow_drop * scale
                and abs(wx - nose_x) > cfg.face_level_side_offset * scale
            ):
                return True
        return False

    def _head_down(
        self, track_id: int, head: HeadPose, camera_baseline: Optional[float], kps: np.ndarray,
    ) -> bool:
        shoulders_ok = kps[KP_LEFT_SHOULDER, 2] >= KP_CONF and kps[KP_RIGHT_SHOULDER, 2] >= KP_CONF
        if shoulders_ok and head.face_visibility < self.cfg.face_hidden_visibility:
            return True
        if head.pitch_ratio is None:
            return False
        history = self._pitch_history[track_id]
        candidates = [camera_baseline] if camera_baseline is not None else []
        if len(history) >= self.cfg.baseline_min_samples:
            candidates.append(float(np.percentile(history, self.cfg.baseline_percentile)))
        baseline = max(candidates, default=None)
        if baseline is None or baseline <= 0:
            return False
        return head.pitch_ratio < self.cfg.head_down_ratio * baseline

    def _head_on_arms(self, kps: np.ndarray) -> bool:
        """Both wrists up at shoulder height or higher, next to a lowered head."""
        wrists = [w for w in (KP_LEFT_WRIST, KP_RIGHT_WRIST) if kps[w, 2] >= KP_CONF]
        shoulders = [s for s in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER) if kps[s, 2] >= KP_CONF]
        if len(wrists) < 2 or not shoulders:
            return False
        shoulder_y = float(np.mean(kps[shoulders, 1]))
        return all(kps[w, 1] <= shoulder_y for w in wrists)

    def _head_resting(self, head: HeadPose) -> bool:
        """Nose or ears down at shoulder level: head on the desk, not working at it."""
        levels = [v for v in (head.pitch_ratio, head.ear_lift) if v is not None]
        return bool(levels) and min(levels) <= self.cfg.resting_pitch

    def _neighbour_faces_back(self, seat_id: str, rel_yaw: float, seat_rel_yaw: dict[str, float]) -> bool:
        me = self.seatmap.position(seat_id)
        if me is None:
            return False
        for nb in self.seatmap.neighbours(seat_id):
            nb_yaw = seat_rel_yaw.get(nb)
            nb_pos = self.seatmap.position(nb)
            if nb_yaw is None or nb_pos is None:
                continue
            towards_nb = np.sign(nb_pos[0] - me[0]) == np.sign(rel_yaw)
            nb_towards_me = np.sign(me[0] - nb_pos[0]) == np.sign(nb_yaw)
            if towards_nb and nb_towards_me and abs(nb_yaw) >= self.cfg.yaw_turn_threshold:
                return True
        return False

    def _update_wrist_motion(self, track_id: int, kps: np.ndarray, scale: float) -> Optional[float]:
        """Std of wrist position (in shoulder widths) over the recent window."""
        visible = [w for w in (KP_LEFT_WRIST, KP_RIGHT_WRIST) if kps[w, 2] >= KP_CONF]
        history = self._wrist_history[track_id]
        history.append(kps[visible, :2].mean(axis=0) / scale if visible else None)
        pts = [h for h in history if h is not None]
        if len(pts) < 3:
            return None
        return float(np.linalg.norm(np.std(np.array(pts), axis=0)))


class TemporalSmoother:
    """Majority vote over the last `window` states of each track."""

    def __init__(self, window: int = 9) -> None:
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id: int, state: StudentState) -> tuple[StudentState, float]:
        history = self._history[track_id]
        history.append(state)
        known = [s for s in history if s != StudentState.UNKNOWN]
        if not known:
            return StudentState.UNKNOWN, 0.0
        winner, votes = Counter(known).most_common(1)[0]
        return winner, votes / len(history)

    def _local_consensus(self, students: list[PersonObs], heads: dict[int, HeadPose]) -> dict[int, float]:
        """Median yaw of each student's nearest valid seated classmates (0.0 if too few)."""
        valid = [
            p for p in students
            if p.seated_now and heads[p.track_id].yaw is not None
            and heads[p.track_id].confidence >= self.cfg.min_head_confidence
        ]
        if len(valid) < 3:
            return {p.track_id: 0.0 for p in students}
        centres = np.array([[(p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2] for p in valid])
        yaws = np.array([heads[p.track_id].yaw for p in valid])
        k = min(self.cfg.consensus_neighbours, len(valid) - 1)

        consensus = {}
        for p in students:
            me = np.array([(p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2])
            dist = np.linalg.norm(centres - me, axis=1)
            dist[[q.track_id == p.track_id for q in valid]] = np.inf   # exclude self
            nearest = np.argsort(dist)[:k]
            consensus[p.track_id] = float(np.median(yaws[nearest]))
        return consensus

    def forget(self, active_ids: set[int]) -> None:
        for tid in set(self._history) - active_ids:
            del self._history[tid]
