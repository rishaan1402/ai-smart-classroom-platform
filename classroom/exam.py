"""
Exam proctoring: per-student signals → reviewable flags.

Flags are prompts for an invigilator, never verdicts. Each flag is a
sustained signal (shorter blips are ignored), carries the evidence that
triggered it and a short clip, and is confirmed or dismissed by a person
(api/routes/lessons.py, dashboard Exam review page).

Signals, per sampled frame, for each seated student:
  gaze_neighbour  head turned well off the local consensus, toward an
                  adjacent seat whose student is not facing back
  talking         two adjacent students facing each other (both flagged)
  turned_around   back to camera, or an extreme head turn
  reaching        a wrist moved well away from the student's own normal
                  position, towards a neighbour's seat
  device          a phone-class object near a wrist. COCO "cell phone"
                  also fires on calculators, so exams that allow
                  calculators should run with flag_devices=False
  out_of_seat     the same persistence + distance gate as lesson mode

Every signal is judged against the student's *own* normal, not an absolute
norm: students at row ends sit angled, and perspective and handedness move
a writing wrist anywhere relative to the seat, so absolute rules flagged
ordinary posture all exam long.

Signals are suppressed while a non-student (invigilator) is next to the
student: overlapping bodies scramble keypoints, and the invigilator leaning
in is legitimate.

Only students matched to their seat in the current frame are assessed for
head/hand signals (a passing invigilator inherits a seat for a few seconds
through smoothing), and only if at least min_person_px tall; smaller
students are reported as not assessable rather than flagged on noise.
"""

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .headpose import KP_CONF, estimate_head_pose
from .seatmap import SeatMap
from .states import PersonObs, local_yaw_consensus

KP_WRISTS = (9, 10)

SIGNALS = ("gaze_neighbour", "talking", "turned_around", "reaching", "device", "out_of_seat")


@dataclass
class ExamConfig:
    gaze_yaw: float = 0.35          # shoulder widths off the local consensus
    turned_yaw: float = 0.8
    min_head_confidence: float = 0.35
    min_person_px: float = 80.0     # smaller students: head/hand signals not assessable
    baseline_min_history: int = 15  # frames of a student's own yaw/wrist history before judging
    consensus_neighbours: int = 6
    reach_fraction: float = 0.5     # wrist moved this far (of the seat-to-seat distance) towards a neighbour
    min_duration_s: dict = field(default_factory=lambda: {
        "gaze_neighbour": 3.0, "talking": 2.0, "turned_around": 2.0,
        "reaching": 2.0, "device": 2.0, "out_of_seat": 3.0,
    })
    max_gap_s: float = 2.0          # a signal may drop out this long and still count as one episode
    flag_devices: bool = True
    other_person_factor: float = 1.5  # non-student within this × student box width → suppress
    high_priority_score: float = 0.6


@dataclass
class ExamFlag:
    id: str
    type: str
    seat_id: Optional[str]
    track_id: int
    start_s: float
    end_s: float
    score: float                    # 0–1: duration relative to the minimum × signal strength
    neighbour_seat: Optional[str] = None
    evidence: dict = field(default_factory=dict)
    student_id: Optional[str] = None
    clip: Optional[str] = None
    vlm: Optional[dict] = None      # classroom/vlm_verify.py second opinion
    priority: Optional[float] = None  # score adjusted by the VLM verdict (= score without VLM)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Episode:
    start: float
    last: float
    frames: int = 0
    strength_sum: float = 0.0
    neighbour: Optional[str] = None
    seat: Optional[str] = None


class ExamMonitor:
    def __init__(self, seatmap: SeatMap, cfg: Optional[ExamConfig] = None) -> None:
        self.seatmap = seatmap
        self.cfg = cfg or ExamConfig()
        self._open: dict[tuple[int, str], _Episode] = {}
        self._wrist_offsets: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=90))
        self._yaw_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=90))
        self.assessed_student_frames = 0
        self.student_frames = 0
        self._flags: list[ExamFlag] = []
        self.frames_seen = 0
        self.student_seconds = 0.0
        self._last_t: Optional[float] = None

    # ------------------------------------------------------------------
    # Per frame
    # ------------------------------------------------------------------

    def observe(self, t: float, people: list[PersonObs], width: int, height: int) -> dict[int, dict[str, tuple]]:
        """Update episodes with this frame's signals. Returns {track: {signal: (strength, neighbour)}}."""
        students = [p for p in people if p.is_student and p.seat_id]
        others = [p for p in people if not p.is_student]
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        self.frames_seen += 1
        self.student_seconds += dt * len(students)

        heads = {p.track_id: estimate_head_pose(p.keypoints, p.bbox) for p in students}
        consensus = local_yaw_consensus(students, heads, self.cfg.min_head_confidence, self.cfg.consensus_neighbours)
        rel_yaw = {}
        for p in students:
            h = heads[p.track_id]
            if h.yaw is None or h.confidence < self.cfg.min_head_confidence:
                continue
            ry = h.yaw - consensus[p.track_id]
            history = self._yaw_history[p.track_id]
            baseline = float(np.median(history)) if len(history) >= self.cfg.baseline_min_history else None
            history.append(ry)
            if baseline is not None:
                rel_yaw[p.track_id] = ry - baseline        # deviation from this student's own usual direction
        by_seat = {p.seat_id: p for p in students}

        signals: dict[int, dict[str, tuple]] = defaultdict(dict)
        for p in students:
            tid = p.track_id
            head = heads[tid]
            ry = rel_yaw.get(tid)

            self.student_frames += 1
            if not p.seated_now:
                signals[tid]["out_of_seat"] = (1.0, None)
                continue
            if not p.at_seat_now or (p.bbox[3] - p.bbox[1]) < self.cfg.min_person_px:
                continue
            if self._other_person_near(p, others):
                continue
            self.assessed_student_frames += 1

            if head.back_turned or (ry is not None and abs(ry) >= self.cfg.turned_yaw):
                signals[tid]["turned_around"] = (min(1.0, abs(ry or self.cfg.turned_yaw) / self.cfg.turned_yaw), None)
            elif ry is not None and abs(ry) >= self.cfg.gaze_yaw:
                nb = self._neighbour_in_direction(p.seat_id, ry)
                if nb is not None:
                    nb_person = by_seat.get(nb)
                    nb_ry = rel_yaw.get(nb_person.track_id) if nb_person else None
                    facing_back = nb_ry is not None and abs(nb_ry) >= self.cfg.gaze_yaw and \
                        self._neighbour_in_direction(nb, nb_ry) == p.seat_id
                    kind = "talking" if facing_back else "gaze_neighbour"
                    signals[tid][kind] = (min(1.0, abs(ry) / (2 * self.cfg.gaze_yaw)), nb)

            reach_nb = self._reaching(p, width, height)
            if reach_nb is not None:
                signals[tid]["reaching"] = (1.0, reach_nb)

            if p.has_phone and self.cfg.flag_devices:
                signals[tid]["device"] = (1.0, None)

        self._update_episodes(t, students, signals)
        return signals

    def _other_person_near(self, p: PersonObs, others: list[PersonObs]) -> bool:
        cx, cy = (p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2
        reach = self.cfg.other_person_factor * (p.bbox[2] - p.bbox[0])
        return any(
            np.hypot((o.bbox[0] + o.bbox[2]) / 2 - cx, (o.bbox[1] + o.bbox[3]) / 2 - cy) < reach for o in others
        )

    def _neighbour_in_direction(self, seat_id: str, rel_yaw: float) -> Optional[str]:
        """Nearest adjacent seat on the side the head is turned towards (image-space x)."""
        me = self.seatmap.position(seat_id)
        if me is None:
            return None
        best, best_d = None, np.inf
        for nb in self.seatmap.neighbours(seat_id):
            pos = self.seatmap.position(nb)
            if pos is None or np.sign(pos[0] - me[0]) != np.sign(rel_yaw):
                continue
            d = np.hypot(pos[0] - me[0], pos[1] - me[1])
            if d < best_d:
                best, best_d = nb, d
        return best

    def _reaching(self, p: PersonObs, width: int, height: int) -> Optional[str]:
        """
        Neighbour seat a wrist has moved towards, relative to this student's own normal.

        Absolute wrist positions don't work: camera perspective and handedness
        put a writing wrist anywhere relative to the seat anchor (a pen on
        one's own paper can sit, in the image, towards the next row). So each
        student gets a running median wrist offset from their seat, and
        reaching means the wrist moving away from that baseline towards a
        neighbour by ≥ reach_fraction of the distance to them.
        """
        own = self.seatmap.position(p.seat_id)
        if own is None:
            return None
        kps = np.asarray(p.keypoints)
        best, best_progress = None, 0.0
        for w in KP_WRISTS:
            if kps[w, 2] < KP_CONF:
                continue
            offset = np.array([kps[w, 0] / width - own[0], kps[w, 1] / height - own[1]])
            history = self._wrist_offsets[(p.track_id, w)]
            baseline = np.median(np.array(history), axis=0) if len(history) >= self.cfg.baseline_min_history else None
            history.append(offset)
            if baseline is None:
                continue
            move = offset - baseline
            for nb in self.seatmap.neighbours(p.seat_id):
                pos = self.seatmap.position(nb)
                towards = np.array([pos[0] - own[0], pos[1] - own[1]])
                dist = float(np.linalg.norm(towards))
                if dist < 1e-6:
                    continue
                progress = float(move @ towards) / dist ** 2          # fraction of the way to the neighbour
                if progress >= self.cfg.reach_fraction and progress > best_progress:
                    best, best_progress = nb, progress
        return best

    # ------------------------------------------------------------------
    # Episodes → flags
    # ------------------------------------------------------------------

    def _update_episodes(self, t: float, students: list[PersonObs], signals: dict) -> None:
        active = set()
        for p in students:
            for kind, (strength, nb) in signals.get(p.track_id, {}).items():
                key = (p.track_id, kind)
                active.add(key)
                ep = self._open.get(key)
                if ep is None or t - ep.last > self.cfg.max_gap_s:
                    if ep is not None:
                        self._close(key, ep)
                    ep = self._open[key] = _Episode(start=t, last=t, seat=p.seat_id)
                ep.last = t
                ep.frames += 1
                ep.strength_sum += strength
                ep.neighbour = nb or ep.neighbour
        for key, ep in list(self._open.items()):
            if key not in active and t - ep.last > self.cfg.max_gap_s:
                self._close(key, ep)

    def _close(self, key: tuple[int, str], ep: _Episode) -> None:
        self._open.pop(key, None)
        tid, kind = key
        duration = ep.last - ep.start
        min_d = self.cfg.min_duration_s[kind]
        if duration < min_d:
            return
        strength = ep.strength_sum / max(1, ep.frames)
        score = round(min(1.0, duration / (3 * min_d)) * strength, 3)
        self._flags.append(ExamFlag(
            id=f"F{len(self._flags) + 1:04d}", type=kind, seat_id=ep.seat, track_id=tid,
            start_s=round(ep.start, 2), end_s=round(ep.last, 2), score=score, neighbour_seat=ep.neighbour,
            evidence={"duration_s": round(duration, 1), "frames": ep.frames, "mean_strength": round(strength, 2)},
        ))

    def finish(self) -> list[ExamFlag]:
        for key, ep in list(self._open.items()):
            self._close(key, ep)
        return sorted(self._flags, key=lambda f: (-f.score, f.start_s))

    def summary(self, flags: list[ExamFlag]) -> dict:
        hours = self.student_seconds / 3600
        counts = {s: sum(f.type == s for f in flags) for s in SIGNALS}
        return {
            "flags": len(flags),
            "high_priority": sum(f.score >= self.cfg.high_priority_score for f in flags),
            "by_type": counts,
            "student_hours": round(hours, 3),
            "flags_per_student_hour": round(len(flags) / hours, 2) if hours else None,
            "assessable_share": round(self.assessed_student_frames / max(1, self.student_frames), 3),
        }


def room_noise_flags(curve, min_seconds: int = 3) -> list[dict]:
    """Class-level 'talking in the room' flags from audio.room.noise_curve output."""
    flags, start = [], None
    noisy = list(curve["noisy"]) + [False]
    for sec, is_noisy in enumerate(noisy):
        if is_noisy and start is None:
            start = sec
        elif not is_noisy and start is not None:
            if sec - start >= min_seconds:
                flags.append({"type": "room_talking", "start_s": float(start), "end_s": float(sec)})
            start = None
    return flags
