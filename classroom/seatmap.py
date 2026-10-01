"""
Seat map — stable, anonymous per-student identity from seat position.

Seats are learned from where people stay still (calibrate), saved per room
as JSON, and matched to tracker IDs every frame with the Hungarian
algorithm. Each seat gets its own match radius (half the distance to its
nearest neighbour), so near-camera seats, which are further apart in
pixels, still match correctly.

Coordinates are normalized (0–1) so a map survives resolution changes.
"""

import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

DEFAULT_RADIUS = 0.05
MIN_RADIUS = 0.015
MAX_RADIUS = 0.08


@dataclass
class Seat:
    seat_id: str
    x: float
    y: float
    radius: float = DEFAULT_RADIUS


class SeatMap:
    def __init__(self, seats: list[Seat], camera_id: str = "") -> None:
        self.camera_id = camera_id
        self.seats = seats
        self._pts = np.array([[s.x, s.y] for s in seats], dtype=np.float32).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.seats)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"camera_id": self.camera_id, "seats": [asdict(s) for s in self.seats]}, indent=2,
        ))

    @classmethod
    def from_json(cls, path: Path) -> "SeatMap":
        data = json.loads(path.read_text())
        return cls([Seat(**s) for s in data["seats"]], camera_id=data.get("camera_id", ""))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @classmethod
    def calibrate(
        cls,
        anchors_by_frame: list[np.ndarray],
        camera_id: str = "",
        min_occupancy: float = 0.3,
        density_factor: float = 0.4,
        suppress_factor: float = 0.8,
        default_scale: float = 0.06,
    ) -> "SeatMap":
        """
        Learn seats from per-frame person anchors by density peaks.

        Each anchor's density is the number of distinct frames with an anchor
        within density_factor × its person scale (normalized shoulder width).
        Anchors are taken in descending density as seat centres, suppressing
        any within suppress_factor × scale of an existing centre. Scaling by
        person size keeps small, tightly packed back-row seats apart while
        tolerating the larger jitter of near-camera students; a walking path
        never builds a dense peak, so the teacher doesn't become a seat.

        Args:
            anchors_by_frame: One array per frame, shape (N_i, 3) as
                [x, y, scale] normalized, or (N_i, 2) to use default_scale.
            min_occupancy: Minimum fraction of frames a seat must be occupied;
                filters out walkers (teacher) and one-off false detections.
        """
        n_frames = len(anchors_by_frame)
        non_empty = [(i, a) for i, a in enumerate(anchors_by_frame) if len(a)]
        if not non_empty:
            return cls([], camera_id)
        data = np.concatenate([
            a if a.shape[1] == 3 else np.column_stack([a, np.full(len(a), default_scale)])
            for _, a in non_empty
        ]).astype(np.float32)
        pts, scale = data[:, :2], data[:, 2]
        frame_of = np.concatenate([np.full(len(a), i) for i, a in non_empty])

        tree = cKDTree(pts)
        neighbours = tree.query_ball_point(pts, r=density_factor * scale)
        density = np.array([len(np.unique(frame_of[nb])) for nb in neighbours])

        centres: list[np.ndarray] = []
        centre_scales: list[float] = []
        for i in np.argsort(-density, kind="stable"):
            if density[i] / n_frames < min_occupancy:
                break
            if any(
                np.linalg.norm(pts[i] - c) < suppress_factor * max(scale[i], s)
                for c, s in zip(centres, centre_scales)
            ):
                continue
            core = neighbours[i]
            centres.append(np.median(pts[core], axis=0))
            centre_scales.append(float(np.median(scale[core])))
        if not centres:
            return cls([], camera_id)

        return cls(_name_and_size(np.array(centres)), camera_id)

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign(self, anchors: dict[int, tuple[float, float]]) -> dict[int, Optional[str]]:
        """
        Match track anchors (normalized) to seats, one track per seat.

        Returns track_id → seat_id, or None when no seat is within its radius.
        """
        result: dict[int, Optional[str]] = {tid: None for tid in anchors}
        if not anchors or not self.seats:
            return result
        tids = list(anchors)
        pts = np.array([anchors[t] for t in tids], dtype=np.float32)
        dist = np.linalg.norm(pts[:, None, :] - self._pts[None, :, :], axis=2)
        radii = np.array([s.radius for s in self.seats])
        cost = np.where(dist <= radii[None, :], dist / radii[None, :], 1e6)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] < 1e6:
                result[tids[r]] = self.seats[c].seat_id
        return result

    def neighbours(self, seat_id: str, max_factor: float = 2.5) -> list[str]:
        """Seats within max_factor × this seat's radius-derived spacing."""
        idx = next((i for i, s in enumerate(self.seats) if s.seat_id == seat_id), None)
        if idx is None:
            return []
        spacing = 2 * self.seats[idx].radius
        dist = np.linalg.norm(self._pts - self._pts[idx], axis=1)
        return [self.seats[i].seat_id for i in np.where((dist > 0) & (dist <= max_factor * spacing))[0]]

    def position(self, seat_id: str) -> Optional[tuple[float, float]]:
        seat = next((s for s in self.seats if s.seat_id == seat_id), None)
        return (seat.x, seat.y) if seat else None

    def distance_in_radii(self, seat_id: Optional[str], point: tuple[float, float]) -> float:
        """How far a normalized point is from a seat, in units of that seat's radius (inf if unknown)."""
        seat = next((s for s in self.seats if s.seat_id == seat_id), None)
        if seat is None:
            return float("inf")
        return float(np.hypot(point[0] - seat.x, point[1] - seat.y) / max(seat.radius, 1e-6))


def _name_and_size(centres: np.ndarray) -> list[Seat]:
    """Order seats into rows (by y gaps) and columns, and size their radii."""
    if len(centres) == 1:
        return [Seat("R1C1", float(centres[0, 0]), float(centres[0, 1]), DEFAULT_RADIUS)]

    tree = cKDTree(centres)
    nn_dist = tree.query(centres, k=2)[0][:, 1]
    radii = np.clip(nn_dist / 2, MIN_RADIUS, MAX_RADIUS)

    # Row split: a new row starts where the y gap exceeds ~0.6 of the local seat spacing
    order = np.argsort(centres[:, 1])
    rows: list[list[int]] = [[order[0]]]
    for prev, cur in zip(order[:-1], order[1:]):
        gap = centres[cur, 1] - centres[prev, 1]
        if gap > 0.6 * min(nn_dist[prev], nn_dist[cur]):
            rows.append([cur])
        else:
            rows[-1].append(cur)

    seats = []
    for r, members in enumerate(rows, start=1):
        for c, i in enumerate(sorted(members, key=lambda j: centres[j, 0]), start=1):
            seats.append(Seat(f"R{r}C{c}", float(centres[i, 0]), float(centres[i, 1]), float(radii[i])))
    return seats


class SeatTracker:
    """
    Smooths track→seat assignments over time and derives each track's role.

    A track's seat is the most common seat in its recent assignments. A track
    that has been seated for at least `student_share` of its lifetime is a
    student; otherwise it is OTHER (teacher, visitor).

    Out of seat needs both persistence and displacement: a single frame without
    a seat match is usually assignment jitter (leaning, occlusion), not a
    student leaving their seat.
    """

    def __init__(self, window: int = 15, student_share: float = 0.5) -> None:
        self._recent: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self._seated: Counter = Counter()
        self._seen: Counter = Counter()
        self._unseated_run: Counter = Counter()
        self._student_share = student_share

    def update(self, assignments: dict[int, Optional[str]]) -> dict[int, Optional[str]]:
        smoothed = {}
        for tid, seat in assignments.items():
            self._seen[tid] += 1
            if seat is not None:
                self._seated[tid] += 1
                self._unseated_run[tid] = 0
            else:
                self._unseated_run[tid] += 1
            self._recent[tid].append(seat)
            votes = Counter(s for s in self._recent[tid] if s is not None)
            smoothed[tid] = votes.most_common(1)[0][0] if votes else None
        return smoothed

    def is_student(self, track_id: int) -> bool:
        seen = self._seen[track_id]
        return seen > 0 and self._seated[track_id] / seen >= self._student_share

    def currently_seated(self, track_id: int) -> bool:
        recent = self._recent.get(track_id)
        return bool(recent) and recent[-1] is not None

    def is_out_of_seat(self, track_id: int, away_radii: float, min_frames: int, min_away_radii: float = 2.0) -> bool:
        """Unmatched for ≥ min_frames consecutive frames AND ≥ min_away_radii from its usual seat."""
        return self._unseated_run[track_id] >= min_frames and away_radii >= min_away_radii
