"""
Consent-aware student identity: gallery, per-seat identification, head blurring.

Gallery        Enrolled embeddings of consented students only, encrypted at rest
               (Fernet, key from FACE_ENCRYPTION_KEY as in SentinelCore) and
               searched through SentinelCore's FaceIndex.
SeatIdentifier Collects face evidence per seat across a lesson and resolves
               seat → student by voting, combined with the school's seating
               chart. Only (student_id, similarity) votes are kept; embeddings of
               unmatched faces are discarded immediately and never stored.
blur_heads     Blurs the head of everyone not identified as a consented student.
               This is the default for every exported frame or video.

Decision per seat:
  face+chart  face votes agree with the seating chart        → identified
  face        face votes only (no chart entry)               → identified
  chart       chart only, no usable face evidence            → identified, lower confidence
  conflict    face votes point to a different student        → anonymous, flagged for review
  none        no consented match                             → anonymous
"""

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from sentinel_engine.workers.face import FaceIndex

from .consent import ConsentRegistry
from .faces import SFACE_MATCH_THRESHOLD, FaceEmbedder

KEY_ENV = "FACE_ENCRYPTION_KEY"


# ----------------------------------------------------------------------
# Gallery
# ----------------------------------------------------------------------

class Gallery:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._embeddings: dict[str, list[np.ndarray]] = {}
        self._index: Optional[FaceIndex] = None

    @property
    def students(self) -> set[str]:
        return set(self._embeddings)

    def __len__(self) -> int:
        return len(self._embeddings)

    def add(self, student_id: str, embeddings: list[np.ndarray]) -> None:
        self._embeddings.setdefault(student_id, []).extend(_unit(e) for e in embeddings)
        self._index = None

    def remove(self, student_id: str) -> bool:
        removed = self._embeddings.pop(student_id, None) is not None
        self._index = None
        return removed

    def restrict_to(self, allowed: set[str]) -> None:
        """Drop anyone whose consent no longer allows this purpose."""
        for sid in self.students - allowed:
            self.remove(sid)

    def search(self, embedding: np.ndarray, threshold: float) -> Optional[tuple[str, float]]:
        if not self._embeddings:
            return None
        if self._index is None:
            self._build_index()
        hit = self._index.search(_unit(embedding), k=5, threshold=threshold)
        return (hit[1], hit[2]) if hit else None

    def _build_index(self) -> None:
        self._index = FaceIndex(embedding_dim=self.dim)
        self._index.initialize()
        for sid, embs in self._embeddings.items():
            for i, e in enumerate(embs):
                self._index.add(f"{sid}#{i}", sid, e)

    # Encrypted persistence ------------------------------------------------

    def save(self, path: Path, key: Optional[bytes] = None) -> None:
        payload = json.dumps({
            "dim": self.dim,
            "students": {sid: [e.tolist() for e in embs] for sid, embs in self._embeddings.items()},
        }).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_fernet(key).encrypt(payload))

    @classmethod
    def load(cls, path: Path, key: Optional[bytes] = None, dim: int = 128) -> "Gallery":
        gallery = cls(dim)
        if not path.exists():
            return gallery
        data = json.loads(_fernet(key).decrypt(path.read_bytes()))
        gallery.dim = data["dim"]
        for sid, embs in data["students"].items():
            gallery.add(sid, [np.array(e, dtype=np.float32) for e in embs])
        return gallery


def _fernet(key: Optional[bytes]):
    from cryptography.fernet import Fernet
    key = key or os.environ.get(KEY_ENV, "").encode()
    if not key:
        raise RuntimeError(
            f"{KEY_ENV} is not set. Generate one with: "
            "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(key)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)


def enroll_from_images(
    gallery: Gallery, embedder: FaceEmbedder, student_id: str, images: list[np.ndarray], min_face_px: int = 80,
) -> int:
    """Add one embedding per image with exactly one clear face. Returns the number used."""
    embs = []
    for img in images:
        faces = [f for f in embedder.detect(img) if f.size >= min_face_px]
        if len(faces) == 1:
            embs.append(embedder.embed(img, faces[0]))
    if embs:
        gallery.add(student_id, embs)
    return len(embs)


# ----------------------------------------------------------------------
# Per-seat identification
# ----------------------------------------------------------------------

@dataclass
class SeatIdentity:
    seat_id: str
    student_id: Optional[str]
    source: str                   # face+chart | face | chart | conflict | none
    confidence: float
    face_votes: int = 0
    candidates: dict[str, int] = field(default_factory=dict)   # student_id → votes (review aid)


class SeatIdentifier:
    def __init__(
        self,
        gallery: Gallery,
        embedder: FaceEmbedder,
        match_threshold: float = SFACE_MATCH_THRESHOLD,
        min_face_px: int = 24,
        min_votes: int = 3,
        min_share: float = 0.6,
    ) -> None:
        self.gallery = gallery
        self.embedder = embedder
        self.match_threshold = match_threshold
        self.min_face_px = min_face_px
        self.min_votes = min_votes
        self.min_share = min_share
        self._votes: dict[str, list[tuple[Optional[str], float]]] = defaultdict(list)

    def observe(self, frame: np.ndarray, seat_id: str, head: tuple[float, float, float, float]) -> None:
        """Look for one face in the seat's head region; record who it matches (or None)."""
        if not len(self.gallery):
            return
        crop, scale = _upscaled_crop(frame, head, target=160)
        if crop is None:
            return
        faces = [f for f in self.embedder.detect(crop) if f.size / scale >= self.min_face_px]
        if not faces:
            return
        face = max(faces, key=lambda f: f.size * f.score * (0.5 + 0.5 * f.frontalness))
        embedding = self.embedder.embed(crop, face)
        match = self.gallery.search(embedding, self.match_threshold)
        del embedding   # unmatched or matched, the embedding itself is never kept
        self._votes[seat_id].append(match if match else (None, 0.0))

    def resolve(self, seat_ids: list[str], seating_chart: Optional[dict[str, str]] = None,
                allowed: Optional[set[str]] = None) -> dict[str, SeatIdentity]:
        chart = {s: sid for s, sid in (seating_chart or {}).items() if allowed is None or sid in allowed}
        return {seat: self._resolve_seat(seat, chart.get(seat), allowed) for seat in seat_ids}

    def _resolve_seat(self, seat: str, chart_student: Optional[str], allowed: Optional[set[str]]) -> SeatIdentity:
        votes = self._votes.get(seat, [])
        tally: dict[str, int] = defaultdict(int)
        sims: dict[str, list[float]] = defaultdict(list)
        for sid, sim in votes:
            if sid is not None and (allowed is None or sid in allowed):
                tally[sid] += 1
                sims[sid].append(sim)

        face_student, face_conf = None, 0.0
        if tally:
            best, n = max(tally.items(), key=lambda kv: kv[1])
            if n >= self.min_votes and n / len(votes) >= self.min_share:
                face_student = best
                # Share of this seat's face sightings that matched, scaled by match strength
                # (SFace same-person similarity is typically 0.5–0.8; 2× threshold ≈ 0.73)
                strength = min(1.0, float(np.mean(sims[best])) / (2 * self.match_threshold))
                face_conf = (n / len(votes)) * strength

        candidates = dict(tally)
        if face_student and chart_student:
            if face_student == chart_student:
                return SeatIdentity(seat, face_student, "face+chart", min(1.0, face_conf + 0.3), len(votes), candidates)
            return SeatIdentity(seat, None, "conflict", 0.0, len(votes), candidates)
        if face_student:
            return SeatIdentity(seat, face_student, "face", face_conf, len(votes), candidates)
        if chart_student:
            # A clear face that matches nobody enrolled argues against the chart
            unmatched = sum(1 for sid, _ in votes if sid is None)
            if unmatched >= self.min_votes and unmatched / max(1, len(votes)) >= self.min_share:
                return SeatIdentity(seat, None, "conflict", 0.0, len(votes), candidates)
            return SeatIdentity(seat, chart_student, "chart", 0.5, len(votes), candidates)
        return SeatIdentity(seat, None, "none", 0.0, len(votes), candidates)


def _upscaled_crop(frame: np.ndarray, box: tuple[float, float, float, float], target: int):
    """Crop x1,y1,x2,y2 and upscale so the short side is ≥ target px (YuNet misses tiny faces)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, 1.0
    crop = frame[y1:y2, x1:x2]
    scale = max(1.0, target / min(crop.shape[:2]))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, scale


# ----------------------------------------------------------------------
# Head regions & blurring
# ----------------------------------------------------------------------

HEAD_KPS = [0, 1, 2, 3, 4]


def head_box(kps: np.ndarray, bbox: list[float], margin: float = 0.6) -> tuple[float, float, float, float]:
    """Head region from face keypoints (expanded), falling back to the top of the person box."""
    kps = np.asarray(kps)
    visible = kps[HEAD_KPS][kps[HEAD_KPS, 2] >= 0.3]
    x1, y1, x2, y2 = bbox
    if len(visible) >= 2:
        hx1, hy1 = visible[:, :2].min(axis=0)
        hx2, hy2 = visible[:, :2].max(axis=0)
        size = max(hx2 - hx1, hy2 - hy1, 0.25 * (x2 - x1))
        cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        half = size * (0.5 + margin)
        return cx - half, cy - half * 1.1, cx + half, cy + half * 0.9
    return x1, y1, x2, y1 + 0.3 * (y2 - y1)


def blur_heads(frame: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Heavy blur over each head box, in place."""
    h, w = frame.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = frame[y1:y2, x1:x2]
        k = max(15, (min(roi.shape[:2]) // 2) | 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return frame


class FaceBlurrer:
    """
    Whole-frame face detection for exports, so a face the person detector
    missed (back rows, occlusion) is still blurred. Detection only: no
    embedding is computed. Upscales the frame because YuNet misses faces
    smaller than ~20 px.
    """

    def __init__(self, upscale: float = 2.0, score_threshold: float = 0.5) -> None:
        from .faces import YUNET_PATH
        self.upscale = upscale
        self._detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), score_threshold)

    def face_boxes(self, frame: np.ndarray, margin: float = 0.3) -> list[tuple[float, float, float, float]]:
        big = cv2.resize(frame, None, fx=self.upscale, fy=self.upscale, interpolation=cv2.INTER_LINEAR)
        self._detector.setInputSize((big.shape[1], big.shape[0]))
        _, rows = self._detector.detect(big)
        boxes = []
        for r in rows if rows is not None else []:
            x, y, w, h = (v / self.upscale for v in r[:4])
            boxes.append((x - margin * w, y - margin * h, x + w * (1 + margin), y + h * (1 + margin)))
        return boxes


def outside_all(box: tuple[float, float, float, float], keep: list[tuple[float, float, float, float]]) -> bool:
    """True unless the box's centre lies inside one of the `keep` boxes (identified students' heads)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return not any(k[0] <= cx <= k[2] and k[1] <= cy <= k[3] for k in keep)


def save_identities(path: Path, identities: dict[str, SeatIdentity]) -> None:
    path.write_text(json.dumps({s: asdict(i) for s, i in identities.items()}, indent=2))


def load_seating_chart(path: Optional[Path], registry: Optional[ConsentRegistry] = None) -> dict[str, str]:
    """CSV with seat_id,student_id. Unknown students are ignored."""
    if path is None or not path.exists():
        return {}
    import csv
    with path.open(newline="", encoding="utf-8") as f:
        rows = {r["seat_id"].strip(): r["student_id"].strip() for r in csv.DictReader(f) if r.get("seat_id")}
    if registry is not None:
        rows = {s: sid for s, sid in rows.items() if registry.get(sid) is not None}
    return rows
