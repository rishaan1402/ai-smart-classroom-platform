"""
Face detection + embedding with commercially licensed models.

OpenCV Zoo YuNet (detector, MIT) + SFace (recognizer, Apache-2.0), both built
into OpenCV. SentinelCore's InsightFace buffalo_l weights are licensed for
non-commercial research only, so they can't ship in the product. Anything
implementing the FaceEmbedder protocol can be swapped in.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

YUNET_PATH = Path("models/face_detection_yunet_2023mar.onnx")
SFACE_PATH = Path("models/face_recognition_sface_2021dec.onnx")

# OpenCV's published SFace cosine threshold for same-identity at ~LFW accuracy
SFACE_MATCH_THRESHOLD = 0.363


@dataclass
class Face:
    bbox: tuple[float, float, float, float]   # x, y, w, h in the image passed to detect()
    landmarks: np.ndarray                     # (5, 2): eyes, nose, mouth corners
    score: float
    raw: np.ndarray                           # detector row, needed for alignment

    @property
    def size(self) -> float:
        return min(self.bbox[2], self.bbox[3])

    @property
    def frontalness(self) -> float:
        """1.0 when the nose sits midway between the eyes, → 0 as the head turns."""
        (lx, _), (rx, _), (nx, _) = self.landmarks[0], self.landmarks[1], self.landmarks[2]
        span = abs(rx - lx)
        if span < 1e-6:
            return 0.0
        return float(max(0.0, 1.0 - abs((nx - (lx + rx) / 2) / span) * 2))


class FaceEmbedder(Protocol):
    dim: int

    def detect(self, image: np.ndarray) -> list[Face]: ...

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray: ...


class OpenCVFaceEmbedder:
    dim = 128

    def __init__(self, score_threshold: float = 0.7) -> None:
        if not YUNET_PATH.exists() or not SFACE_PATH.exists():
            raise FileNotFoundError(f"Face models missing: {YUNET_PATH}, {SFACE_PATH} (see docs/step4_status.md)")
        self._detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), score_threshold)
        self._recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")

    def detect(self, image: np.ndarray) -> list[Face]:
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        _, rows = self._detector.detect(image)
        if rows is None:
            return []
        return [
            Face(bbox=tuple(r[:4]), landmarks=r[4:14].reshape(5, 2), score=float(r[14]), raw=r)
            for r in rows
        ]

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        aligned = self._recognizer.alignCrop(image, face.raw)
        feat = self._recognizer.feature(aligned).reshape(-1).astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-12)
