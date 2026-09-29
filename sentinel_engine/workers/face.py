"""
FaceWorker — Redis Streams consumer for face identification.

Uses InsightFace (SCRFD + ArcFace) + FAISS HNSW for fast search.

Key design decisions:
- Read-write lock: search() takes shared lock, add()/rebuild() exclusive
  (simpler than copy-on-write, uses half the memory)
- _deleted_face_ids set: post-match filter for faces pending anonymization
  (populated by anonymize(), cleared after index rebuild)
- Identification cache: per-person-id cooldown before re-querying
- Minimum crop size: reject < 64×64 pixels (prevents garbage ArcFace embeddings)
- Face embeddings stored durably in PostgreSQL; FAISS is in-memory search only

Bug fixes from existing code (face_id.py):
- _identification_cache was defined but never read/written in identify()
- No minimum crop size guard
"""

import json
import logging
import threading
import time
from collections import deque
from typing import Any, Optional
from uuid import UUID

import numpy as np
import redis as redis_lib

from ._base_stream_worker import BaseStreamWorker

logger = logging.getLogger(__name__)


class _RWLock:
    """Simple readers-writer lock favouring readers."""

    def __init__(self) -> None:
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self) -> None:
        with self._read_ready:
            self._readers += 1

    def release_read(self) -> None:
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self) -> None:
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self) -> None:
        self._read_ready.release()


class FaceIndex:
    """
    Thread-safe FAISS HNSW index for face embedding search.

    Embeddings are stored in PostgreSQL for durability.
    FAISS is the in-memory search acceleration layer only.
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        self._dim = embedding_dim
        self._lock = _RWLock()
        self._index = None
        self._id_map: list[str] = []  # index position → face_id (UUID str)
        self._name_map: dict[str, str] = {}  # face_id → name
        self._deleted_face_ids: set[str] = set()
        self._initialized = False

    def initialize(self) -> None:
        try:
            import faiss
            self._index = faiss.IndexHNSWFlat(self._dim, 32)
            self._initialized = True
            logger.info(f"FAISS HNSW index initialized (dim={self._dim})")
        except ImportError:
            logger.warning("FAISS not installed — face search disabled. Install with: pip install faiss-cpu")

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._index is not None

    def add(self, face_id: str, name: str, embedding: np.ndarray) -> None:
        """Add a face embedding. Takes exclusive write lock."""
        if not self.is_ready:
            return
        self._lock.acquire_write()
        try:
            # Normalize like search() and rebuild_from_embeddings(): the L2 → cosine
            # conversion in search() is only valid for unit vectors
            vec = self._normalize(embedding).astype(np.float32).reshape(1, -1)
            self._index.add(vec)
            self._id_map.append(face_id)
            self._name_map[face_id] = name
        finally:
            self._lock.release_write()

    def search(
        self, embedding: np.ndarray, k: int = 1, threshold: float = 0.6
    ) -> Optional[tuple[str, str, float]]:
        """
        Search for closest matching face.

        Returns (face_id, name, similarity) or None if no match above threshold.
        Takes shared read lock (non-blocking with other searches).
        """
        if not self.is_ready or len(self._id_map) == 0:
            return None

        self._lock.acquire_read()
        try:
            vec = self._normalize(embedding).reshape(1, -1)
            distances, indices = self._index.search(vec, min(k, len(self._id_map)))
        finally:
            self._lock.release_read()

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            face_id = self._id_map[idx]
            # Skip deleted (anonymized) faces
            if face_id in self._deleted_face_ids:
                continue
            # FAISS IndexHNSWFlat uses L2 distance; convert to cosine similarity
            similarity = 1.0 - float(dist) / 2.0
            if similarity >= threshold:
                return face_id, self._name_map.get(face_id, "Unknown"), similarity

        return None

    def mark_deleted(self, face_id: str) -> None:
        """Mark a face as anonymized — excluded from search results immediately."""
        self._deleted_face_ids.add(face_id)
        logger.info(f"Face {face_id} marked as anonymized in search index")

    def rebuild_from_embeddings(
        self, embeddings: list[tuple[str, str, np.ndarray]]
    ) -> None:
        """
        Rebuild index from scratch in background thread.
        After rebuild, atomically swap index reference and clear deleted set.
        """
        if not self.is_ready:
            return

        def _rebuild():
            import faiss
            new_index = faiss.IndexHNSWFlat(self._dim, 32)
            new_id_map = []
            new_name_map = {}

            for face_id, name, emb in embeddings:
                if face_id in self._deleted_face_ids:
                    continue  # Skip deleted faces
                vec = self._normalize(emb).astype(np.float32).reshape(1, -1)
                new_index.add(vec)
                new_id_map.append(face_id)
                new_name_map[face_id] = name

            # Atomic swap under write lock
            self._lock.acquire_write()
            try:
                self._index = new_index
                self._id_map = new_id_map
                self._name_map = new_name_map
                self._deleted_face_ids.clear()
            finally:
                self._lock.release_write()
            logger.info(f"FAISS index rebuilt: {len(new_id_map)} faces")

        threading.Thread(target=_rebuild, daemon=True, name="FaissRebuild").start()

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm


class FaceWorker(BaseStreamWorker):
    """
    Consumes sentinel:face:{tenant_id} and performs face identification.

    For each event:
    1. Check identification cache (skip if recently checked)
    2. Extract face crop from event metadata
    3. Run InsightFace detection on crop
    4. Search FAISS index for match
    5. Update gatekeeper identity state via Redis pub/sub
    6. Forward enriched event to log stream
    """

    def __init__(
        self,
        redis_client: redis_lib.Redis,
        cfg: dict[str, Any],
        face_index: Optional[FaceIndex] = None,
        worker_index: int = 0,
    ) -> None:
        tenant_id = cfg.get("tenant_id", "default")
        super().__init__(
            name=f"FaceWorker-{worker_index}",
            stream_key=f"sentinel:face:{tenant_id}",
            group_name="face_workers",
            consumer_name=f"face_worker_{worker_index}",
            redis_client=redis_client,
            cfg=cfg,
            dlq_stream=f"sentinel:dlq:failed:{tenant_id}",
        )
        self._face_index = face_index or FaceIndex()
        self._log_stream = f"sentinel:log:{tenant_id}"
        self._similarity_threshold = float(cfg.get("face_similarity_threshold", 0.6))
        self._check_cooldown = float(cfg.get("face_check_cooldown_s", 5.0))
        self._min_crop = int(cfg.get("face_min_crop_size", 64))

        # Identification cache: person_id → (name, confidence, timestamp)
        # Bug fix: existing code defined _identification_cache but never used it
        self._id_cache: dict[int, tuple[str, float, float]] = {}
        self._app = None  # InsightFace FaceAnalysis

    def initialize_insightface(self) -> bool:
        """Load InsightFace model. Call from orchestrator before starting worker."""
        if not self._cfg.get("face_enabled", False):
            logger.info("Face recognition disabled (face_enabled=false)")
            return False
        try:
            from insightface.app import FaceAnalysis
            model_name = self._cfg.get("face_model", "buffalo_l")
            self._app = FaceAnalysis(
                name=model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info(f"InsightFace loaded: {model_name}")
            return True
        except ImportError:
            logger.error("InsightFace not installed: pip install insightface onnxruntime")
            return False
        except Exception as e:
            logger.error(f"InsightFace init failed: {e}")
            return False

    def _process_message(self, message_id: bytes, data: dict) -> None:
        event_json = data.get(b"event", b"{}").decode()
        trace_id = data.get(b"trace_id", b"unknown").decode()

        event = json.loads(event_json)
        token = event.get("token", {})
        person_id = token.get("person_id", -1)

        # Check cooldown cache
        now = time.time()
        cached = self._id_cache.get(person_id)
        if cached and (now - cached[2]) < self._check_cooldown:
            name, confidence, _ = cached
            event["face_name"] = name
            event["face_confidence"] = confidence
            event["face_recognition_was_active"] = True
        else:
            # Run face identification
            face_result = self._identify_from_event(event, trace_id)
            event.update(face_result)

            # Update cache
            if face_result.get("face_recognition_was_active"):
                self._id_cache[person_id] = (
                    face_result.get("face_name", "Unknown"),
                    face_result.get("face_confidence", 0.0),
                    now,
                )

        # Forward to log stream
        self._redis.xadd(
            self._log_stream,
            {"event": json.dumps(event), "trace_id": trace_id},
        )

    def _identify_from_event(self, event: dict, trace_id: str) -> dict:
        """Run InsightFace on the event's image crop."""
        result = {
            "face_recognition_was_active": False,
            "face_name": None,
            "face_confidence": 0.0,
        }

        if self._app is None:
            return result

        # Get image crop from event metadata
        img_b64 = event.get("metadata", {}).get("image_b64")
        if not img_b64:
            return result

        try:
            import base64
            import cv2
            img_bytes = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning(f"Failed to decode crop image: {e}", extra={"trace_id": trace_id})
            return result

        # Minimum crop size guard
        if frame is None or frame.shape[0] < self._min_crop or frame.shape[1] < self._min_crop:
            logger.debug(
                f"Crop too small ({frame.shape if frame is not None else 'None'}) — skipping",
                extra={"trace_id": trace_id},
            )
            return result

        result["face_recognition_was_active"] = True

        try:
            faces = self._app.get(frame)
        except Exception as e:
            logger.error(f"InsightFace inference failed: {e}", extra={"trace_id": trace_id})
            return result

        if not faces:
            result["face_name"] = "Unknown"
            return result

        # Use largest face (closest to camera)
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        if not self._face_index.is_ready:
            result["face_name"] = "Unknown"
            return result

        match = self._face_index.search(
            face.embedding, k=1, threshold=self._similarity_threshold
        )
        if match:
            face_id, name, similarity = match
            result["face_id"] = face_id
            result["face_name"] = name
            result["face_confidence"] = float(similarity)
        else:
            result["face_name"] = "Unknown"
            result["face_confidence"] = 0.0

        return result
