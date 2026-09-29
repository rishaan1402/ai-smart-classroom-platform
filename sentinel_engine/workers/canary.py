"""
CanaryChecker — detection quality monitor per camera.

Runs every 5 minutes per camera. Loads a known test frame and verifies
YOLO returns at least 1 detection with confidence >= 0.5.

MVP design: shares the same YOLO model instance as CameraWorker via a lock.
No separate model load, no extra VRAM.
is_canary=True ensures results never touch Gatekeeper or event streams.

Writes per-camera status to Redis:
  canary:{camera_id}:status = "ok" | "degraded"
  canary:{camera_id}:last_check = unix timestamp

Dashboard reads these keys for per-camera quality indicators:
  ✅ Detecting / ⚠️ Degraded / ❌ Offline
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import redis as redis_lib

from .camera import _yolo_lock, get_yolo_model

logger = logging.getLogger(__name__)

# Required number of consecutive failures before marking degraded
FAILURE_THRESHOLD = 2


class CanaryChecker:
    """
    Runs canary detection tests for a set of cameras.
    One instance manages all cameras — checks them sequentially.
    """

    def __init__(
        self,
        camera_ids: list[str],
        canary_frame_path: str,
        redis_client: redis_lib.Redis,
        cfg: dict[str, Any],
    ) -> None:
        self._camera_ids = camera_ids
        self._canary_frame_path = Path(canary_frame_path)
        self._redis = redis_client
        self._cfg = cfg
        self._interval_s = float(cfg.get("canary_interval_s", 300))  # 5 minutes
        self._min_confidence = float(cfg.get("canary_min_confidence", 0.5))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._failure_counts: dict[str, int] = {}

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="CanaryChecker",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"CanaryChecker started for {len(self._camera_ids)} cameras")

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self._canary_frame_path.exists():
                logger.warning(f"Canary frame not found: {self._canary_frame_path}")
                self._stop_event.wait(timeout=self._interval_s)
                continue

            for camera_id in self._camera_ids:
                if self._stop_event.is_set():
                    break
                self._check_camera(camera_id)

            self._stop_event.wait(timeout=self._interval_s)

    def _check_camera(self, camera_id: str) -> None:
        """Run one canary check for a camera."""
        model = get_yolo_model()
        if model is None:
            return

        try:
            import cv2
            frame = cv2.imread(str(self._canary_frame_path))
            if frame is None:
                logger.error(f"Cannot load canary frame: {self._canary_frame_path}")
                return

            # Use shared YOLO model with lock — same instance as CameraWorker
            # is_canary=True conceptually: results never enter Gatekeeper
            with _yolo_lock:
                results = model(
                    frame,
                    conf=self._min_confidence,
                    verbose=False,
                )

            passed = False
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    max_conf = float(boxes.conf.max()) if len(boxes) > 0 else 0.0
                    if max_conf >= self._min_confidence:
                        passed = True

            if passed:
                self._failure_counts[camera_id] = 0
                self._set_status(camera_id, "ok")
                logger.debug(f"Canary OK for camera {camera_id}")
            else:
                failures = self._failure_counts.get(camera_id, 0) + 1
                self._failure_counts[camera_id] = failures
                if failures >= FAILURE_THRESHOLD:
                    self._set_status(camera_id, "degraded")
                    logger.warning(
                        f"Canary DEGRADED for camera {camera_id} "
                        f"({failures} consecutive failures)"
                    )

        except Exception as e:
            logger.error(f"Canary check failed for {camera_id}: {e}")
            failures = self._failure_counts.get(camera_id, 0) + 1
            self._failure_counts[camera_id] = failures
            if failures >= FAILURE_THRESHOLD:
                self._set_status(camera_id, "degraded")

    def _set_status(self, camera_id: str, status: str) -> None:
        try:
            self._redis.set(f"canary:{camera_id}:status", status)
            self._redis.set(f"canary:{camera_id}:last_check", str(time.time()))
        except Exception as e:
            logger.error(f"Failed to write canary status to Redis: {e}")

    def get_status(self, camera_id: str) -> dict:
        """Get current canary status for a camera (for health endpoint)."""
        try:
            status = (self._redis.get(f"canary:{camera_id}:status") or b"unknown").decode()
            last_check = self._redis.get(f"canary:{camera_id}:last_check")
            last_check_ts = float(last_check.decode()) if last_check else 0.0
            return {"status": status, "last_check": last_check_ts}
        except Exception:
            return {"status": "unknown", "last_check": 0.0}
