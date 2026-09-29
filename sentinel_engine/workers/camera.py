"""
CameraWorker — per-camera RTSP capture + YOLO inference + tokenize + gate.

Key design decisions:
- RTSP timeout via OPENCV_FFMPEG_CAPTURE_OPTIONS (stimeout=5s in microseconds)
- cap.read() in daemon thread with threading.Timer watchdog (prevents indefinite block)
- Compressed ring buffer: JPEG bytes (~90MB/cam vs ~5.4GB raw numpy)
- Reads camera config from PostgreSQL on startup; hot-reloads via Redis pub/sub
- Backpressure: skips every 2nd frame when Redis Stream is > 40000 entries deep
- WatchdogThread monitors heartbeat and marks camera offline

Bug fixes from existing code:
- Tracker fallback IDs: np.arange(n) * -1 - 1  (not * -1, which gives 0)
- botsort_custom.yaml used consistently via config (not hardcoded "botsort.yaml")
"""

import logging
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

import cv2
import numpy as np
import redis as redis_lib

from ..config import validate_hardware
from ..gatekeeper import Gatekeeper
from ..tokenizer import Tokenizer

logger = logging.getLogger(__name__)

# Shared YOLO model + lock for MVP (no InferencePool yet)
_yolo_model = None
_yolo_lock = threading.Lock()


def init_yolo(model_path: str, device: str) -> Any:
    """Initialize and return the shared YOLO model (call once from orchestrator)."""
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            from ultralytics import YOLO
            _yolo_model = YOLO(model_path)
            logger.info(f"YOLO model loaded: {model_path} on {device}")
    return _yolo_model


def get_yolo_model() -> Any:
    return _yolo_model


class _FrameReader:
    """
    Reads frames from cv2.VideoCapture in a daemon thread.
    Uses a threading.Timer watchdog to break out of stuck FFmpeg reads.
    """

    def __init__(self, cap: cv2.VideoCapture, timeout_s: float = 10.0) -> None:
        self._cap = cap
        self._timeout_s = timeout_s
        self._result: Optional[tuple[bool, Any]] = None
        self._done = threading.Event()

    def read(self) -> tuple[bool, Any]:
        self._result = None
        self._done.clear()

        def _read_fn():
            try:
                self._result = self._cap.read()
            except Exception:
                self._result = (False, None)
            finally:
                self._done.set()

        def _watchdog():
            if not self._done.is_set():
                logger.warning("cap.read() timed out — releasing stuck capture")
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._done.set()

        t = threading.Thread(target=_read_fn, daemon=True)
        watchdog = threading.Timer(self._timeout_s, _watchdog)
        t.start()
        watchdog.start()
        self._done.wait()
        watchdog.cancel()

        if self._result is None:
            return False, None
        return self._result


class CameraWorker(threading.Thread):
    """
    Per-camera worker thread.

    Responsibilities:
    1. Open RTSP stream with TCP timeout
    2. Run YOLO inference (shared model, locked)
    3. Tokenize detections → BehaviorTokens
    4. Evaluate via Gatekeeper → SentinelEvents → Redis Streams
    5. Maintain compressed JPEG ring buffer for clip encoding
    6. Reconnect with exponential backoff on stream failure
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        cfg: dict[str, Any],
        gatekeeper: Gatekeeper,
        tokenizer: Tokenizer,
        redis_client: Optional[redis_lib.Redis] = None,
        clip_queue: Optional[queue.Queue] = None,
        device: str = "cpu",
        preopen_cap: Optional[cv2.VideoCapture] = None,
    ) -> None:
        super().__init__(name=f"CameraWorker-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self._cfg = cfg
        self._gatekeeper = gatekeeper
        self._tokenizer = tokenizer
        self._redis = redis_client
        self._clip_queue = clip_queue
        self._device = device

        self._preopen_cap = preopen_cap  # Pre-opened on main thread (macOS webcam)
        self._stop_event = threading.Event()
        self._heartbeat = time.time()

        # Compressed ring buffer: deque of JPEG bytes
        ring_size = int(cfg.get("ring_buffer_frames", 900))
        self._ring_buffer: deque[bytes] = deque(maxlen=ring_size)
        self._jpeg_quality = int(cfg.get("ring_buffer_jpeg_quality", 70))

        self._reconnect_base = float(cfg.get("camera_reconnect_base_s", 2.0))
        self._reconnect_max = float(cfg.get("camera_reconnect_max_s", 30.0))

        self._tenant_id = cfg.get("tenant_id", "default")
        self._frame_count = 0
        self._skip_next = False  # Backpressure flag

    def run(self) -> None:
        reconnect_delay = self._reconnect_base
        logger.info(f"CameraWorker {self.camera_id} starting")

        while not self._stop_event.is_set():
            # Use pre-opened capture on first iteration (macOS webcam opened on main thread)
            if self._preopen_cap is not None:
                cap = self._preopen_cap
                self._preopen_cap = None
            else:
                cap = self._open_capture()
            if cap is None:
                logger.warning(f"[{self.camera_id}] Failed to open stream — retrying in {reconnect_delay}s")
                self._stop_event.wait(timeout=reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, self._reconnect_max)
                continue

            reconnect_delay = self._reconnect_base  # Reset on successful connect
            logger.info(f"[{self.camera_id}] Stream opened")
            reader = _FrameReader(cap, timeout_s=10.0)

            # Read frame dimensions and update zones
            ret, frame = reader.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                # Zones come from gatekeeper zone_rules — tokenizer set_zones is
                # called by orchestrator with frame dimensions once we have them
                self._notify_frame_size(w, h)

            while not self._stop_event.is_set():
                ret, frame = reader.read()

                if not ret or frame is None:
                    logger.warning(f"[{self.camera_id}] Frame read failed — reconnecting")
                    break

                self._heartbeat = time.time()
                self._frame_count += 1

                # Backpressure: skip every other frame when stream depth is high
                if self._check_backpressure():
                    if self._frame_count % 2 == 0:
                        self._store_frame(frame)
                        continue

                # Store compressed frame in ring buffer
                self._store_frame(frame)

                # Run YOLO inference (shared model + lock)
                self._process_frame(frame)

            cap.release()
            if not self._stop_event.is_set():
                logger.info(f"[{self.camera_id}] Reconnecting in {reconnect_delay}s")
                self._stop_event.wait(timeout=reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, self._reconnect_max)

        logger.info(f"CameraWorker {self.camera_id} stopped")

    def stop(self) -> None:
        self._stop_event.set()

    def get_heartbeat(self) -> float:
        return self._heartbeat

    def snapshot_ring_buffer(self) -> list[bytes]:
        """Return a copy of the current ring buffer for clip encoding."""
        return list(self._ring_buffer)

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """
        Open video capture. Supports:
          - RTSP URLs: rtsp://...
          - Local files: file:///path/to/video.mp4
          - Webcam: webcam://0  or  0  or  1  (device index)
        """
        url = self.rtsp_url

        # Webcam detection: "webcam://0", "0", "1", etc.
        if url.startswith("webcam://"):
            device_index = int(url.replace("webcam://", "") or "0")
            return self._open_webcam(device_index)
        try:
            device_index = int(url)
            return self._open_webcam(device_index)
        except ValueError:
            pass

        # RTSP / file — use FFmpeg backend
        rtsp_timeout_us = int(self._cfg.get("rtsp_timeout_us", 5_000_000))
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;tcp|stimeout;{rtsp_timeout_us}"
        )
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                return None
            return cap
        except Exception as e:
            logger.error(f"[{self.camera_id}] cv2.VideoCapture failed: {e}")
            return None

    def _open_webcam(self, index: int) -> Optional[cv2.VideoCapture]:
        """Open a local webcam by device index (no FFmpeg)."""
        try:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                logger.error(f"[{self.camera_id}] Webcam index {index} not available")
                return None
            # Optional: cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # Optional: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            logger.info(f"[{self.camera_id}] Opened webcam device {index}")
            return cap
        except Exception as e:
            logger.error(f"[{self.camera_id}] Webcam open failed: {e}")
            return None

    def _store_frame(self, frame: np.ndarray) -> None:
        """Compress frame as JPEG and append to ring buffer."""
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if ok:
            self._ring_buffer.append(buf.tobytes())

    def _check_backpressure(self) -> bool:
        """Return True if Redis stream depth exceeds backpressure threshold."""
        if self._redis is None:
            return False
        try:
            depth = self._redis.xlen(f"sentinel:events:{self._tenant_id}")
            return depth > 40_000
        except Exception:
            return False

    def _process_frame(self, frame: np.ndarray) -> None:
        """Run YOLO, tokenize, gate, and emit events."""
        model = get_yolo_model()
        if model is None:
            return

        h, w = frame.shape[:2]

        with _yolo_lock:
            try:
                tracker_config = self._cfg.get("tracker_config", "botsort_custom.yaml")
                results = model.track(
                    frame,
                    persist=True,
                    tracker=tracker_config,
                    conf=float(self._cfg.get("yolo_confidence", 0.5)),
                    iou=float(self._cfg.get("yolo_iou", 0.45)),
                    verbose=False,
                )
            except Exception as e:
                logger.error(f"[{self.camera_id}] YOLO inference failed: {e}")
                return

        if not results or len(results) == 0:
            return

        result = results[0]
        self._tokenizer.set_zones(
            self._zones if hasattr(self, "_zones") else [],
            w, h
        )

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return

        # Extract track IDs — bug fix: use * -1 - 1 to avoid ID 0 collision
        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().numpy().tolist()
        else:
            # No tracker IDs (detection-only mode): assign negative IDs
            # Bug fix from existing code: was np.arange(n) * -1 → produces [0,-1,-2,...]
            # which collides with real track ID 0. Fix: * -1 - 1 → [-1,-2,-3,...]
            n = len(boxes)
            track_ids = (np.arange(n, dtype=np.int32) * -1 - 1).tolist()

        bboxes = boxes.xyxy.cpu().numpy().tolist()
        keypoints_all = []
        if result.keypoints is not None:
            kps_data = result.keypoints.data.cpu().numpy()
            keypoints_all = kps_data.tolist()
        else:
            keypoints_all = [[[0.0, 0.0, 0.0]] * 17] * len(bboxes)

        active_ids = set(track_ids)
        self._tokenizer.cleanup(active_ids)

        for track_id, bbox, kps in zip(track_ids, bboxes, keypoints_all):
            token = self._tokenizer.tokenize(
                person_id=int(track_id),
                bbox=bbox,
                keypoints=kps,
            )
            events = self._gatekeeper.evaluate(token)

            for event in events:
                # Attach JPEG snapshot to event metadata for VLM analysis
                try:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        import base64
                        event.metadata["image_b64"] = base64.b64encode(buf.tobytes()).decode()
                except Exception:
                    pass

                # Push ring buffer snapshot for clip encoding
                if self._clip_queue is not None:
                    try:
                        self._clip_queue.put_nowait({
                            "event": event.to_dict(),
                            "frames": self.snapshot_ring_buffer(),
                        })
                    except queue.Full:
                        logger.warning(f"[{self.camera_id}] Clip queue full — dropping clip for {event.event_type.value}")

    def _notify_frame_size(self, w: int, h: int) -> None:
        """Called once on first frame to set frame dimensions for zone scaling."""
        # Orchestrator sets zones via tokenizer.set_zones externally.
        # This is a hook for the orchestrator to update zones on dimension change.
        pass

    def set_zones(self, zones: list[dict]) -> None:
        """Set zone definitions (called by orchestrator)."""
        self._zones = zones


class WatchdogThread(threading.Thread):
    """
    Monitors CameraWorker heartbeats and marks cameras as offline.
    """

    def __init__(
        self,
        workers: dict[str, CameraWorker],
        timeout_s: float = 30.0,
        on_camera_offline: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(name="WatchdogThread", daemon=True)
        self._workers = workers
        self._timeout_s = timeout_s
        self._on_offline = on_camera_offline
        self._stop_event = threading.Event()
        self._camera_status: dict[str, str] = {}

    def run(self) -> None:
        logger.info("WatchdogThread started")
        while not self._stop_event.is_set():
            now = time.time()
            for cam_id, worker in self._workers.items():
                age = now - worker.get_heartbeat()
                if age > self._timeout_s:
                    if self._camera_status.get(cam_id) != "offline":
                        logger.warning(f"Camera {cam_id} offline (no heartbeat for {age:.0f}s)")
                        self._camera_status[cam_id] = "offline"
                        if self._on_offline:
                            try:
                                self._on_offline(cam_id)
                            except Exception as e:
                                logger.error(f"on_camera_offline callback error: {e}")
                else:
                    if self._camera_status.get(cam_id) == "offline":
                        logger.info(f"Camera {cam_id} back online")
                    self._camera_status[cam_id] = "online"

            self._stop_event.wait(timeout=10.0)

    def stop(self) -> None:
        self._stop_event.set()

    def get_status(self) -> dict[str, str]:
        return dict(self._camera_status)
