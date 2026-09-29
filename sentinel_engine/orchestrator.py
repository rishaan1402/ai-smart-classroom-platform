"""
SentinelCore — main orchestrator.

Wires all components together:
- Loads config, detects device
- Cleans stale shared memory segments
- Initializes shared YOLO model
- Pre-warms Ollama before starting cameras
- Starts all worker threads
- Handles graceful shutdown (SIGTERM/SIGINT)

Graceful shutdown order:
  1. stop_event → CameraWorkers stop capturing
  2. Drain Redis Streams (drain sentinel published)
  3. Wait ≤5s for LLMWorkers
  4. Wait ≤10s for VLMWorkers
  5. Save ring buffer checkpoints
  6. Wait ≤5s for EventLogger
  7. Release cameras, sys.exit(0)
"""

import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import redis as redis_lib
import requests

from .config import load_config, validate_hardware
from .credentials import CredentialStore
from .device import detect_device
from .gatekeeper import Gatekeeper
from .scene_calibrator import SceneCalibrator
from .shm_cleanup import clean_stale_shm
from .storage import create_storage
from .tokenizer import Tokenizer
from .workers.camera import CameraWorker, WatchdogThread, init_yolo
from .workers.canary import CanaryChecker
from .workers.clip_encoder import ClipEncoderPool
from .workers.face import FaceIndex, FaceWorker
from .workers.llm import LLMWorker, _warmup_ollama
from .workers.logger import EventLogger
from .workers.notifier import NotificationDeduplicator
from .workers.stream_trimmer import StreamTrimmer
from .workers.vlm import VLMWorker

logger = logging.getLogger(__name__)


class SentinelCore:
    """
    Main orchestrator for the Sentinel AI surveillance engine.

    Usage:
        core = SentinelCore(config_path="config/app_config.yaml")
        core.run()  # blocks until SIGTERM/SIGINT
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._cfg = load_config(config_path)
        self._device = detect_device()
        self._stop_event = False

        # Components initialized in run()
        self._redis: Optional[redis_lib.Redis] = None
        self._camera_workers: dict[str, CameraWorker] = {}
        self._watchdog: Optional[WatchdogThread] = None
        self._llm_workers: list[LLMWorker] = []
        self._vlm_workers: list[VLMWorker] = []
        self._face_workers: list[FaceWorker] = []
        self._event_loggers: list[EventLogger] = []
        self._stream_trimmer: Optional[StreamTrimmer] = None
        self._clip_pool: Optional[ClipEncoderPool] = None
        self._canary: Optional[CanaryChecker] = None
        self._clip_queues: dict[str, queue.Queue] = {}

    def run(self, cameras: Optional[list[dict]] = None, zones: Optional[list[dict]] = None) -> None:
        """
        Start the engine. Blocks until shutdown signal received.

        Args:
            cameras: List of camera dicts with id, rtsp_url, etc.
                     If None, loads from PostgreSQL or env config.
            zones: List of zone dicts. If None, loads from zones.json.
        """
        self._setup_signals()

        logger.info(f"SentinelCore starting on device: {self._device}")

        # 1. Clean up any orphaned shared memory from previous crashed runs
        clean_stale_shm()

        # 2. Connect to Redis
        self._redis = self._connect_redis()

        # 3. Load zones
        if zones is None:
            zones = self._load_zones()

        # 4. Load cameras
        if cameras is None:
            cameras = self._load_cameras()

        if not cameras:
            logger.error("No cameras configured — exiting")
            return

        # 5. Validate hardware
        target_fps = int(self._cfg.get("target_fps", 15))
        validate_hardware(self._device, len(cameras), target_fps)

        # 6. Initialize shared YOLO model
        model_path = self._cfg.get("yolo_model", "yolo11n-pose.pt")
        init_yolo(model_path, self._device)

        # 7. Pre-warm Ollama before cameras start
        ollama_url = self._cfg.get("ollama_url", "http://localhost:11434")
        llm_model = self._cfg.get("llm_model", "qwen3:8b")
        if not _warmup_ollama(ollama_url, llm_model):
            logger.warning("Ollama warmup failed — LLM triage will have cold-start latency")

        # 8. Initialize storage
        storage = create_storage(self._cfg)

        # 9. ClipEncoderPool + resume checkpoints
        self._clip_pool = ClipEncoderPool(storage, self._cfg)
        self._clip_pool.resume_from_checkpoint()

        # 10. Start StreamTrimmer
        self._stream_trimmer = StreamTrimmer(self._redis, self._cfg)
        self._stream_trimmer.start()

        # 11. Start Redis Stream workers
        self._start_workers(storage)

        # 12. Start CameraWorkers
        self._start_cameras(cameras, zones)

        # 13. Start WatchdogThread
        self._watchdog = WatchdogThread(self._camera_workers)
        self._watchdog.start()

        # 14. Start CanaryChecker
        canary_frame = self._cfg.get("canary_frame_path", "tests/fixtures/canary_frame.jpg")
        if self._redis and Path(canary_frame).exists():
            self._canary = CanaryChecker(
                camera_ids=list(self._camera_workers.keys()),
                canary_frame_path=canary_frame,
                redis_client=self._redis,
                cfg=self._cfg,
            )
            self._canary.start()

        logger.info(f"SentinelCore running: {len(cameras)} cameras, device={self._device}")
        logger.info("Press Ctrl+C or send SIGTERM to stop")

        # Block until shutdown
        try:
            while not self._stop_event:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

        self._shutdown()

    def _start_workers(self, storage) -> None:
        """Start Redis Stream consumer workers."""
        # LLMWorkers
        n_llm = int(self._cfg.get("num_llm_workers", 2))
        for i in range(n_llm):
            w = LLMWorker(self._redis, self._cfg, worker_index=i)
            w.start()
            self._llm_workers.append(w)

        # VLMWorkers
        n_vlm = int(self._cfg.get("num_vlm_workers", 1))
        for i in range(n_vlm):
            w = VLMWorker(self._redis, self._cfg, worker_index=i)
            w.start()
            self._vlm_workers.append(w)

        # FaceWorkers (if enabled)
        if self._cfg.get("face_enabled", False):
            face_index = FaceIndex()
            face_index.initialize()
            n_face = int(self._cfg.get("num_face_workers", 1))
            for i in range(n_face):
                w = FaceWorker(self._redis, self._cfg, face_index=face_index, worker_index=i)
                w.initialize_insightface()
                w.start()
                self._face_workers.append(w)

        # EventLoggers — consume sentinel:events directly when no LLM workers
        n_log = int(self._cfg.get("num_log_workers", 1))
        tenant_id = self._cfg.get("tenant_id", "default")
        log_stream = (
            f"sentinel:events:{tenant_id}" if n_llm == 0
            else f"sentinel:log:{tenant_id}"
        )
        for i in range(n_log):
            w = EventLogger(self._redis, self._cfg, storage=None, worker_index=i,
                            stream_key=log_stream)
            w.start()
            self._event_loggers.append(w)

        logger.info(
            f"Workers started: {n_llm} LLM, {n_vlm} VLM, "
            f"{len(self._face_workers)} Face, {n_log} Logger"
        )

    def _run_scene_calibration(self, cameras: list[dict], cap_map: dict) -> None:
        """
        Run VLM scene calibration for each camera and merge overrides into cfg.
        cap_map: {camera_id: cv2.VideoCapture} — pre-opened captures (main thread).
        Skipped if ollama_url not reachable or calibration disabled in config.
        """
        if not self._cfg.get("scene_calibration_enabled", True):
            logger.info("Scene calibration disabled in config")
            return

        calibrator = SceneCalibrator(self._cfg)
        for cam in cameras:
            cam_id = cam["id"]
            cap = cap_map.get(cam_id)
            if cap is None or not cap.isOpened():
                logger.warning(f"[{cam_id}] Scene calibration skipped — camera not open")
                continue

            overrides = calibrator.calibrate(cam_id, cap)
            if overrides:
                # Merge into cfg (strip internal _keys before logging)
                visible = {k: v for k, v in overrides.items() if not k.startswith("_")}
                logger.info(f"[{cam_id}] Applying calibration overrides: {visible}")
                self._cfg.update(overrides)

    def _preopen_webcam_capture(self, rtsp_url: str) -> Optional[Any]:
        """
        Open webcam capture on the main thread.
        macOS AVFoundation requires camera access from the main thread — if we open it
        here (before spawning the CameraWorker thread) the OS permission is satisfied.
        Returns the open cv2.VideoCapture, or None if not a webcam URL.
        """
        import cv2
        is_webcam = rtsp_url.startswith("webcam://")
        if not is_webcam:
            try:
                int(rtsp_url)
                is_webcam = True
            except ValueError:
                pass
        if not is_webcam:
            return None

        index = int(rtsp_url.replace("webcam://", "") or "0") if rtsp_url.startswith("webcam://") else int(rtsp_url)
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            logger.info(f"Pre-opened webcam {index} on main thread (macOS AVFoundation)")
            return cap
        logger.warning(f"Could not pre-open webcam {index} on main thread")
        return None

    def _start_cameras(self, cameras: list[dict], zones: list[dict]) -> None:
        """Start per-camera workers."""
        tenant_id = self._cfg.get("tenant_id", "default")

        # Pre-open all webcam captures on the main thread (macOS AVFoundation requirement)
        # and collect them for scene calibration before threads start.
        preopen_caps: dict[str, Any] = {}
        for cam in cameras:
            cap = self._preopen_webcam_capture(cam.get("rtsp_url", ""))
            if cap is not None:
                preopen_caps[cam["id"]] = cap

        # Run scene calibration using the pre-opened captures (updates self._cfg)
        if preopen_caps:
            self._run_scene_calibration(cameras, preopen_caps)

        for cam in cameras:
            cam_id = cam["id"]
            rtsp_url = cam.get("rtsp_url", "")

            if not rtsp_url:
                logger.warning(f"Camera {cam_id} has no RTSP URL — skipping")
                continue

            # Use the already pre-opened capture (from pre-open + calibration phase above)
            preopen_cap = preopen_caps.get(cam_id)

            clip_q: queue.Queue = queue.Queue(maxsize=3)
            self._clip_queues[cam_id] = clip_q

            gatekeeper = Gatekeeper(self._cfg, redis_client=self._redis)
            gatekeeper.load_zone_rules(zones)

            tokenizer = Tokenizer(camera_id=cam_id, tenant_id=tenant_id)
            tokenizer.set_zones(zones, 1920, 1080)  # Default; updated on first frame

            worker = CameraWorker(
                camera_id=cam_id,
                rtsp_url=rtsp_url,
                cfg=self._cfg,
                gatekeeper=gatekeeper,
                tokenizer=tokenizer,
                redis_client=self._redis,
                clip_queue=clip_q,
                device=self._device,
                preopen_cap=preopen_cap,
            )
            worker.set_zones(zones)
            worker.start()
            self._camera_workers[cam_id] = worker

            # Clip queue consumer thread
            clip_consumer = threading.Thread(
                target=self._consume_clip_queue,
                args=(cam_id, clip_q),
                daemon=True,
                name=f"ClipConsumer-{cam_id}",
            )
            clip_consumer.start()

        logger.info(f"Started {len(self._camera_workers)} camera workers")

    def _consume_clip_queue(self, camera_id: str, clip_q: queue.Queue) -> None:
        """Forward clip jobs from camera queue to ClipEncoderPool."""
        while True:
            try:
                job = clip_q.get(timeout=1.0)
                if self._clip_pool:
                    self._clip_pool.submit(job["event"], job["frames"])
            except queue.Empty:
                if self._stop_event:
                    break
            except Exception as e:
                logger.error(f"ClipConsumer-{camera_id} error: {e}")

    def _shutdown(self) -> None:
        """Ordered graceful shutdown — 20 second budget."""
        logger.info("Graceful shutdown initiated")
        self._stop_event = True

        # 1. Stop CameraWorkers
        for worker in self._camera_workers.values():
            worker.stop()
        logger.info("CameraWorkers stopped")

        # 2. Flush ring buffers to checkpoint before workers drain
        self._flush_checkpoints()

        # 3. Wait for LLMWorkers (≤5s)
        deadline = time.time() + 5
        for w in self._llm_workers:
            remaining = max(0, deadline - time.time())
            w.stop()
            w.join(timeout=remaining)

        # 4. Wait for VLMWorkers (≤10s)
        deadline = time.time() + 10
        for w in self._vlm_workers:
            remaining = max(0, deadline - time.time())
            w.stop()
            w.join(timeout=remaining)

        # 5. Wait for FaceWorkers + EventLoggers (≤5s)
        deadline = time.time() + 5
        for w in self._face_workers + self._event_loggers:
            remaining = max(0, deadline - time.time())
            w.stop()
            w.join(timeout=remaining)

        # 6. Stop ancillary threads
        if self._stream_trimmer:
            self._stream_trimmer.stop()
        if self._watchdog:
            self._watchdog.stop()
        if self._canary:
            self._canary.stop()

        logger.info("SentinelCore shutdown complete")
        sys.exit(0)

    def _flush_checkpoints(self) -> None:
        """Save in-progress ring buffers to disk for ClipEncoder recovery on next start."""
        if self._clip_pool is None:
            return
        for cam_id, worker in self._camera_workers.items():
            # Check if there's a pending clip (from clip queue)
            clip_q = self._clip_queues.get(cam_id)
            if clip_q:
                try:
                    job = clip_q.get_nowait()
                    self._clip_pool.save_checkpoint(job["event"], job["frames"])
                except queue.Empty:
                    pass

    def _connect_redis(self) -> redis_lib.Redis:
        # Config file takes priority; fall back to REDIS_PASSWORD env var
        password = (
            self._cfg.get("redis_password")
            or os.environ.get("REDIS_PASSWORD")
            or None
        )
        r = redis_lib.Redis(
            host=self._cfg.get("redis_host") or os.environ.get("REDIS_HOST", "localhost"),
            port=int(self._cfg.get("redis_port", 6379)),
            password=password,
            db=int(self._cfg.get("redis_db", 0)),
            decode_responses=False,
        )
        try:
            r.ping()
            logger.info("Redis connected")
        except redis_lib.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise
        return r

    def _load_zones(self) -> list[dict]:
        zones_path = Path(self._cfg.get("zones_config", "config/zones.json"))
        if zones_path.exists():
            with open(zones_path) as f:
                import json
                return json.load(f)
        return []

    def _load_cameras(self) -> list[dict]:
        """Load cameras from config file (MVP) or PostgreSQL."""
        cameras_path = Path(self._cfg.get("cameras_config", "config/cameras.json"))
        if cameras_path.exists():
            with open(cameras_path) as f:
                import json
                return json.load(f)
        return []

    def _setup_signals(self) -> None:
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        logger.info(f"Received signal {signum}")
        self._stop_event = True
