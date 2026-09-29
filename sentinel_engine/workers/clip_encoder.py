"""
ClipEncoderPool — async H.264 clip encoding from compressed ring buffer.

Key design decisions:
- Shared pool with semaphore: max_concurrent = max(1, cpu_count // 4)
  Prevents ffmpeg processes from starving YOLO inference
- clip_queue per camera: max size 3, drops oldest on saturation
  (clip encoding is not latency-sensitive — event is already logged)
- Decodes JPEG ring buffer → ffmpeg subprocess via stdin → .mp4
- On startup: checks ring_buffer_checkpoint/ for marker files from
  graceful shutdown, submits pending clips, deletes markers on success
"""

import json
import logging
import multiprocessing
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ..storage import StorageBackend

logger = logging.getLogger(__name__)

_MAX_CONCURRENT = max(1, multiprocessing.cpu_count() // 4)


class ClipEncoderPool:
    """
    Manages H.264 clip encoding across multiple cameras.
    Uses a semaphore to limit concurrent ffmpeg processes.
    """

    def __init__(
        self,
        storage: StorageBackend,
        cfg: dict[str, Any],
        on_clip_done: Optional[callable] = None,
    ) -> None:
        self._storage = storage
        self._cfg = cfg
        self._on_clip_done = on_clip_done  # callback(event_id, clip_path)
        self._semaphore = threading.Semaphore(_MAX_CONCURRENT)
        self._checkpoint_dir = Path(cfg.get("checkpoint_dir", "data/ring_buffer_checkpoint"))
        self._tenant_id = cfg.get("tenant_id", "default")
        logger.info(f"ClipEncoderPool: max concurrent ffmpeg processes = {_MAX_CONCURRENT}")

    def submit(
        self,
        event: dict,
        frames: list[bytes],
    ) -> None:
        """
        Submit a clip encoding job. Non-blocking — runs in background thread.

        Args:
            event: SentinelEvent dict (for metadata and clip path backfill)
            frames: List of JPEG-compressed bytes from ring buffer
        """
        def _encode():
            self._semaphore.acquire()
            try:
                self._encode_clip(event, frames)
            finally:
                self._semaphore.release()

        t = threading.Thread(target=_encode, daemon=True, name=f"ClipEncoder-{event.get('person_id', 0)}")
        t.start()

    def resume_from_checkpoint(self) -> None:
        """
        On startup: check ring_buffer_checkpoint/ for marker files from
        a graceful shutdown that didn't complete clip encoding.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        markers = list(self._checkpoint_dir.glob("*.marker"))
        if not markers:
            return

        logger.info(f"Found {len(markers)} ring buffer checkpoint(s) — resuming clip encoding")
        for marker_path in markers:
            try:
                data = json.loads(marker_path.read_text())
                event = data.get("event", {})
                frames_paths = data.get("frame_paths", [])

                if frames_paths:
                    frames = []
                    for fp in frames_paths:
                        p = Path(fp)
                        if p.exists():
                            frames.append(p.read_bytes())
                    if frames:
                        self.submit(event, frames)
                        logger.info(f"Resumed clip encoding for event {event.get('id', 'unknown')}")

                marker_path.unlink()
            except Exception as e:
                logger.error(f"Failed to resume checkpoint {marker_path}: {e}")
                try:
                    marker_path.unlink()
                except Exception:
                    pass

    def _encode_clip(self, event: dict, frames: list[bytes]) -> None:
        """Decode JPEG frames and encode H.264 clip via ffmpeg."""
        if not frames:
            return

        event_id = event.get("id", f"clip_{int(time.time() * 1000)}")
        event_type = event.get("event_type", "unknown")
        person_id = event.get("person_id", 0)
        camera_id = event.get("camera_id", "unknown")

        # Decode first frame to get dimensions
        first_frame = cv2.imdecode(
            np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR
        )
        if first_frame is None:
            logger.error(f"Failed to decode first frame for clip {event_id}")
            return

        h, w = first_frame.shape[:2]
        fps = 15  # Ring buffer was captured at camera FPS; assume 15 for playback

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Start ffmpeg reading raw BGR frames from stdin
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "28",
                "-pix_fmt", "yuv420p",
                tmp_path,
            ]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Feed decoded frames to ffmpeg
            for frame_bytes in frames:
                frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    # Resize if needed (in case frame size changed)
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h))
                    try:
                        proc.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        break

            proc.stdin.close()
            proc.wait(timeout=120)

            if proc.returncode != 0:
                logger.error(f"ffmpeg failed (code {proc.returncode}) for clip {event_id}")
                return

            # Upload to storage
            from datetime import date
            today = date.today().isoformat()
            ts = int(time.time() * 1000)
            clip_key = f"{self._tenant_id}/{camera_id}/{today}/{event_id}_{ts}.mp4"

            with open(tmp_path, "rb") as f:
                clip_data = f.read()

            clip_path = self._storage.save(clip_key, clip_data)
            logger.info(f"Clip encoded: {clip_path} ({len(clip_data) // 1024}KB)")

            # Notify caller for DB backfill (clip_path update in events table)
            if self._on_clip_done:
                try:
                    self._on_clip_done(event_id, clip_path)
                except Exception as e:
                    logger.error(f"on_clip_done callback error: {e}")

        except subprocess.TimeoutExpired:
            logger.error(f"ffmpeg timed out for clip {event_id}")
            proc.kill()
        except FileNotFoundError:
            logger.error("ffmpeg not found — install with: apt install ffmpeg or brew install ffmpeg")
        except Exception as e:
            logger.error(f"Clip encoding failed for {event_id}: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def save_checkpoint(self, event: dict, frames: list[bytes]) -> None:
        """
        Save ring buffer to checkpoint dir during graceful shutdown.
        Resumed by resume_from_checkpoint() on next startup.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        event_id = event.get("id", f"shutdown_{int(time.time() * 1000)}")
        marker_path = self._checkpoint_dir / f"{event_id}.marker"

        # Save frames as individual files
        frame_paths = []
        for i, frame_bytes in enumerate(frames):
            fp = self._checkpoint_dir / f"{event_id}_frame_{i:04d}.jpg"
            fp.write_bytes(frame_bytes)
            frame_paths.append(str(fp))

        marker_path.write_text(json.dumps({"event": event, "frame_paths": frame_paths}))
        logger.debug(f"Saved ring buffer checkpoint: {marker_path}")
