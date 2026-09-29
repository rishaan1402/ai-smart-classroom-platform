"""
SceneCalibrator — VLM-powered automatic YOLO/Gatekeeper calibration.

On engine startup (before detection workers begin), captures N frames from the
camera and sends them to the VLM to understand the scene context. The VLM
response is used to adjust detection thresholds for the specific environment.

Calibration is cached in data/calibration/{camera_id}.json so it only runs
once unless forced with --recalibrate or the cache is older than 24 hours.

Adjustments made:
  yolo_confidence      — higher in bright/sparse scenes, lower in dim/crowded
  fall_frames          — raised significantly if seated environment detected
  zone_intrusion_frames — tighter in sparse scenes, looser in crowded
  loitering_threshold_s — raised in busy environments
  cooldowns            — per-event-type adjustments based on scene density
"""

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

CALIBRATION_PROMPT = """\
You are calibrating a security camera AI system. Analyze this camera image \
and provide precise calibration parameters for the detection algorithms.

Respond ONLY with valid JSON — no prose, no markdown:
{
  "scene_type": "office|warehouse|outdoor|retail|corridor|parking|home",
  "typical_posture": "seated|standing|mixed",
  "lighting": "bright|moderate|dim",
  "crowding_expected": "sparse|moderate|dense",
  "seated_environment": true,
  "notes": "one sentence describing what you see",
  "calibration": {
    "yolo_confidence": 0.55,
    "fall_frames": 8,
    "zone_intrusion_frames": 5,
    "loitering_threshold_s": 90,
    "hands_raised_frames": 5,
    "cooldown_zone_intrusion": 60,
    "cooldown_fall": 30
  }
}

Rules for calibration values:
- yolo_confidence: 0.35 (dim/crowded) to 0.70 (bright/sparse)
- fall_frames: 15+ if seated environment (people sitting = not fallen),
              5 if standing environment
- zone_intrusion_frames: 3 (sparse) to 10 (crowded/busy)
- loitering_threshold_s: 30 (strict) to 300 (relaxed/busy area)
- cooldown_zone_intrusion: 30 (sparse) to 120 (crowded)
"""


class SceneCalibrator:
    """
    Captures frames from a camera, sends to VLM, and returns calibration
    adjustments as a dict that can be merged into the engine config.
    """

    CACHE_TTL_S = 86_400  # 24 hours

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._ollama_url = cfg.get("ollama_url", "http://localhost:11434")
        self._vlm_model = cfg.get("vlm_model", "llava-phi3:latest")
        self._timeout = int(cfg.get("vlm_timeout_s", 30))
        self._cache_dir = Path(cfg.get("calibration_cache_dir", "data/calibration"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def calibrate(self, camera_id: str, cap: cv2.VideoCapture) -> dict[str, Any]:
        """
        Run calibration for a camera. Returns merged config overrides.
        Uses cached result if fresh (< 24h). Returns empty dict on failure.
        """
        cache_path = self._cache_dir / f"{camera_id}.json"

        # Return cached result if still fresh
        if cache_path.exists():
            age_s = time.time() - cache_path.stat().st_mtime
            if age_s < self.CACHE_TTL_S:
                try:
                    cached = json.loads(cache_path.read_text())
                    logger.info(
                        f"[{camera_id}] SceneCalibrator: using cached calibration "
                        f"(age {age_s/3600:.1f}h) — {cached.get('notes', '')}"
                    )
                    return cached.get("overrides", {})
                except Exception:
                    pass

        logger.info(f"[{camera_id}] SceneCalibrator: capturing frames for VLM analysis…")
        frames = self._capture_frames(cap, n=5, interval_s=0.5)
        if not frames:
            logger.warning(f"[{camera_id}] SceneCalibrator: no frames captured — skipping calibration")
            return {}

        # Use the clearest frame (highest Laplacian variance = sharpest)
        best_frame = max(frames, key=lambda f: cv2.Laplacian(f, cv2.CV_64F).var())
        img_b64 = self._frame_to_b64(best_frame)

        result = self._query_vlm(img_b64, camera_id)
        if not result:
            return {}

        overrides = self._build_overrides(result)

        # Cache result
        try:
            cache_path.write_text(json.dumps({
                "camera_id": camera_id,
                "scene_type": result.get("scene_type"),
                "typical_posture": result.get("typical_posture"),
                "lighting": result.get("lighting"),
                "crowding_expected": result.get("crowding_expected"),
                "seated_environment": result.get("seated_environment"),
                "notes": result.get("notes", ""),
                "raw": result,
                "overrides": overrides,
                "calibrated_at": time.time(),
            }, indent=2))
        except Exception as e:
            logger.debug(f"[{camera_id}] SceneCalibrator: cache write failed: {e}")

        logger.info(
            f"[{camera_id}] SceneCalibrator: {result.get('scene_type')} scene, "
            f"{result.get('lighting')} lighting, "
            f"{'seated' if result.get('seated_environment') else 'standing'} posture. "
            f"Adjustments: {overrides}"
        )
        return overrides

    def _capture_frames(
        self, cap: cv2.VideoCapture, n: int = 5, interval_s: float = 0.5
    ) -> list[np.ndarray]:
        """Read N frames from the capture with interval spacing."""
        frames = []
        for _ in range(n):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
            time.sleep(interval_s)
        return frames

    def _frame_to_b64(self, frame: np.ndarray) -> str:
        """Convert BGR frame to base64-encoded JPEG for Ollama."""
        # Resize to max 640px wide to keep token count low
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
        rgb = frame[:, :, ::-1]
        pil_img = __import__("PIL.Image", fromlist=["Image"]).Image.fromarray(rgb.astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def _query_vlm(self, img_b64: str, camera_id: str) -> Optional[dict]:
        """Send frame to VLM and parse calibration JSON response."""
        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={
                    "model": self._vlm_model,
                    "prompt": CALIBRATION_PROMPT,
                    "images": [img_b64],
                    "stream": False,
                    "options": {"temperature": 0.05, "num_predict": 512},
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
            return self._parse_json(text)
        except requests.Timeout:
            logger.warning(f"[{camera_id}] SceneCalibrator: VLM timeout — using defaults")
        except Exception as e:
            logger.warning(f"[{camera_id}] SceneCalibrator: VLM error ({e}) — using defaults")
        return None

    def _parse_json(self, text: str) -> Optional[dict]:
        """Extract JSON from VLM response, handling extra prose."""
        try:
            decoder = json.JSONDecoder()
            start = text.find("{")
            if start != -1:
                obj, _ = decoder.raw_decode(text, start)
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _build_overrides(self, result: dict) -> dict[str, Any]:
        """
        Convert VLM calibration response into config overrides dict.
        VLM values are trusted but clamped to safe ranges.
        """
        cal = result.get("calibration", {})
        seated = result.get("seated_environment", False)

        overrides: dict[str, Any] = {}

        # YOLO confidence — clamp 0.30–0.75
        if "yolo_confidence" in cal:
            overrides["yolo_confidence"] = max(0.30, min(0.75, float(cal["yolo_confidence"])))

        # Fall detection — seated environments need much higher thresholds
        if seated:
            overrides["fall_frames"] = max(int(cal.get("fall_frames", 15)), 15)
        elif "fall_frames" in cal:
            overrides["fall_frames"] = max(3, min(15, int(cal["fall_frames"])))

        # Zone intrusion frames — clamp 3–15
        if "zone_intrusion_frames" in cal:
            overrides["zone_intrusion_frames"] = max(3, min(15, int(cal["zone_intrusion_frames"])))

        # Loitering threshold — clamp 30–600s
        if "loitering_threshold_s" in cal:
            overrides["loitering_threshold_s"] = max(30, min(600, int(cal["loitering_threshold_s"])))

        # Hands raised frames — clamp 3–10
        if "hands_raised_frames" in cal:
            overrides["hands_raised_frames"] = max(3, min(10, int(cal["hands_raised_frames"])))

        # Cooldown overrides
        cooldowns = dict(self._cfg.get("cooldowns", {}))
        if "cooldown_zone_intrusion" in cal:
            cooldowns["ZONE_INTRUSION"] = max(30.0, min(300.0, float(cal["cooldown_zone_intrusion"])))
        if "cooldown_fall" in cal:
            cooldowns["FALL_DETECTED"] = max(15.0, min(120.0, float(cal["cooldown_fall"])))
        if cooldowns:
            overrides["cooldowns"] = cooldowns

        # Store scene metadata for logging/dashboard
        overrides["_scene_type"] = result.get("scene_type", "unknown")
        overrides["_seated_environment"] = seated
        overrides["_lighting"] = result.get("lighting", "unknown")
        overrides["_crowding"] = result.get("crowding_expected", "unknown")

        return overrides
