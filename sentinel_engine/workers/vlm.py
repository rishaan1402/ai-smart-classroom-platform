"""
VLMWorker — Redis Streams consumer for visual threat analysis.

Uses Ollama qwen2.5vl:7b for image-based threat assessment.
Only invoked for events escalated by LLMWorker (selective, ~5% of events).
Explicit 15-second timeout.
"""

import base64
import io
import json
import logging
from typing import Any, Optional

import numpy as np
import redis as redis_lib
import requests
from PIL import Image

from ._base_stream_worker import BaseStreamWorker

logger = logging.getLogger(__name__)

THREAT_PROMPT = """\
A person is {action} in a {zone}.
Analyze their body language and surroundings.
Are they holding a weapon or acting aggressively?

Respond ONLY with valid JSON:
{{
  "threat_level": "low" or "high",
  "reason": "brief explanation under 80 words"
}}"""


class VLMWorker(BaseStreamWorker):
    """
    Consumes sentinel:vlm:{tenant_id} and runs visual threat analysis.

    Requires the image crop to be in the event's snapshot_path field
    or encoded as base64 in event metadata.
    """

    def __init__(
        self,
        redis_client: redis_lib.Redis,
        cfg: dict[str, Any],
        worker_index: int = 0,
    ) -> None:
        tenant_id = cfg.get("tenant_id", "default")
        super().__init__(
            name=f"VLMWorker-{worker_index}",
            stream_key=f"sentinel:vlm:{tenant_id}",
            group_name="vlm_workers",
            consumer_name=f"vlm_worker_{worker_index}",
            redis_client=redis_client,
            cfg=cfg,
            dlq_stream=f"sentinel:dlq:failed:{tenant_id}",
        )
        self._ollama_url = cfg.get("ollama_url", "http://localhost:11434")
        self._model = cfg.get("vlm_model", "qwen2.5vl:7b")
        self._timeout = int(cfg.get("vlm_timeout_s", 15))
        self._log_stream = f"sentinel:log:{tenant_id}"

    def _process_message(self, message_id: bytes, data: dict) -> None:
        event_json = data.get(b"event", b"{}").decode()
        trace_id = data.get(b"trace_id", b"unknown").decode()

        event = json.loads(event_json)
        token = event.get("token", {})

        # Get image from metadata (base64 encoded by CameraWorker on event trigger)
        img_b64 = event.get("metadata", {}).get("image_b64")
        if not img_b64:
            # No image — suppress rather than forward unconfirmed
            logger.warning(
                f"VLM: no image for {event.get('event_type')} — suppressing "
                f"(event will not appear on dashboard)",
                extra={"trace_id": trace_id},
            )
            suppressed_key = f"sentinel:suppressed:{event.get('tenant_id', 'default')}"
            self._redis.incr(suppressed_key)
            return

        action = self._describe_action(token)
        zone = token.get("zone_name") or "unknown area"
        prompt = THREAT_PROMPT.format(action=action, zone=zone)

        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 256},
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            result_text = resp.json().get("response", "")
            vlm_result = self._parse_json(result_text)
        except requests.Timeout:
            raise TimeoutError(f"VLM timeout after {self._timeout}s")
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama VLM request failed: {e}")

        vlm_threat = vlm_result.get("threat_level", "low")
        event["threat_level"] = vlm_threat
        event["threat_description"] = vlm_result.get("reason", "")
        event["vlm_processed"] = True

        # VLM is the final gate — only confirmed HIGH threat events reach
        # the dashboard. LOW events are suppressed (counted but not logged).
        if vlm_threat == "high":
            self._redis.xadd(
                self._log_stream,
                {"event": json.dumps(event), "trace_id": trace_id},
            )
            logger.info(
                f"VLM CONFIRMED threat={vlm_threat} "
                f"type={event.get('event_type')} — forwarding to log",
                extra={"trace_id": trace_id},
            )
        else:
            # Increment suppressed counter for metrics (no DB write)
            suppressed_key = f"sentinel:suppressed:{event.get('tenant_id', 'default')}"
            self._redis.incr(suppressed_key)
            logger.info(
                f"VLM SUPPRESSED threat={vlm_threat} "
                f"type={event.get('event_type')} — not shown on dashboard",
                extra={"trace_id": trace_id},
            )

    def _describe_action(self, token: dict) -> str:
        posture = token.get("posture", "standing")
        velocity = token.get("velocity", "stationary")
        hands = token.get("hands_state", "lowered")

        if posture == "fallen":
            return "lying on the ground"
        if velocity == "running":
            return "running"
        if hands == "raised":
            return "standing with hands raised"
        if posture == "crouching":
            return "crouching"
        return "standing"

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from VLM response using raw_decode (handles nested braces)."""
        try:
            decoder = json.JSONDecoder()
            start = text.find("{")
            if start != -1:
                obj, _ = decoder.raw_decode(text, start)
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

        threat = "high" if "high" in text.lower() else "low"
        return {"threat_level": threat, "reason": text[:200]}

    @staticmethod
    def frame_to_b64(frame: np.ndarray) -> str:
        """Convert BGR numpy array to base64 PNG for Ollama."""
        rgb = frame[:, :, ::-1]  # BGR → RGB
        pil = Image.fromarray(rgb.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
