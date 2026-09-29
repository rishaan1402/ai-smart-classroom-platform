"""
LLMWorker — Redis Streams consumer for LLM threat triage.

Uses Ollama qwen3:8b for fast text-only analysis of YOLO detection events.
Returns threat_level + escalate_to_vlm flag.

Explicit 10-second timeout on Ollama calls.
Failed messages go to DLQ stream.
"""

import json
import logging
import time
from typing import Any, Optional

import redis as redis_lib
import requests

from ._base_stream_worker import BaseStreamWorker

logger = logging.getLogger(__name__)

TRIAGE_PROMPT = """\
Analyze this security event and assess the threat level.

Event type: {event_type}
Severity: {severity}
Zone: {zone}
Person posture: {posture}
Person velocity: {velocity}
Hands state: {hands_state}

Respond ONLY with valid JSON:
{{
  "threat_level": "low" or "high",
  "description": "brief explanation under 100 words",
  "escalate_to_vlm": true or false
}}

escalate_to_vlm should be true only for high threat_level events or if visual confirmation would change the response."""


def _warmup_ollama(ollama_url: str, model: str, timeout: int = 45) -> bool:
    """Pre-warm Ollama model before cameras start."""
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": "ready", "stream": False},
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Ollama warmup failed: {e}")
        return False


class LLMWorker(BaseStreamWorker):
    """
    Consumes sentinel:events:{tenant_id} and runs LLM triage.

    Successful events are XACK'd.
    Events requiring VLM escalation are published to sentinel:vlm:{tenant_id}.
    Failed events go to sentinel:dlq:failed:{tenant_id}.
    """

    def __init__(
        self,
        redis_client: redis_lib.Redis,
        cfg: dict[str, Any],
        worker_index: int = 0,
    ) -> None:
        tenant_id = cfg.get("tenant_id", "default")
        super().__init__(
            name=f"LLMWorker-{worker_index}",
            stream_key=f"sentinel:events:{tenant_id}",
            group_name="llm_workers",
            consumer_name=f"llm_worker_{worker_index}",
            redis_client=redis_client,
            cfg=cfg,
            dlq_stream=f"sentinel:dlq:failed:{tenant_id}",
        )
        self._ollama_url = cfg.get("ollama_url", "http://localhost:11434")
        self._model = cfg.get("llm_model", "qwen3:8b")
        self._timeout = int(cfg.get("llm_timeout_s", 10))
        self._vlm_stream = f"sentinel:vlm:{tenant_id}"
        self._log_stream = f"sentinel:log:{tenant_id}"

    def _process_message(self, message_id: bytes, data: dict) -> None:
        event_json = data.get(b"event", b"{}").decode()
        trace_id = data.get(b"trace_id", b"unknown").decode()

        try:
            event = json.loads(event_json)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid event JSON: {event_json[:100]}")

        token = event.get("token", {})
        prompt = TRIAGE_PROMPT.format(
            event_type=event.get("event_type", "unknown"),
            severity=event.get("severity", "unknown"),
            zone=event.get("zone_id") or "unknown",
            posture=token.get("posture", "unknown"),
            velocity=token.get("velocity", "unknown"),
            hands_state=token.get("hands_state", "unknown"),
        )

        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 256},
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            result_text = resp.json().get("response", "")
            llm_result = self._parse_json(result_text)
        except requests.Timeout:
            raise TimeoutError(f"LLM timeout after {self._timeout}s")
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {e}")

        # Enrich event
        event["threat_level"] = llm_result.get("threat_level", "low")
        event["threat_description"] = llm_result.get("description", "")
        event["llm_processed"] = True

        enriched_json = json.dumps(event)

        # Always escalate to VLM — visual confirmation is the final gate.
        # Events only reach sentinel:log (and therefore the dashboard) once
        # VLMWorker confirms them. Low-threat events are suppressed by VLM.
        self._redis.xadd(
            self._vlm_stream,
            {"event": enriched_json, "trace_id": trace_id},
        )
        logger.debug(
            f"Forwarded to VLM for visual confirmation "
            f"(llm_threat={event['threat_level']})",
            extra={"trace_id": trace_id},
        )

    def _parse_json(self, text: str) -> dict:
        """Robustly parse JSON from LLM response (handles extra prose)."""
        import re
        # Try raw decode first (handles nested braces correctly)
        try:
            decoder = json.JSONDecoder()
            # Find the first { in the text
            start = text.find("{")
            if start != -1:
                obj, _ = decoder.raw_decode(text, start)
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: heuristic
        threat = "high" if "high" in text.lower() else "low"
        escalate = "true" in text.lower() and "escalate" in text.lower()
        return {
            "threat_level": threat,
            "description": text[:200],
            "escalate_to_vlm": escalate,
        }
