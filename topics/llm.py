"""
Minimal Ollama chat client with JSON-schema structured output.

Follows SentinelCore's worker conventions (requests + explicit timeouts,
raw_decode JSON fallback from sentinel_engine/workers/vlm.py).
"""

import json
import logging
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL, timeout_s: int = 300) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def chat_json(self, system: str, user: str, schema: dict, temperature: float = 0.1) -> dict:
        """Return the model's reply parsed as JSON, constrained to `schema`."""
        resp = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "format": schema,
                "stream": False,
                "think": False,          # qwen3: skip the reasoning trace, we only want the JSON
                "options": {"temperature": temperature, "num_ctx": 16384},
            },
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        parsed = parse_json(content)
        if parsed is None:
            raise ValueError(f"LLM returned non-JSON: {content[:200]}")
        return parsed


def parse_json(text: str) -> Optional[Any]:
    """Parse the first JSON object in text (handles surrounding prose / nested braces)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        return obj
    except (json.JSONDecodeError, ValueError):
        return None
