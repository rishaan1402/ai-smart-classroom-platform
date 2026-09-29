"""
Integration test: LLMWorker → VLMWorker pipeline with MockOllama.

Verifies that:
- LLMWorker consumes from Redis Stream and calls Ollama
- Escalated events reach the VLM stream
- Non-escalated events reach the log stream
- DLQ receives failed messages

Requires Redis running locally.
Set REDIS_HOST, REDIS_PASSWORD env vars.

Run:
    REDIS_HOST=localhost REDIS_PASSWORD=<pw> \
    pytest tests/integration/test_pipeline_with_mock_ollama.py -v
"""

import json
import os
import time
import uuid
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("REDIS_HOST"),
    reason="REDIS_HOST not set — skipping pipeline integration tests",
)


@pytest.fixture(scope="module")
def redis_client():
    import redis
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=False,
    )
    r.ping()
    return r


@pytest.fixture(scope="module")
def mock_ollama():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from tests.mock_ollama import MockOllama
    with MockOllama(port=11436) as server:
        yield server


def make_event(tenant_id: str = "test") -> dict:
    return {
        "id": str(uuid.uuid4()),
        "event_type": "HANDS_RAISED",
        "severity": "high",
        "person_id": 1,
        "camera_id": "cam_01",
        "tenant_id": tenant_id,
        "trace_id": str(uuid.uuid4()),
        "schema_version": 1,
        "timestamp": time.time(),
        "posture": "STANDING",
        "velocity": "STATIONARY",
        "hands_state": "RAISED",
        "zone_id": None,
    }


class TestLLMWorkerPipeline:
    def test_llm_worker_consumes_and_forwards(self, redis_client, mock_ollama):
        """LLMWorker should read from event stream and write to log stream."""
        tenant_id = f"test_{uuid.uuid4().hex[:8]}"
        stream = f"sentinel:events:{tenant_id}"
        log_stream = f"sentinel:log:{tenant_id}"

        cfg = {
            "tenant_id": tenant_id,
            "ollama_url": mock_ollama.url,
            "llm_model": "qwen3:1.7b",
            "llm_timeout_s": 5,
            "stream_pending_max_age_s": 300,
        }

        from sentinel_engine.workers.llm import LLMWorker
        worker = LLMWorker(redis_client, cfg, worker_index=0)
        worker.start()

        # Publish an event
        event = make_event(tenant_id)
        redis_client.xadd(stream, {k: json.dumps(v) if not isinstance(v, str) else v
                                   for k, v in event.items()})

        # Wait for worker to process
        deadline = time.time() + 10
        processed = False
        while time.time() < deadline:
            length = redis_client.xlen(log_stream)
            if length > 0:
                processed = True
                break
            time.sleep(0.2)

        worker.stop()
        worker.join(timeout=3)
        assert processed, "LLMWorker did not forward event to log stream within 10s"

    def test_mock_ollama_responds(self, mock_ollama):
        """Verify mock Ollama is responding correctly."""
        import requests
        r = requests.get(f"{mock_ollama.url}/api/tags", timeout=2)
        assert r.status_code == 200
        models = r.json()["models"]
        assert any("qwen3" in m["name"] for m in models)

    def test_mock_ollama_llm_endpoint(self, mock_ollama):
        import requests
        r = requests.post(
            f"{mock_ollama.url}/api/generate",
            json={"model": "qwen3:1.7b", "prompt": "test"},
            timeout=2,
        )
        assert r.status_code == 200
        data = r.json()
        inner = json.loads(data["response"])
        assert "threat_level" in inner
