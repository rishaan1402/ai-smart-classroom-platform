"""Tests for the JSON parser used in LLM/VLM workers (handles nested braces)."""

import pytest
from sentinel_engine.workers.llm import LLMWorker


def _make_worker():
    """Minimal LLMWorker instance without Redis for testing _parse_json."""
    class _MockRedis:
        def xgroup_create(self, *a, **kw): pass
    cfg = {"tenant_id": "test", "ollama_url": "http://localhost:11434",
           "llm_model": "qwen3:1.7b", "llm_timeout_s": 5,
           "stream_pending_max_age_s": 300}
    w = object.__new__(LLMWorker)
    w._cfg = cfg
    w._tenant_id = "test"
    w._redis = _MockRedis()
    return w


def parse(text: str) -> dict:
    return _make_worker()._parse_json(text)


class TestJsonParser:
    def test_clean_json(self):
        result = parse('{"threat_level": "low", "escalate_to_vlm": false}')
        assert result["threat_level"] == "low"
        assert result["escalate_to_vlm"] is False

    def test_json_with_preamble(self):
        """LLM often emits text before the JSON object."""
        result = parse('Sure, here is the analysis: {"threat_level": "high", "description": "running"}')
        assert result["threat_level"] == "high"

    def test_nested_braces(self):
        """Regex fails here; raw_decode handles it correctly."""
        text = '{"threat_level": "medium", "metadata": {"zone": "entrance", "count": 2}}'
        result = parse(text)
        assert result["threat_level"] == "medium"
        assert result["metadata"]["zone"] == "entrance"

    def test_json_with_trailing_text(self):
        result = parse('{"threat_level": "low"} some trailing text')
        assert result["threat_level"] == "low"

    def test_no_json_falls_back_to_defaults(self):
        """When no JSON present, _parse_json returns a safe default dict (never crashes)."""
        result = parse("Sorry, I cannot analyze this.")
        assert "threat_level" in result
        assert "escalate_to_vlm" in result

    def test_empty_string_falls_back_to_defaults(self):
        result = parse("")
        assert "threat_level" in result

    def test_escalate_true(self):
        result = parse('Analysis done. {"threat_level": "critical", "escalate_to_vlm": true, "description": "weapon visible"}')
        assert result["escalate_to_vlm"] is True
        assert result["threat_level"] == "critical"
