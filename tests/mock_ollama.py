"""
Mock Ollama server for integration tests.

Starts a tiny HTTP server on localhost:11435 that returns deterministic
LLM/VLM responses — no GPU needed for CI.

Usage:
    with MockOllama() as server:
        # test code hitting OLLAMA_URL=http://localhost:11435
        ...
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

LLM_RESPONSE = json.dumps({
    "model": "qwen3:1.7b",
    "response": json.dumps({
        "threat_level": "low",
        "description": "Person walking normally",
        "escalate_to_vlm": False,
    }),
    "done": True,
})

VLM_RESPONSE = json.dumps({
    "model": "qwen2.5vl:3b",
    "response": json.dumps({
        "threat_level": "low",
        "description": "No threat detected in image",
        "actions": [],
    }),
    "done": True,
})

TAGS_RESPONSE = json.dumps({
    "models": [
        {"name": "qwen3:1.7b"},
        {"name": "qwen2.5vl:3b"},
    ]
})


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/tags":
            self._respond(200, TAGS_RESPONSE)
        else:
            self._respond(404, '{"error":"not found"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode() if length else ""
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {}

        model = data.get("model", "")
        if "vl" in model:
            self._respond(200, VLM_RESPONSE)
        else:
            self._respond(200, LLM_RESPONSE)

    def _respond(self, code: int, body: str):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass  # suppress request logs in test output


class MockOllama:
    def __init__(self, port: int = 11435):
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._server = HTTPServer(("localhost", self.port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"
