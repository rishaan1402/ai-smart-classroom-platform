"""
Integration tests for API auth flow.

Requires a running PostgreSQL + Redis (use docker compose up -d postgres redis).
Set TEST_DATABASE_URL env var before running.

Run:
    TEST_DATABASE_URL=postgresql://sentinel:sentinel@localhost:5432/sentinel \
    REDIS_HOST=localhost REDIS_PASSWORD=<pw> \
    pytest tests/integration/test_api_auth.py -v
"""

import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Skip entire module if DB not configured
pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL not set — skipping integration tests",
)


@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    os.environ.setdefault("JWT_SECRET", "test-secret-key-integration")
    os.environ.setdefault("CREDS_MASTER_KEY", "")

    from api.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestAuthFlow:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code in (200, 503)  # 503 ok if Ollama not running
        data = r.json()
        assert "status" in data

    def test_register_first_user_is_admin(self, client):
        r = client.post("/api/auth/register", json={
            "email": "admin_inttest@test.com",
            "password": "TestPass123!",
        })
        assert r.status_code == 200
        assert r.json()["role"] == "admin"

    def test_register_second_user_is_viewer(self, client):
        r = client.post("/api/auth/register", json={
            "email": "viewer_inttest@test.com",
            "password": "TestPass123!",
        })
        assert r.status_code == 200
        assert r.json()["role"] == "viewer"

    def test_duplicate_register_fails(self, client):
        r = client.post("/api/auth/register", json={
            "email": "admin_inttest@test.com",
            "password": "TestPass123!",
        })
        assert r.status_code == 409

    def test_login_valid(self, client):
        r = client.post("/api/auth/login", json={
            "email": "admin_inttest@test.com",
            "password": "TestPass123!",
        })
        assert r.status_code == 200
        assert "access_token" in r.cookies

    def test_login_wrong_password(self, client):
        r = client.post("/api/auth/login", json={
            "email": "admin_inttest@test.com",
            "password": "wrong",
        })
        assert r.status_code == 401

    def test_get_me_authenticated(self, client):
        # Log in first
        login = client.post("/api/auth/login", json={
            "email": "admin_inttest@test.com",
            "password": "TestPass123!",
        })
        assert login.status_code == 200
        r = client.get("/api/auth/me")
        assert r.status_code == 200
        data = r.json()
        assert data["role"] == "admin"

    def test_get_me_unauthenticated(self, client):
        # Fresh client without cookies
        from api.main import app
        with TestClient(app) as fresh:
            r = fresh.get("/api/auth/me")
            assert r.status_code == 401

    def test_events_requires_auth(self, client):
        from api.main import app
        with TestClient(app) as fresh:
            r = fresh.get("/api/events")
            assert r.status_code == 401

    def test_events_returns_list(self, client):
        client.post("/api/auth/login", json={
            "email": "admin_inttest@test.com",
            "password": "TestPass123!",
        })
        r = client.get("/api/events")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_cameras_admin_only_create(self, client):
        # Viewer should not create cameras
        client.post("/api/auth/login", json={
            "email": "viewer_inttest@test.com",
            "password": "TestPass123!",
        })
        r = client.post("/api/cameras", json={
            "name": "Test Cam",
            "host": "192.168.1.1:554",
            "path": "/stream",
        })
        assert r.status_code == 403

    def test_cameras_admin_can_create(self, client):
        client.post("/api/auth/login", json={
            "email": "admin_inttest@test.com",
            "password": "TestPass123!",
        })
        r = client.post("/api/cameras", json={
            "name": "Test Cam",
            "host": "192.168.1.1:554",
            "path": "/stream1",
        })
        assert r.status_code == 201

    def test_ws_token_rate_limited(self, client):
        client.post("/api/auth/login", json={
            "email": "admin_inttest@test.com",
            "password": "TestPass123!",
        })
        # Should succeed first few times
        for _ in range(5):
            r = client.get("/api/auth/ws-token")
        # 6th should be rate limited
        r = client.get("/api/auth/ws-token")
        assert r.status_code == 429
