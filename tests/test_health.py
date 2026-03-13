"""
Tests for health check endpoints.
"""

import time

import jwt
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_TEST_SECRET = "test-jwt-secret-for-unit-tests-only"


def _make_token() -> str:
    now = int(time.time())
    payload = {
        "sub": "test-user-123",
        "email": "test@example.com",
        "aud": "authenticated",
        "iat": now,
        "exp": now + 3600,
        "user_metadata": {"org_id": "test-org", "role": "admin"},
    }
    return jwt.encode(payload, _TEST_SECRET, algorithm="HS256")


class TestHealthEndpoints:
    """Test health check routes."""

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["app"] == "OrthoLink"

    def test_health_has_version(self):
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_detailed_health_check(self):
        token = _make_token()
        response = client.get("/health/detailed", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert "app" in data["checks"]
