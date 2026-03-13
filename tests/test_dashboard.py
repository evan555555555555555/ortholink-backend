"""
Tests for GET /api/v1/dashboard/system-status

Verifies the system status aggregator returns correct structure
and enforces JWT authentication.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.middleware.auth import AuthenticatedUser, get_current_user

_app = create_app()


@pytest.fixture
def mock_reviewer_user():
    async def reviewer():
        return AuthenticatedUser(
            user_id="reviewer-user",
            email="reviewer@test.com",
            org_id="org-1",
            role="reviewer",
        )

    _app.dependency_overrides[get_current_user] = reviewer
    yield
    _app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def client(mock_reviewer_user):
    return TestClient(_app)


def test_dashboard_requires_auth():
    """System status endpoint requires JWT authentication."""
    c = TestClient(_app)
    r = c.get("/api/v1/dashboard/system-status")
    assert r.status_code in (401, 403)


def test_dashboard_returns_structure(client):
    """System status returns all expected top-level keys."""
    r = client.get("/api/v1/dashboard/system-status")
    assert r.status_code == 200
    data = r.json()

    # Top-level keys
    assert "global_health" in data
    assert "global_verdict" in data
    assert "latency_ms" in data
    assert "timestamp" in data
    assert "components" in data
    assert "defensive_architecture" in data
    assert "test_suite" in data

    assert isinstance(data["global_health"], bool)
    assert data["global_verdict"] in ("ALL_SYSTEMS_NOMINAL", "REVIEW_REQUIRED")
    assert data["latency_ms"] >= 0


def test_dashboard_components_present(client):
    """All three subsystem components are reported."""
    r = client.get("/api/v1/dashboard/system-status")
    data = r.json()
    components = data["components"]

    assert "redis_v3_cache" in components
    assert "faiss_vector_store" in components
    assert "hmac_integrity_engine" in components

    # Each component has a status field
    for name, comp in components.items():
        assert "status" in comp, f"Component {name} missing 'status'"


def test_dashboard_defensive_architecture_flags(client):
    """Defensive architecture invariants are all reported as True."""
    r = client.get("/api/v1/dashboard/system-status")
    arch = r.json()["defensive_architecture"]

    assert arch["country_isolation"] is True
    assert arch["revoked_law_filter"] is True
    assert arch["static_fallback_floor"] is True
    assert arch["cross_country_contamination_blocked"] is True
    assert arch["cache_key_format"] == "faiss:v3:{COUNTRY}:{SHA256}"


def test_dashboard_cache_version_v3(client):
    """Cache component reports v3 key format."""
    r = client.get("/api/v1/dashboard/system-status")
    cache = r.json()["components"]["redis_v3_cache"]
    assert cache["version"] == "v3"
    assert cache["key_format"] == "faiss:v3:{COUNTRY}:{SHA256}"
