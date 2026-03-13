"""
HC-4: Out-of-scope queries return structured refusal with status REFUSED.
PRD: "Nuclear fuel rod query returns {status: REFUSED}"
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.middleware.auth import AuthenticatedUser, get_current_user


async def mock_get_current_user():
    return AuthenticatedUser(
        user_id="test-user-id",
        email="test@ortholink.test",
        org_id="test-org",
        role="admin",
    )


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def client_with_auth(client):
    """Override auth so /query can be called without real JWT."""
    from app.main import app as fastapi_app
    fastapi_app.dependency_overrides[get_current_user] = mock_get_current_user
    yield client
    fastapi_app.dependency_overrides.pop(get_current_user, None)


class TestNuclearRefusal:
    """Nuclear fuel rod and other out-of-scope queries must return status REFUSED."""

    def test_is_out_of_scope_nuclear_fuel_rod(self):
        """Scope check rejects nuclear fuel rod queries."""
        from app.core.anti_hallucination import is_out_of_scope

        refusal = is_out_of_scope(
            "nuclear fuel rod requirements for licensing", "US", "II"
        )
        assert refusal is not None
        assert refusal.refused is True
        assert "nuclear" in refusal.reason.lower() or "medical" in refusal.reason.lower()

    def test_nuclear_fuel_rod_returns_refused(self, client_with_auth):
        """PRD: ROA query 'nuclear fuel rod requirements' returns {status: REFUSED}."""
        response = client_with_auth.get(
            "/api/v1/query",
            params={"q": "nuclear fuel rod requirements for licensing", "country": "US"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "REFUSED"
        assert data.get("refused") is True
        assert data.get("refusal_reason")

    def test_refusal_status_field_exists(self):
        """Response model must include status field that can be REFUSED."""
        from app.routes.query import QueryResponse

        r = QueryResponse(
            question="nuclear fuel rod",
            answer="",
            confidence=0.0,
            country="US",
            refused=True,
            refusal_reason="Out of scope",
            status="REFUSED",
        )
        assert r.status == "REFUSED"
        assert r.refused is True
