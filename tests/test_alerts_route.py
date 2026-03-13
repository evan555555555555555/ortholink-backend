"""
Tests for POST /api/v1/alerts/check-changes and the RAA alert store.

Unit tests mock the raa_agent and Supabase so they run without network calls.
Integration tests (marked @pytest.mark.integration) need a live server + FAISS index.
"""

import time
from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient

TEST_JWT_SECRET = "test-jwt-secret-for-unit-tests-only"


def _make_token(role: str = "admin", org_id: str = "test-org") -> str:
    """Create a signed JWT for test requests."""
    payload = {
        "sub": "test-user",
        "aud": "authenticated",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
        "user_metadata": {"org_id": org_id, "role": role},
    }
    return jwt.encode(payload, TEST_JWT_SECRET, algorithm="HS256")


@pytest.fixture
def client(monkeypatch):
    """FastAPI test client with stubbed Supabase."""
    monkeypatch.setenv("SUPABASE_JWT_SECRET", TEST_JWT_SECRET)
    from app.core.config import get_settings
    get_settings.cache_clear()
    # Prevent alert_store from trying to connect to Supabase
    monkeypatch.setattr(
        "app.services.alert_store._supabase_loaded", True
    )
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


# ─── GET /api/v1/alerts ──────────────────────────────────────────────────────

class TestListAlerts:
    def test_returns_empty_list_with_valid_token(self, client):
        """Authenticated request returns alerts list."""
        token = _make_token(role="reviewer")
        resp = client.get(
            "/api/v1/alerts",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data
        assert isinstance(data["alerts"], list)

    def test_returns_401_without_token(self, client):
        """Unauthenticated request is rejected."""
        resp = client.get("/api/v1/alerts")
        assert resp.status_code in (401, 403)


# ─── POST /api/v1/alerts/subscribe ──────────────────────────────────────────

class TestSubscribeAlerts:
    def test_subscribe_to_country(self, client):
        """Valid subscription request stores the subscription."""
        token = _make_token(role="reviewer", org_id="acme")
        resp = client.post(
            "/api/v1/alerts/subscribe",
            json={"country": "US"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["subscribed"] is True
        assert data["country"] == "US"

    def test_subscribe_normalises_country_to_uppercase(self, client):
        """Country code should be returned uppercase."""
        token = _make_token(role="reviewer")
        resp = client.post(
            "/api/v1/alerts/subscribe",
            json={"country": "eu"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["country"] == "EU"


# ─── GET /api/v1/alerts/subscriptions ───────────────────────────────────────

class TestListSubscriptions:
    def test_returns_subscriptions_list(self, client):
        """Returns the list of countries the org subscribed to."""
        token = _make_token(role="reviewer", org_id="sub-test-org")
        # Subscribe first
        client.post(
            "/api/v1/alerts/subscribe",
            json={"country": "UA"},
            headers={"Authorization": f"Bearer {token}"},
        )
        resp = client.get(
            "/api/v1/alerts/subscriptions",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "countries" in data
        assert "UA" in data["countries"]


# ─── POST /api/v1/alerts/check-changes ──────────────────────────────────────

class TestCheckChanges:
    def test_requires_admin_role(self, client):
        """Non-admin role is rejected with 403."""
        token = _make_token(role="reviewer")
        resp = client.post(
            "/api/v1/alerts/check-changes",
            json={"country": "US"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_unknown_country_returns_failed_status(self, client):
        """Country with no monitored docs returns failed status (not 500)."""
        token = _make_token(role="admin")
        resp = client.post(
            "/api/v1/alerts/check-changes",
            json={"country": "ZZ"},  # not in registry
            headers={"Authorization": f"Bearer {token}"},
        )
        # Route should return 200 with status=failed, not crash
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "ZZ" in data.get("message", "")

    def test_unknown_document_id_returns_failed_status(self, client):
        """document_id not in registry returns failed (not 500)."""
        token = _make_token(role="admin")
        resp = client.post(
            "/api/v1/alerts/check-changes",
            json={"country": "US", "document_id": "NONEXISTENT_DOC"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "NONEXISTENT_DOC" in data.get("message", "")

    def test_missing_country_field_returns_422(self, client):
        """Missing required 'country' field returns 422 Unprocessable Entity."""
        token = _make_token(role="admin")
        resp = client.post(
            "/api/v1/alerts/check-changes",
            json={},  # country missing
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422

    def test_check_changes_no_change(self, client):
        """When RAA finds no change, returns completed with 0 alerts."""
        token = _make_token(role="admin")
        with patch("app.routes.alerts._run_check_changes_sync") as mock_sync:
            from app.routes.alerts import CheckChangesResult
            mock_sync.return_value = CheckChangesResult(
                alerts_emitted=0,
                changed_chunks=0,
                country="US",
                document_id=None,
                status="completed",
                message="No regulation changes detected for US.",
            )
            resp = client.post(
                "/api/v1/alerts/check-changes",
                json={"country": "US"},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["alerts_emitted"] == 0
        assert data["country"] == "US"

    def test_check_changes_with_alert(self, client):
        """When RAA detects a change, alerts_emitted is > 0."""
        token = _make_token(role="admin")
        with patch("app.routes.alerts._run_check_changes_sync") as mock_sync:
            from app.routes.alerts import CheckChangesResult
            mock_sync.return_value = CheckChangesResult(
                alerts_emitted=1,
                changed_chunks=1,
                country="US",
                document_id="US_FDA_21CFR820",
                status="completed",
                message="1 alert(s) emitted for US.",
            )
            resp = client.post(
                "/api/v1/alerts/check-changes",
                json={"country": "US", "document_id": "US_FDA_21CFR820"},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["alerts_emitted"] == 1
        assert data["status"] == "completed"

    def test_async_mode_returns_202_with_job_id(self, client):
        """async_mode=True returns HTTP 202 and a job_id."""
        token = _make_token(role="admin")
        with patch("app.services.job_store.create_job", return_value="test-job-123"):
            resp = client.post(
                "/api/v1/alerts/check-changes",
                json={"country": "US", "async_mode": True},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"


# ─── GET /api/v1/alerts/{country} ───────────────────────────────────────────

class TestAlertsByCountry:
    def test_returns_filtered_alerts(self, client):
        """Returns alerts filtered by the given country."""
        token = _make_token(role="reviewer")
        resp = client.get(
            "/api/v1/alerts/US",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data
        assert data["country"] == "US"


# ─── alert_store unit tests ──────────────────────────────────────────────────

class TestAlertStore:
    def test_add_and_retrieve_alert(self):
        """add_alert persists to in-memory store; get_alerts retrieves it."""
        from app.services.alert_store import (
            add_alert, clear_for_tests, get_alerts,
            subscribe, get_subscriptions,
        )
        clear_for_tests()

        event = {
            "country": "UA",
            "document_id": "UA_MOH_ORDER690",
            "change_summary": "Test change",
            "notified_orgs": ["org-1"],
        }
        add_alert(event)

        alerts = get_alerts(limit=50)
        assert len(alerts) == 1
        assert alerts[0]["country"] == "UA"

    def test_get_alerts_filters_by_country(self):
        """get_alerts filters by country correctly."""
        from app.services.alert_store import add_alert, clear_for_tests, get_alerts
        clear_for_tests()

        add_alert({"country": "US", "document_id": "US_FDA_21CFR820", "notified_orgs": []})
        add_alert({"country": "EU", "document_id": "EU_MDR_2017_745", "notified_orgs": []})

        us_alerts = get_alerts(country="US", limit=50)
        assert len(us_alerts) == 1
        assert us_alerts[0]["country"] == "US"

    def test_subscribe_and_get_subscriptions(self):
        """subscribe stores subscription; get_subscriptions returns it."""
        from app.services.alert_store import (
            clear_for_tests, subscribe, get_subscriptions,
        )
        clear_for_tests()

        subscribe("org-abc", "EU")
        subscribe("org-abc", "US")
        countries = get_subscriptions("org-abc")

        assert "EU" in countries
        assert "US" in countries

    def test_get_subscribed_orgs(self):
        """get_subscribed_orgs returns orgs watching a specific country."""
        from app.services.alert_store import (
            clear_for_tests, subscribe, get_subscribed_orgs,
        )
        clear_for_tests()

        subscribe("org-x", "UA")
        subscribe("org-y", "US")

        ua_orgs = get_subscribed_orgs("UA")
        assert "org-x" in ua_orgs
        assert "org-y" not in ua_orgs
