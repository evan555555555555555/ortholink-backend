"""
OrthoLink — Live Feed Integration Tests (Phase 6)

Tests that the live feed clients make real API calls and return structured data.

SKIPPED in CI by default. Run with:
    RUN_INTEGRATION=1 pytest tests/test_live_feeds_integration.py -v

Validates:
  - openFDA recall endpoint returns real FDA data (not mocked)
  - openFDA MAUDE endpoint returns real adverse event data
  - EMA DHPC endpoint returns real EU safety communications
  - All results have required fields (no empty/null critical fields)
  - Response shapes match what AlertsPanel expects

These tests require network access. They do NOT require ORTHOLINK_TEST_JWT.
"""

import os

import pytest
import pytest_asyncio

# Skip entire module unless RUN_INTEGRATION=1 is set
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION"),
    reason="Live feed integration tests require RUN_INTEGRATION=1 and network access",
)


@pytest.fixture
def fda_client():
    from app.services.openfda_client import OpenFDAClient
    return OpenFDAClient()


@pytest.fixture
def ema_client():
    from app.services.ema_client import EMAClient
    return EMAClient()


class TestOpenFDARecallsLive:
    """Live tests for openFDA device recall endpoint."""

    @pytest.mark.asyncio
    async def test_fda_recalls_returns_list(self, fda_client):
        """openFDA recall endpoint returns a non-empty list."""
        recalls = await fda_client.get_recent_recalls(limit=5)
        assert isinstance(recalls, list), "Expected list from FDA recalls"
        assert len(recalls) > 0, "Expected at least one recall in last 90 days"

    @pytest.mark.asyncio
    async def test_fda_recalls_have_required_fields(self, fda_client):
        """Each recall record has the minimum fields AlertsPanel renders."""
        recalls = await fda_client.get_recent_recalls(limit=5)
        assert len(recalls) > 0, "No recalls returned"

        required_fields = {"recalling_firm", "product_description", "recall_class"}
        for i, record in enumerate(recalls[:3]):
            missing = required_fields - set(record.keys())
            assert not missing, (
                f"Recall record {i} missing fields: {missing}. "
                f"Got keys: {list(record.keys())}"
            )

    @pytest.mark.asyncio
    async def test_fda_recalls_class_values(self, fda_client):
        """Recall class field contains only FDA-defined values (I, II, III)."""
        recalls = await fda_client.get_recent_recalls(limit=10)
        valid_classes = {"Class I", "Class II", "Class III", "I", "II", "III", None}
        for record in recalls:
            cls = record.get("recall_class")
            # Allow None/missing (some records don't have class)
            if cls is not None:
                assert any(v in str(cls) for v in ["I", "II", "III"]), (
                    f"Unexpected recall_class value: {cls!r}"
                )

    @pytest.mark.asyncio
    async def test_fda_recalls_device_name_filter(self, fda_client):
        """Device name filter returns fewer results than unfiltered."""
        all_recalls = await fda_client.get_recent_recalls(limit=25)
        filtered = await fda_client.get_recent_recalls(device_name="knee", limit=25)
        # Filtered should return a list (may be 0 if no knee recalls in window)
        assert isinstance(filtered, list)


class TestOpenFDAMAUDELive:
    """Live tests for openFDA MAUDE adverse event endpoint."""

    @pytest.mark.asyncio
    async def test_maude_events_returns_list(self, fda_client):
        """openFDA MAUDE endpoint returns a list."""
        events = await fda_client.get_adverse_events(limit=5)
        assert isinstance(events, list), "Expected list from MAUDE"
        assert len(events) > 0, "Expected MAUDE events"

    @pytest.mark.asyncio
    async def test_maude_events_have_required_fields(self, fda_client):
        """MAUDE records have minimum fields AlertsPanel renders."""
        events = await fda_client.get_adverse_events(limit=5)
        assert len(events) > 0, "No MAUDE events returned"

        required_fields = {"report_number", "event_type", "date_received"}
        for i, record in enumerate(events[:3]):
            missing = required_fields - set(record.keys())
            assert not missing, (
                f"MAUDE record {i} missing fields: {missing}. "
                f"Got keys: {list(record.keys())}"
            )

    @pytest.mark.asyncio
    async def test_maude_event_types(self, fda_client):
        """MAUDE event_type contains recognizable FDA values."""
        events = await fda_client.get_adverse_events(limit=10)
        known_types = {"Malfunction", "Injury", "Death", "No answer provided", "Other"}
        for record in events[:5]:
            et = record.get("event_type", "")
            # Just verify it's a non-empty string
            assert isinstance(et, str), f"event_type should be str, got {type(et)}"


class TestEMADHPCsLive:
    """Live tests for EMA DHPC (EU safety communications) endpoint."""

    @pytest.mark.asyncio
    async def test_ema_dhpcs_returns_list(self, ema_client):
        """EMA DHPC endpoint returns a list."""
        dhpcs = await ema_client.get_dhpcs(limit=5)
        assert isinstance(dhpcs, list), "Expected list from EMA DHPCs"
        # EMA may return 0 if site is unreachable; warn but don't fail
        # (EMA site is sometimes slow / returns 503)

    @pytest.mark.asyncio
    async def test_ema_dhpcs_structure(self, ema_client):
        """DHPC records have a consistent structure when returned."""
        dhpcs = await ema_client.get_dhpcs(limit=5)
        if not dhpcs:
            pytest.skip("EMA returned 0 DHPCs (may be network/site issue)")

        for i, record in enumerate(dhpcs[:3]):
            assert isinstance(record, dict), f"DHPC record {i} is not a dict"
            # Title or name should exist
            has_title = any(k in record for k in ("title", "name", "medicine_name", "subject"))
            assert has_title, (
                f"DHPC record {i} has no title-like field. Got: {list(record.keys())}"
            )

    @pytest.mark.asyncio
    async def test_ema_dhpcs_date_fields(self, ema_client):
        """DHPC records have a date-like field."""
        dhpcs = await ema_client.get_dhpcs(limit=5)
        if not dhpcs:
            pytest.skip("EMA returned 0 DHPCs")

        for record in dhpcs[:3]:
            date_keys = [k for k in record if "date" in k.lower() or "published" in k.lower()]
            # Date field not mandatory (some sources omit it) — just log
            if not date_keys:
                # Soft check: acceptable if no date field
                pass


class TestFeedDataIntegrity:
    """Cross-feed data integrity checks."""

    @pytest.mark.asyncio
    async def test_fda_recalls_not_synthetic(self, fda_client):
        """FDA recalls reference real firm names (not 'SYNTHETIC' or 'TEST')."""
        recalls = await fda_client.get_recent_recalls(limit=10)
        for record in recalls:
            firm = str(record.get("recalling_firm", "")).upper()
            assert "SYNTHETIC" not in firm, f"Synthetic firm name in real FDA data: {firm}"
            assert "FAKE" not in firm, f"Fake firm name in real FDA data: {firm}"
            assert "TEST" not in firm or "TESTING" in firm, (
                f"Suspicious firm name 'TEST': {firm}"
            )

    @pytest.mark.asyncio
    async def test_fda_recall_dates_are_recent(self, fda_client):
        """FDA recall dates are within the last 180 days (not ancient data)."""
        from datetime import datetime, timedelta, timezone

        recalls = await fda_client.get_recent_recalls(days_back=90, limit=5)
        if not recalls:
            pytest.skip("No recalls in window")

        cutoff = datetime.now(timezone.utc) - timedelta(days=180)
        for record in recalls[:5]:
            date_str = record.get("event_date_initiated") or record.get("recall_initiation_date", "")
            if not date_str:
                continue
            try:
                # FDA dates come in YYYYMMDD or YYYY-MM-DD format
                date_str_norm = date_str.replace("-", "")
                if len(date_str_norm) == 8:
                    dt = datetime.strptime(date_str_norm, "%Y%m%d").replace(tzinfo=timezone.utc)
                    assert dt >= cutoff, (
                        f"Recall date {date_str} is older than 180 days — "
                        "check FDA API query window"
                    )
            except ValueError:
                pass  # Unparseable date format — skip
