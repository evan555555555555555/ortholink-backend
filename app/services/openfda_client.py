"""
OrthoLink — Live openFDA API Client

Provides real-time access to FDA device data:
  - Device recalls (Class I/II/III)
  - MAUDE adverse event reports
  - 510(k) clearance records
  - Device classification lookups

No API key required for up to 1,000 req/hr.
Set FDA_API_KEY in .env for 120,000 req/hr.

Usage:
    from app.services.openfda_client import fda_client
    recalls = await fda_client.get_recent_recalls(device_name="knee implant")
    events  = await fda_client.get_adverse_events(product_code="KYF")
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_BASE = "https://api.fda.gov/device"


class OpenFDAClient:
    """Thin async wrapper around the openFDA device endpoints."""

    def __init__(self):
        settings = get_settings()
        self._api_key: str = getattr(settings, "fda_api_key", "") or ""

    def _params(self, extra: dict) -> dict:
        p = dict(extra)
        if self._api_key:
            p["api_key"] = self._api_key
        return p

    async def get_recent_recalls(
        self,
        device_name: Optional[str] = None,
        product_code: Optional[str] = None,
        days_back: int = 90,
        limit: int = 25,
    ) -> list[dict]:
        """
        Fetch recent FDA device recalls.
        Returns list of recall records with firm, product, classification, date.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y%m%d")
        parts = [f"event_date_initiated:[{cutoff} TO 99991231]"]
        if device_name:
            parts.append(f'product_description:"{device_name}"')
        if product_code:
            parts.append(f"product_code:{product_code}")

        search_query = " AND ".join(parts)
        params = self._params({"search": search_query, "limit": limit})
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(f"{_BASE}/recall.json", params=params)
                r.raise_for_status()
                data = r.json()
                results = data.get("results", [])
                logger.info("openFDA recalls: %d records returned", len(results))
                return results
        except httpx.HTTPStatusError as e:
            logger.warning("openFDA recalls HTTP error %s: %s", e.response.status_code, e)
            return []
        except Exception as e:
            logger.warning("openFDA recalls fetch failed: %s", e)
            return []

    async def get_adverse_events(
        self,
        product_code: Optional[str] = None,
        device_name: Optional[str] = None,
        days_back: int = 180,
        limit: int = 25,
    ) -> list[dict]:
        """
        Fetch MAUDE adverse event (MDR) reports.
        Returns list of event records with device, patient outcome, event type.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y%m%d")
        parts = [f"date_received:[{cutoff} TO 99991231]"]
        if product_code:
            parts.append(f"device.device_report_product_code:{product_code}")
        if device_name:
            parts.append(f'device.brand_name:"{device_name}"')

        search_query = " AND ".join(parts)
        params = self._params({"search": search_query, "limit": limit})
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(f"{_BASE}/event.json", params=params)
                r.raise_for_status()
                results = r.json().get("results", [])
                logger.info("openFDA MAUDE events: %d records returned", len(results))
                return results
        except httpx.HTTPStatusError as e:
            logger.warning("openFDA MAUDE HTTP error %s: %s", e.response.status_code, e)
            return []
        except Exception as e:
            logger.warning("openFDA MAUDE fetch failed: %s", e)
            return []

    async def get_510k_clearances(
        self,
        device_name: Optional[str] = None,
        product_code: Optional[str] = None,
        decision: str = "SESE",  # SESE = Substantially Equivalent
        limit: int = 10,
    ) -> list[dict]:
        """
        Fetch 510(k) clearances. Useful for predicate device research.
        """
        parts = [f"decision_code:{decision}"]
        if device_name:
            parts.append(f'device_name:"{device_name}"')
        if product_code:
            parts.append(f"product_code:{product_code}")

        params = self._params({"search": " AND ".join(parts), "limit": limit})
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{_BASE}/510k.json", params=params)
                r.raise_for_status()
                results = r.json().get("results", [])
                logger.info("openFDA 510k: %d records returned", len(results))
                return results
        except Exception as e:
            logger.warning("openFDA 510k fetch failed: %s", e)
            return []

    async def get_device_classification(
        self,
        product_code: Optional[str] = None,
        device_name: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Look up device classification (Class I/II/III) by product code or name.
        """
        parts = []
        if product_code:
            parts.append(f"product_code:{product_code}")
        if device_name:
            parts.append(f'device_name:"{device_name}"')
        if not parts:
            return []

        params = self._params({"search": " AND ".join(parts), "limit": limit})
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{_BASE}/classification.json", params=params)
                r.raise_for_status()
                results = r.json().get("results", [])
                return results
        except Exception as e:
            logger.warning("openFDA classification fetch failed: %s", e)
            return []

    def format_recall_summary(self, recalls: list[dict]) -> str:
        """Format recall records into a readable text summary for RAA alerts."""
        if not recalls:
            return "No recent FDA device recalls found."
        lines = [f"FDA Device Recalls (last 90 days): {len(recalls)} records\n"]
        for r in recalls[:10]:
            firm = r.get("recalling_firm", "Unknown firm")
            product = r.get("product_description", "Unknown device")[:80]
            cls = r.get("classification", "?")
            date = r.get("recall_initiation_date", "?")
            lines.append(f"  [{cls}] {firm} — {product} (initiated {date})")
        return "\n".join(lines)

    def format_maude_summary(self, events: list[dict]) -> str:
        """Format MAUDE events into a readable text summary for PMS agent."""
        if not events:
            return "No recent MAUDE adverse events found."
        lines = [f"MAUDE Adverse Events (last 180 days): {len(events)} records\n"]
        for e in events[:10]:
            devices = e.get("device", [{}])
            brand = devices[0].get("brand_name", "Unknown") if devices else "Unknown"
            event_type = e.get("event_type", "?")
            outcome = e.get("patient", [{}])
            outcome_code = outcome[0].get("sequence_number_outcome", "?") if outcome else "?"
            date = e.get("date_received", "?")
            lines.append(f"  [{event_type}] {brand} — outcome: {outcome_code} (received {date})")
        return "\n".join(lines)


# Singleton
fda_client = OpenFDAClient()
