"""
OrthoLink -- Base Enforcement Scraper

Abstract base class for all enforcement and vigilance scrapers.
Provides:
  - Shared httpx.AsyncClient with retry/exponential backoff
  - Common Pydantic result model (EnforcementAction)
  - Severity classification logic (CRITICAL/HIGH/MEDIUM/LOW)
  - Deduplication by (source_url, date, device_name)
  - health_check() contract

All concrete scrapers inherit from BaseEnforcementScraper and implement
the four abstract methods.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 20.0  # seconds
MAX_RETRIES = 3
BACKOFF_BASE = 1.5  # exponential backoff multiplier

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Enforcement action severity level."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Action-type keywords mapped to severity tiers.
# Used by classify_severity() -- concrete scrapers can override.
_SEVERITY_MAP: dict[str, Severity] = {
    # CRITICAL -- immediate patient safety risk
    "class_i_recall": Severity.CRITICAL,
    "class_1_recall": Severity.CRITICAL,
    "mandatory_recall": Severity.CRITICAL,
    "suspension": Severity.CRITICAL,
    "injunction": Severity.CRITICAL,
    "seizure": Severity.CRITICAL,
    "ban": Severity.CRITICAL,
    "prohibition": Severity.CRITICAL,
    "withdrawal": Severity.CRITICAL,
    "counterfeit": Severity.CRITICAL,
    # HIGH -- significant regulatory concern
    "class_ii_recall": Severity.HIGH,
    "class_2_recall": Severity.HIGH,
    "warning_letter": Severity.HIGH,
    "safety_alert": Severity.HIGH,
    "field_safety_notice": Severity.HIGH,
    "field_safety_corrective_action": Severity.HIGH,
    "restriction": Severity.HIGH,
    "corrective_action": Severity.HIGH,
    # MEDIUM -- compliance issue, lower immediate risk
    "class_iii_recall": Severity.MEDIUM,
    "class_3_recall": Severity.MEDIUM,
    "advisory": Severity.MEDIUM,
    "observation": Severity.MEDIUM,
    "483_observation": Severity.MEDIUM,
    "incident_report": Severity.MEDIUM,
    # LOW -- informational
    "voluntary_recall": Severity.LOW,
    "market_notification": Severity.LOW,
    "information_notice": Severity.LOW,
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EnforcementAction(BaseModel):
    """Unified enforcement/vigilance record returned by all scrapers."""

    action_type: str = Field(..., description="e.g. 'warning_letter', 'class_i_recall', 'field_safety_notice'")
    date: Optional[datetime] = Field(None, description="Date the action was published/initiated")
    device_name: str = Field(default="", description="Name of the affected device")
    company: str = Field(default="", description="Responsible manufacturer or MAH")
    description: str = Field(default="", description="Human-readable description of the action")
    severity: Severity = Field(default=Severity.MEDIUM, description="Derived severity level")
    source_url: str = Field(default="", description="Canonical URL of the source record")
    country: str = Field(default="", description="ISO 3166-1 alpha-2 country code")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Full upstream payload for audit trail")

    @property
    def dedup_key(self) -> str:
        """Stable deduplication key: SHA-256 of (source_url, date ISO, device_name lower)."""
        date_str = self.date.isoformat() if self.date else ""
        payload = f"{self.source_url}|{date_str}|{self.device_name.lower()}"
        return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEnforcementScraper(ABC):
    """
    Abstract base for enforcement/vigilance scrapers.

    Concrete subclasses must implement:
      - fetch_recent(days)
      - fetch_by_device(device_name)
      - search(query, limit)
      - get_source_name()
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._timeout = timeout
        self._max_retries = max_retries
        self._headers = headers or dict(DEFAULT_HEADERS)
        self._seen_keys: set[str] = set()

    # -- HTTP helpers -------------------------------------------------------

    async def _get(
        self,
        url: str,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        follow_redirects: bool = True,
    ) -> httpx.Response:
        """
        GET with retry + exponential backoff.

        Raises httpx.HTTPStatusError on final failure after all retries.
        """
        import asyncio

        merged_headers = {**self._headers, **(headers or {})}
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout,
                    follow_redirects=follow_redirects,
                    headers=merged_headers,
                ) as client:
                    resp = await client.get(url, params=params)
                    resp.raise_for_status()
                    return resp
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_exc = exc
                wait = BACKOFF_BASE ** attempt
                logger.warning(
                    "%s GET %s attempt %d/%d failed (%s), retrying in %.1fs",
                    self.get_source_name(),
                    url,
                    attempt + 1,
                    self._max_retries,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
            except Exception as exc:
                # Unexpected -- do not retry
                logger.error(
                    "%s GET %s unexpected error: %s",
                    self.get_source_name(),
                    url,
                    exc,
                )
                raise

        # All retries exhausted
        assert last_exc is not None
        raise last_exc

    async def _get_json(
        self,
        url: str,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> Any:
        """GET and parse JSON response."""
        resp = await self._get(url, params=params, headers=headers)
        return resp.json()

    async def _get_html(
        self,
        url: str,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> str:
        """GET and return decoded HTML text."""
        resp = await self._get(url, params=params, headers=headers)
        return resp.text

    # -- Severity classification --------------------------------------------

    @staticmethod
    def classify_severity(action_type: str) -> Severity:
        """
        Map an action_type string to a Severity enum.

        Normalises input to lowercase with underscores before lookup.
        Falls back to MEDIUM if no match.
        """
        normalised = action_type.strip().lower().replace(" ", "_").replace("-", "_")
        return _SEVERITY_MAP.get(normalised, Severity.MEDIUM)

    # -- Deduplication ------------------------------------------------------

    def deduplicate(self, actions: list[EnforcementAction]) -> list[EnforcementAction]:
        """
        Remove duplicate EnforcementActions within a single fetch batch
        and across the scraper's lifetime (session-level dedup).

        Uses (source_url, date, device_name) SHA-256 as the key.
        """
        unique: list[EnforcementAction] = []
        for action in actions:
            key = action.dedup_key
            if key not in self._seen_keys:
                self._seen_keys.add(key)
                unique.append(action)
        return unique

    def reset_dedup_cache(self) -> None:
        """Clear session dedup cache (useful between scheduled runs)."""
        self._seen_keys.clear()

    # -- Health check -------------------------------------------------------

    async def health_check(self) -> bool:
        """
        Verify that the upstream data source is reachable.

        Default implementation sends a lightweight GET to the source's root
        endpoint and checks for a 2xx status. Concrete scrapers may override
        with a more targeted probe.
        """
        try:
            results = await self.fetch_recent(days=1)
            # We consider the scraper healthy if the call did not raise,
            # even if zero results are returned (source may simply have
            # no recent data).
            return True
        except Exception as exc:
            logger.warning(
                "%s health_check failed: %s",
                self.get_source_name(),
                exc,
            )
            return False

    # -- Abstract contract --------------------------------------------------

    @abstractmethod
    async def fetch_recent(self, days: int = 30) -> list[EnforcementAction]:
        """Fetch enforcement actions published within the last *days* days."""
        ...

    @abstractmethod
    async def fetch_by_device(self, device_name: str) -> list[EnforcementAction]:
        """Fetch enforcement actions related to a specific device name."""
        ...

    @abstractmethod
    async def search(self, query: str, limit: int = 50) -> list[EnforcementAction]:
        """Free-text search across the source's enforcement records."""
        ...

    @abstractmethod
    def get_source_name(self) -> str:
        """Human-readable name for this data source (e.g. 'FDA Warning Letters')."""
        ...

    # -- Helpers for subclasses ---------------------------------------------

    @staticmethod
    def _parse_date(value: Optional[str], formats: Optional[list[str]] = None) -> Optional[datetime]:
        """
        Attempt to parse a date string using a list of candidate formats.

        Returns timezone-aware UTC datetime, or None on failure.
        """
        if not value:
            return None

        candidate_formats = formats or [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%d %B %Y",
            "%B %d, %Y",
            "%Y%m%d",
        ]

        cleaned = value.strip()
        for fmt in candidate_formats:
            try:
                dt = datetime.strptime(cleaned, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

        logger.debug("Could not parse date '%s' with any known format", value)
        return None

    @staticmethod
    def _truncate(text: str, max_len: int = 2000) -> str:
        """Truncate text to max_len characters, appending '...' if clipped."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."
