"""
OrthoLink --- Base Registry Adapter

Abstract base class for all national/regional medical device registry adapters.
Provides:
  - Retry logic with exponential backoff (httpx.AsyncClient)
  - Configurable rate limiting per adapter
  - Structured logging
  - Common Pydantic response models
  - Health-check interface

Subclasses implement concrete API calls for GUDID, EUDAMED, Swissdamed, ANVISA, etc.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Generic, Optional, TypeVar

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common response models
# ---------------------------------------------------------------------------

T = TypeVar("T", bound="DeviceRecord")


class DeviceRecord(BaseModel):
    """Base fields shared by every device registry record."""

    source_registry: str = Field(
        ..., description="Registry identifier, e.g. 'GUDID', 'EUDAMED'"
    )
    device_id: str = Field(
        ..., description="Primary identifier in the source registry"
    )
    device_name: str = Field(default="", description="Trade / brand name")
    manufacturer: str = Field(default="", description="Legal manufacturer name")
    description: str = Field(default="", description="Device description or intended purpose")
    risk_class: str = Field(default="", description="Risk classification (I, IIa, IIb, III, ...)")
    country: str = Field(default="", description="ISO 3166-1 alpha-2 country code")
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="Full raw API response preserved for audit trail",
    )
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RegistrationRecord(BaseModel):
    """A single device registration entry for a given country."""

    registry: str = ""
    country: str = ""
    device_id: str = ""
    device_name: str = ""
    manufacturer: str = ""
    status: str = ""
    valid_until: Optional[str] = None
    raw: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AdapterHealthStatus(BaseModel):
    """Result of a health-check probe."""

    adapter: str
    healthy: bool
    latency_ms: float = 0.0
    error: Optional[str] = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Rate limiter (token-bucket, per-adapter instance)
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Simple async token-bucket rate limiter."""

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens added per second.
            capacity: Maximum tokens in the bucket.
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume one."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._capacity, self._tokens + elapsed * self._rate
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            # Not enough tokens -- wait a short interval and retry
            await asyncio.sleep(1.0 / self._rate)


# ---------------------------------------------------------------------------
# Abstract base adapter
# ---------------------------------------------------------------------------


class BaseRegistryAdapter(ABC, Generic[T]):
    """
    Abstract base for national medical-device registry adapters.

    Subclasses MUST implement:
        fetch_device          -- single-device lookup by UDI-DI
        search_devices        -- free-text search
        fetch_registrations   -- all registrations for a country
        get_source_url        -- canonical source URL for the registry
    """

    # Subclasses may override these defaults
    BASE_URL: str = ""
    REGISTRY_NAME: str = "UNKNOWN"
    DEFAULT_TIMEOUT: float = 15.0
    MAX_RETRIES: int = 3
    BACKOFF_BASE: float = 1.0  # seconds; actual wait = base * 2^attempt
    RATE_LIMIT_RPS: float = 5.0  # requests per second
    RATE_LIMIT_BURST: int = 10  # burst capacity

    def __init__(self) -> None:
        self._bucket = _TokenBucket(
            rate=self.RATE_LIMIT_RPS, capacity=self.RATE_LIMIT_BURST
        )
        self._client_headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "OrthoLink/1.0 (medical-device-compliance-platform)",
        }

    # ------------------------------------------------------------------
    # HTTP helpers with retry + rate-limiting
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """
        Issue an HTTP request with retry/backoff and rate-limit awareness.

        Raises httpx.HTTPStatusError on 4xx/5xx after exhausting retries.
        """
        merged_headers = {**self._client_headers, **(headers or {})}
        effective_timeout = timeout or self.DEFAULT_TIMEOUT
        last_exc: Optional[Exception] = None

        for attempt in range(self.MAX_RETRIES):
            await self._bucket.acquire()
            try:
                async with httpx.AsyncClient(
                    timeout=effective_timeout,
                    follow_redirects=True,
                ) as client:
                    response = await client.request(
                        method,
                        url,
                        params=params,
                        json=json_body,
                        headers=merged_headers,
                    )
                    # 429 Too Many Requests -- honour Retry-After if present
                    if response.status_code == 429:
                        retry_after = float(
                            response.headers.get("Retry-After", self.BACKOFF_BASE * (2 ** attempt))
                        )
                        logger.warning(
                            "[%s] 429 rate-limited, waiting %.1fs (attempt %d/%d)",
                            self.REGISTRY_NAME,
                            retry_after,
                            attempt + 1,
                            self.MAX_RETRIES,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    # 5xx server errors -- retry with backoff
                    if response.status_code >= 500:
                        wait = self.BACKOFF_BASE * (2 ** attempt)
                        logger.warning(
                            "[%s] Server error %d, retrying in %.1fs (attempt %d/%d)",
                            self.REGISTRY_NAME,
                            response.status_code,
                            wait,
                            attempt + 1,
                            self.MAX_RETRIES,
                        )
                        await asyncio.sleep(wait)
                        continue

                    response.raise_for_status()
                    return response

            except httpx.TimeoutException as exc:
                wait = self.BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "[%s] Timeout on %s %s, retrying in %.1fs (attempt %d/%d)",
                    self.REGISTRY_NAME,
                    method,
                    url,
                    wait,
                    attempt + 1,
                    self.MAX_RETRIES,
                )
                last_exc = exc
                await asyncio.sleep(wait)

            except httpx.ConnectError as exc:
                wait = self.BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "[%s] Connection error on %s, retrying in %.1fs (attempt %d/%d)",
                    self.REGISTRY_NAME,
                    url,
                    wait,
                    attempt + 1,
                    self.MAX_RETRIES,
                )
                last_exc = exc
                await asyncio.sleep(wait)

            except httpx.HTTPStatusError as exc:
                # 4xx client errors should not be retried (except 429 handled above)
                logger.error(
                    "[%s] Client error %d on %s: %s",
                    self.REGISTRY_NAME,
                    exc.response.status_code,
                    url,
                    exc.response.text[:300],
                )
                raise

        # Exhausted retries
        msg = (
            f"[{self.REGISTRY_NAME}] All {self.MAX_RETRIES} retries exhausted for "
            f"{method} {url}"
        )
        logger.error(msg)
        if last_exc:
            raise last_exc
        raise httpx.ConnectError(msg)

    async def _get(
        self,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Convenience GET wrapper."""
        url = f"{self.BASE_URL}{path}" if not path.startswith("http") else path
        return await self._request(
            "GET", url, params=params, headers=headers, timeout=timeout
        )

    async def _post(
        self,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Convenience POST wrapper."""
        url = f"{self.BASE_URL}{path}" if not path.startswith("http") else path
        return await self._request(
            "POST", url, json_body=json_body, params=params,
            headers=headers, timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Abstract interface -- every adapter MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch_device(self, udi_di: str) -> Optional[T]:
        """
        Look up a single device by its UDI-DI (or equivalent primary key).
        Returns None if not found or on unrecoverable error.
        """

    @abstractmethod
    async def search_devices(self, query: str, limit: int = 20) -> list[T]:
        """
        Free-text search across the registry.
        Returns up to *limit* matching device records.
        Always returns an empty list on failure -- never raises.
        """

    @abstractmethod
    async def fetch_registrations(self, country: str) -> list[RegistrationRecord]:
        """
        Fetch all device registrations for a given country/market.
        Returns list of RegistrationRecord.
        """

    @abstractmethod
    def get_source_url(self) -> str:
        """Return the canonical public URL for this registry (for citations)."""

    # ------------------------------------------------------------------
    # Alias: registries route calls .search() but interface uses search_devices
    # ------------------------------------------------------------------

    async def search(self, query: str, limit: int = 20) -> list[T]:
        """Alias for search_devices — used by registries route."""
        return await self.search_devices(query=query, limit=limit)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> AdapterHealthStatus:
        """
        Probe the registry API to confirm connectivity.
        Returns AdapterHealthStatus with latency and error info.
        """
        start = time.monotonic()
        try:
            response = await self._get(
                self._health_check_path(),
                timeout=10.0,
            )
            latency = (time.monotonic() - start) * 1000.0
            healthy = 200 <= response.status_code < 400
            logger.info(
                "[%s] Health check %s (%.0fms)",
                self.REGISTRY_NAME,
                "OK" if healthy else f"FAIL({response.status_code})",
                latency,
            )
            return AdapterHealthStatus(
                adapter=self.REGISTRY_NAME,
                healthy=healthy,
                latency_ms=round(latency, 1),
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000.0
            logger.warning(
                "[%s] Health check FAILED (%.0fms): %s",
                self.REGISTRY_NAME,
                latency,
                exc,
            )
            return AdapterHealthStatus(
                adapter=self.REGISTRY_NAME,
                healthy=False,
                latency_ms=round(latency, 1),
                error=str(exc),
            )

    def _health_check_path(self) -> str:
        """
        Override in subclass to provide a lightweight endpoint for health probes.
        Defaults to root path.
        """
        return "/"

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _safe_str(self, value: Any, default: str = "") -> str:
        """Safely coerce API value to str; return default on None."""
        if value is None:
            return default
        return str(value).strip()

    def _safe_list(self, value: Any) -> list:
        """Return value as list; wrap scalar in list; None becomes []."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]
