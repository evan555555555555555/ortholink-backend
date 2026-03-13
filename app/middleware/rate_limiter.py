"""
OrthoLink Rate Limiter Middleware
Sliding-window per-IP rate limiting. Zero new dependencies (stdlib only).

Tiers:
  - AI-heavy endpoints   → 20 RPM  (LLM calls, FAISS queries)
  - All other /api/ paths → 60 RPM
  - Health / docs         → unlimited

Algorithm: per-IP deque of monotonic timestamps, evict outside 60s window.
Thread-safe; deque mutations protected by per-bucket Lock.
"""

import logging
import os
import time
from collections import deque
from threading import Lock
from typing import Deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Paths that consume significant LLM / FAISS resources — stricter rate limit
_AI_PATHS: frozenset[str] = frozenset(
    {
        "/api/v1/verify-distributor",
        "/api/v1/review-document",
        "/api/v1/generate-checklist",
        "/api/v1/plan-strategy",
        "/api/v1/technical-dossier",
        "/api/v1/pms-plan",
        "/api/v1/capa",
        "/api/v1/risk-analysis",
        "/api/v1/gco-analysis",
        "/api/v1/verify-claims",
        "/api/v1/briefing",
        "/api/v1/alerts/check-changes",
    }
)

_WINDOW_SECONDS: int = 60
_MAX_TRACKED_IPS: int = 10_000  # LRU eviction threshold


class _Bucket:
    """Thread-safe sliding-window counter for a single IP address."""

    __slots__ = ("_dq", "_lock")

    def __init__(self) -> None:
        self._dq: Deque[float] = deque()
        self._lock = Lock()

    def _evict(self, cutoff: float) -> None:
        """Remove timestamps older than the window (caller holds lock)."""
        while self._dq and self._dq[0] < cutoff:
            self._dq.popleft()

    def is_allowed(self, limit: int) -> bool:
        now = time.monotonic()
        cutoff = now - _WINDOW_SECONDS
        with self._lock:
            self._evict(cutoff)
            if len(self._dq) >= limit:
                return False
            self._dq.append(now)
            return True

    def remaining(self, limit: int) -> int:
        now = time.monotonic()
        cutoff = now - _WINDOW_SECONDS
        with self._lock:
            self._evict(cutoff)
            return max(0, limit - len(self._dq))

    def oldest_ts(self) -> float:
        """Monotonic timestamp of the oldest request in the window."""
        with self._lock:
            return self._dq[0] if self._dq else time.monotonic()

    def reset_in(self) -> int:
        """Seconds until the oldest request falls outside the window."""
        return max(1, int(self.oldest_ts() + _WINDOW_SECONDS - time.monotonic()) + 1)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Per-IP sliding-window rate limiter.
    Skips health, docs, and non-API paths entirely.
    """

    def __init__(self, app, default_rpm: int = 60, ai_rpm: int = 20) -> None:
        super().__init__(app)
        self._default_rpm = default_rpm
        self._ai_rpm = ai_rpm
        self._buckets: dict[str, _Bucket] = {}
        self._meta_lock = Lock()

    # ── IP extraction ─────────────────────────────────────────────────────────

    # IPs that are allowed to set X-Forwarded-For (Docker bridge + loopback).
    # Extend this if you add a real load balancer with a fixed IP.
    _TRUSTED_PROXIES: frozenset[str] = frozenset({"127.0.0.1", "::1", "172.17.0.1", "172.18.0.1"})

    @staticmethod
    def _client_ip(request: Request) -> str:
        peer = request.client.host if request.client else "unknown"
        xff = request.headers.get("X-Forwarded-For")
        # Only honour the XFF header when the direct peer is a trusted proxy.
        # This prevents clients from spoofing their IP to bypass rate limits.
        if xff and peer in RateLimiterMiddleware._TRUSTED_PROXIES:
            return xff.split(",")[0].strip()
        return peer

    # ── Bucket management ─────────────────────────────────────────────────────

    def _bucket(self, ip: str) -> _Bucket:
        with self._meta_lock:
            if ip not in self._buckets:
                if len(self._buckets) >= _MAX_TRACKED_IPS:
                    # Evict the oldest insertion (dict preserves insertion order in Py3.7+)
                    oldest_key = next(iter(self._buckets))
                    del self._buckets[oldest_key]
                self._buckets[ip] = _Bucket()
            return self._buckets[ip]

    # ── Request dispatch ──────────────────────────────────────────────────────

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Exempt: test environment (unit/integration test suites share one IP)
        if os.environ.get("ENVIRONMENT") == "test":
            return await call_next(request)

        # Exempt: health probes, OpenAPI docs, static assets
        if not path.startswith("/api/"):
            return await call_next(request)

        is_ai = path in _AI_PATHS
        limit = self._ai_rpm if is_ai else self._default_rpm
        ip = self._client_ip(request)
        bucket = self._bucket(ip)

        if not bucket.is_allowed(limit):
            retry_after = bucket.reset_in()
            logger.warning(
                "Rate limit exceeded — ip=%s path=%s limit=%d/min",
                ip,
                path,
                limit,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please slow down.",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                },
            )

        remaining = bucket.remaining(limit)
        response = await call_next(request)

        # Append rate-limit info headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + _WINDOW_SECONDS)

        return response
