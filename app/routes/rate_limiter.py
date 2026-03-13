"""
OrthoLink Rate Limiter
Simple in-memory sliding-window rate limiter middleware.

Tracks requests per IP with a configurable maximum requests per minute
(default: 60, overridable via ``settings.max_rpm``).  Returns HTTP 429
when the threshold is exceeded.

X-Forwarded-For handling:
    XFF is only trusted when the immediate peer IP is in ``_TRUSTED_PROXIES``.
    This prevents spoofing attacks where an attacker injects a fake XFF
    header to bypass the rate limiter.
"""

import logging
import threading
import time
from collections import defaultdict

from fastapi import HTTPException, Request, status

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── Trusted reverse proxies (add your load balancer IPs here) ───────────
_TRUSTED_PROXIES: frozenset[str] = frozenset({
    "127.0.0.1",
    "::1",
    # Docker default bridge
    "172.17.0.1",
    "172.18.0.1",
})

# ── In-memory sliding window store ──────────────────────────────────────
_window: dict[str, list[float]] = defaultdict(list)
_lock = threading.Lock()

# Window duration in seconds
_WINDOW_SECONDS = 60.0


def _get_client_ip(request: Request) -> str:
    """Extract the real client IP, respecting XFF only from trusted proxies.

    If the direct peer is in ``_TRUSTED_PROXIES`` and an X-Forwarded-For
    header is present, the *leftmost* entry is used (original client IP).
    Otherwise, the direct peer IP is returned.
    """
    peer_ip = request.client.host if request.client else "unknown"

    if peer_ip in _TRUSTED_PROXIES:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # Leftmost = original client
            return xff.split(",")[0].strip()

    return peer_ip


def _prune_window(timestamps: list[float], now: float) -> list[float]:
    """Remove timestamps older than the sliding window."""
    cutoff = now - _WINDOW_SECONDS
    return [t for t in timestamps if t > cutoff]


async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency that enforces the per-IP rate limit.

    Usage in a route::

        from app.routes.rate_limiter import check_rate_limit

        @router.post("", dependencies=[Depends(check_rate_limit)])
        async def my_endpoint():
            ...

    Or as application-wide middleware via ``app.middleware``.
    """
    settings = get_settings()
    max_rpm = settings.max_rpm  # default 60

    client_ip = _get_client_ip(request)
    now = time.monotonic()

    with _lock:
        _window[client_ip] = _prune_window(_window[client_ip], now)

        if len(_window[client_ip]) >= max_rpm:
            logger.warning(
                "Rate limit exceeded for IP %s (%d requests in window)",
                client_ip,
                len(_window[client_ip]),
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {max_rpm} requests per minute.",
                headers={"Retry-After": "60"},
            )

        _window[client_ip].append(now)


def reset_rate_limiter() -> None:
    """Clear all rate-limit state (for testing)."""
    with _lock:
        _window.clear()
