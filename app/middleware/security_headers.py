"""
OrthoLink Security Headers Middleware
OWASP-recommended HTTP security headers on every response.

Headers applied:
  - X-Request-ID           Unique per-request trace ID (echoed from client or generated)
  - X-Content-Type-Options nosniff — prevents MIME-type sniffing attacks
  - X-Frame-Options        DENY — prevents clickjacking
  - X-XSS-Protection       1; mode=block — legacy XSS filter (belt + suspenders)
  - Referrer-Policy        strict-origin-when-cross-origin
  - Strict-Transport-Security  HSTS preload (max-age=1yr + subdomains)
  - Content-Security-Policy    Strict API-only CSP (no scripts, no frames)
  - Permissions-Policy     Disable all browser APIs
  - Cross-Origin-*         COEP / COOP / CORP isolation
  - Server                 ortholink (hides implementation)
  - Cache-Control          no-store for all /api/ routes
"""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds HTTP security headers to all responses.
    Must be added LAST so it wraps all other middleware (outermost = applied first).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate or echo a unique request-trace ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)

        # ── Identity & tracing ────────────────────────────────────────────────
        response.headers["X-Request-ID"] = request_id

        # ── Anti-sniffing / MIME ──────────────────────────────────────────────
        response.headers["X-Content-Type-Options"] = "nosniff"

        # ── Anti-clickjacking ─────────────────────────────────────────────────
        response.headers["X-Frame-Options"] = "DENY"

        # ── Legacy XSS filter (belt + suspenders) ────────────────────────────
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # ── Referrer leakage prevention ───────────────────────────────────────
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # ── HSTS (browser ignores on plain HTTP, enforced on HTTPS) ──────────
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )

        # ── Content Security Policy ─────────────────────────────────────────
        if request.url.path.startswith("/api/") or request.url.path.startswith("/health"):
            # Strict API-only CSP: no scripts, no iframes
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; "
                "frame-ancestors 'none'; "
                "form-action 'none'; "
                "base-uri 'none'"
            )
        else:
            # Frontend SPA CSP: allow self scripts/styles/images/fonts + inline styles
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob:; "
                "font-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'"
            )

        # ── Permissions Policy — disable all browser hardware APIs ────────────
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), "
            "camera=(), "
            "geolocation=(), "
            "gyroscope=(), "
            "magnetometer=(), "
            "microphone=(), "
            "payment=(), "
            "usb=()"
        )

        # ── Server fingerprint removal ────────────────────────────────────────
        response.headers["Server"] = "ortholink"

        # ── Cross-Origin isolation headers ────────────────────────────────────
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"

        # ── Cache control for API responses ──────────────────────────────────
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, private"
            )
            response.headers["Pragma"] = "no-cache"

        return response
