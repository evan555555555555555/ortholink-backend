"""
OrthoLink System Status Dashboard — Backend-Only Aggregator
GET /api/v1/dashboard/system-status

Single endpoint that aggregates all telemetry into one JSON payload:
  - Redis v3 cache health (from faiss_cache.cache_stats)
  - FAISS vector coverage (from daily_brief._audit_coverage)
  - HMAC integrity engine status (from crypto_signer + integrity_guard)
  - Global health verdict

Requires JWT authentication — this is sensitive operational telemetry.
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends

from app.middleware.auth import AuthenticatedUser, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _collect_cache_health() -> dict[str, Any]:
    """Redis v3 cache metrics.  Graceful: returns degraded status on failure."""
    try:
        from app.services.faiss_cache import cache_stats

        stats = cache_stats()
        return {
            "status": "online" if stats.get("available") else "offline",
            "version": "v3",
            "key_format": "faiss:v3:{COUNTRY}:{SHA256}",
            "used_memory": stats.get("used_memory_human", "unknown"),
            "keyspace": stats.get("keyspace", {}),
        }
    except Exception as exc:
        logger.debug("Cache health check failed: %s", exc)
        return {"status": "offline", "version": "v3", "error": "unavailable"}


def _collect_faiss_coverage() -> dict[str, Any]:
    """FAISS vector store coverage audit (Reality Checker pattern)."""
    try:
        from app.services.daily_brief import _audit_coverage

        coverage = _audit_coverage()
        return {
            "status": "healthy" if coverage.get("verdict") == "COVERAGE_OK" else "degraded",
            "verdict": coverage.get("verdict", "UNKNOWN"),
            "total_countries": coverage.get("total_countries", 0),
            "healthy_count": coverage.get("healthy_count", 0),
            "low_coverage_count": coverage.get("low_coverage_count", 0),
            "critical_gap_count": coverage.get("critical_gap_count", 0),
            "total_chunks": sum(coverage.get("chunk_counts", {}).values()),
            "chunk_counts": coverage.get("chunk_counts", {}),
            "critical_gaps": [g["country"] for g in coverage.get("critical_gaps", [])],
            "healthy_countries": coverage.get("healthy_countries", []),
        }
    except Exception as exc:
        logger.debug("FAISS coverage check failed: %s", exc)
        return {"status": "offline", "verdict": "UNKNOWN", "total_chunks": 0}


def _collect_integrity_status() -> dict[str, Any]:
    """HMAC crypto signer + IntegrityGuard status."""
    try:
        from app.tools.vector_store import get_vector_store

        store = get_vector_store()
        store._ensure_loaded()

        chunk_count = store.get_chunk_count()
        countries = set(store.get_countries())

        # Verify signer is functional by checking secret availability
        signer_ok = False
        try:
            from app.services.crypto_signer import _get_secret

            _get_secret()
            signer_ok = True
        except Exception:
            pass

        return {
            "status": "online" if signer_ok else "degraded",
            "crypto_signing": signer_ok,
            "algorithm": "HMAC-SHA256",
            "auto_fact_check": True,
            "faiss_backed": True,
            "regulatory_chunks": chunk_count,
            "countries_covered": len(countries),
        }
    except Exception as exc:
        logger.debug("Integrity status check failed: %s", exc)
        return {"status": "offline", "crypto_signing": False}


@router.get("/system-status")
def system_status(
    user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Real-time compliance & health aggregator.

    Returns the complete defensive architecture status in one payload:
    Redis v3 cache, FAISS coverage, HMAC integrity, and a global health verdict.

    Global health is True ONLY when all three subsystems are online.
    """
    t0 = time.monotonic()

    cache = _collect_cache_health()
    faiss = _collect_faiss_coverage()
    integrity = _collect_integrity_status()

    is_cache_online = cache.get("status") == "online"
    is_faiss_healthy = faiss.get("status") == "healthy"
    is_integrity_online = integrity.get("status") == "online"

    global_health = all([is_cache_online, is_faiss_healthy, is_integrity_online])

    latency_ms = round((time.monotonic() - t0) * 1000, 2)

    return {
        "global_health": global_health,
        "global_verdict": "ALL_SYSTEMS_NOMINAL" if global_health else "REVIEW_REQUIRED",
        "latency_ms": latency_ms,
        "timestamp": int(time.time()),
        "components": {
            "redis_v3_cache": cache,
            "faiss_vector_store": faiss,
            "hmac_integrity_engine": integrity,
        },
        "defensive_architecture": {
            "cache_key_format": "faiss:v3:{COUNTRY}:{SHA256}",
            "country_isolation": True,
            "revoked_law_filter": True,
            "static_fallback_floor": True,
            "cross_country_contamination_blocked": True,
        },
        "test_suite": "run pytest for live counts",
    }
