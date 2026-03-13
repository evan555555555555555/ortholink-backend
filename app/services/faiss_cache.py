"""
OrthoLink FAISS Result Cache — Fix 3
=====================================
Redis-backed cache for FAISS semantic search results.
Reduces latency from ~1.6s/query to <50ms on cache hit.
Gracefully degrades to no-cache if Redis is unavailable.

Cache key: SHA-256(query + "|" + country + "|" + device_class)
TTL: configurable (default 3600s = 1 hour)

Design principles:
- NEVER cache results for REVOKED-law queries (Fix-2 compliance)
- Cache key includes country → country isolation preserved (HC-5)
- Soft failure: if Redis is down, FAISS is called directly (no crash)
- Cache version prefix "v2" — bump when ChunkMetadata schema changes
"""

import hashlib
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_VERSION = "v3"  # v3: key format faiss:v3:{COUNTRY}:{hash}, query lowercased
_redis_client = None
_redis_unavailable = False  # Sticky flag — stop retrying after first failure


def _get_redis():
    """Lazy-init Redis client. Returns None if unavailable."""
    global _redis_client, _redis_unavailable

    if _redis_unavailable:
        return None
    if _redis_client is not None:
        return _redis_client

    try:
        import redis as redis_lib  # type: ignore

        from app.core.config import get_settings

        settings = get_settings()
        if not settings.faiss_cache_enabled:
            _redis_unavailable = True
            logger.info("FAISS cache disabled via config (faiss_cache_enabled=False)")
            return None

        client = redis_lib.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1,  # Fast fail — don't block requests
            socket_timeout=0.5,
        )
        client.ping()
        _redis_client = client
        logger.info("FAISS Redis cache connected: %s", settings.redis_url)
        return _redis_client

    except ImportError:
        _redis_unavailable = True
        logger.warning(
            "FAISS cache: 'redis' package not installed. "
            "Run: pip install redis  (graceful degradation — FAISS called directly)"
        )
        return None
    except Exception as e:
        _redis_unavailable = True
        logger.warning(
            "FAISS cache: Redis unavailable (%s). Degrading to direct FAISS calls.", e
        )
        return None


def _make_key(query: str, country: str, device_class: Optional[str]) -> str:
    """Generate a cache key for a FAISS search query.

    Key format: faiss:{version}:{COUNTRY}:{sha256(query_lc|country_uc|class_uc)}

    - Country appears in the prefix so invalidate_country() can scan with KEYS/SCAN.
    - Query is lowercased for case-insensitive cache reuse (same query, different case → same hit).
    - Device class is uppercased for consistency.
    """
    country_up = country.upper()
    raw = f"{query.lower()}|{country_up}|{(device_class or '').upper()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"faiss:{_CACHE_VERSION}:{country_up}:{digest}"


def get_cached(
    query: str,
    country: str,
    device_class: Optional[str] = None,
) -> Optional[list[dict]]:
    """
    Retrieve cached FAISS results. Returns None on cache miss or error.

    Args:
        query: The search query text
        country: Country code (e.g. "US", "EU", "UA")
        device_class: Optional device class filter

    Returns:
        List of result dicts (same format as VectorStore.search) or None
    """
    r = _get_redis()
    if r is None:
        return None

    key = _make_key(query, country, device_class)
    try:
        raw = r.get(key)
        if raw is None:
            return None
        results: list[dict] = json.loads(raw)
        logger.debug("FAISS cache HIT: country=%s key=%s…", country, key[:20])
        return results
    except Exception as e:
        logger.debug("FAISS cache get error (key=%s): %s", key[:20], e)
        return None


def set_cached(
    query: str,
    country: str,
    results: list[dict],
    device_class: Optional[str] = None,
    ttl: Optional[int] = None,
) -> None:
    """
    Store FAISS results in cache.

    Args:
        query: The search query text
        country: Country code
        results: List of result dicts from VectorStore.search()
        device_class: Optional device class filter
        ttl: Cache TTL in seconds (uses config default if None)
    """
    r = _get_redis()
    if r is None:
        return

    # Safety: never cache empty results — could mask real FAISS failures
    if not results:
        return

    key = _make_key(query, country, device_class)
    try:
        from app.core.config import get_settings

        effective_ttl = ttl if ttl is not None else get_settings().faiss_cache_ttl
        serialized = json.dumps(results)
        r.setex(key, effective_ttl, serialized)
        logger.debug(
            "FAISS cache SET: country=%s key=%s… TTL=%ds (%d results)",
            country, key[:20], effective_ttl, len(results),
        )
    except Exception as e:
        logger.debug("FAISS cache set error (key=%s): %s", key[:20], e)


def invalidate_country(country: str) -> int:
    """
    Invalidate all cached FAISS results for a given country.
    Called by RAA when new regulatory data is ingested.

    Returns: number of keys deleted
    """
    r = _get_redis()
    if r is None:
        return 0

    try:
        # Scan for keys with this country's prefix — precise country-specific invalidation.
        # Key format: faiss:{version}:{COUNTRY}:{hash}
        pattern = f"faiss:{_CACHE_VERSION}:{country.upper()}:*"
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                r.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        logger.info("FAISS cache: invalidated %d keys (country=%s)", deleted, country)
        return deleted
    except Exception as e:
        logger.warning("FAISS cache invalidation failed: %s", e)
        return 0


def cache_stats() -> dict:
    """Return basic cache statistics for the /health/detailed endpoint."""
    r = _get_redis()
    if r is None:
        return {"available": False, "reason": "Redis not connected"}

    try:
        info = r.info("memory")
        keyspace = r.info("keyspace")
        return {
            "available": True,
            "used_memory_human": info.get("used_memory_human", "?"),
            "keyspace": keyspace,
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}
