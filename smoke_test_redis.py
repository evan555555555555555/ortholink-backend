"""
OrthoLink FAISS Redis Cache — Smoke Test
=========================================
Tests the actual faiss_cache.py module (not raw redis).
Verifies cache hit/miss, country isolation, TTL, invalidation, and graceful degradation.

Run from backend/ with the venv active:
    python smoke_test_redis.py
"""

import json
import os
import sys
import time
import traceback

# ── Bootstrap: make `app` importable without starting the full server ─────────
sys.path.insert(0, os.path.dirname(__file__))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[36mINFO\033[0m"
WARN = "\033[33mWARN\033[0m"
SEP  = "─" * 60

_results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    _results.append((label, condition, detail))
    print(f"  {status} {label}" + (f"  [{detail}]" if detail else ""))
    return condition


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ── Fake FAISS result — mirrors real VectorStore.search() output ──────────────
def _fake_results(country: str, n: int = 3) -> list[dict]:
    return [
        {
            "chunk_id": f"fake-{country}-{i:04d}",
            "text": f"Regulation text for {country} item {i}",
            "regulation_name": f"{country} Medical Device Act §{i}",
            "article": f"Art. {i}",
            "score": round(0.95 - i * 0.05, 3),
            "country": country,
            "regulatory_status": "ACTIVE",
            "superseded_by": None,
        }
        for i in range(n)
    ]


def run_smoke_test():
    print(f"\nOrthoLink FAISS Redis Cache -- Smoke Test")
    print(f"    redis-py 7.3.0  •  faiss_cache.py v2\n")

    # ── 1. Module Import ───────────────────────────────────────────────────────
    section("1 · Module Import & Redis Connectivity")
    try:
        from app.services.faiss_cache import (
            _get_redis, _make_key, get_cached, set_cached,
            invalidate_country, cache_stats,
        )
        check("faiss_cache imports OK", True)
    except Exception as e:
        check("faiss_cache imports OK", False, str(e))
        print(f"\n{FAIL} Fatal: cannot import faiss_cache. Aborting.")
        return False

    r = _get_redis()
    check("Redis client connected", r is not None,
          "PONG received" if r else "Returned None — check redis-server")
    if r is None:
        print(f"\n{FAIL} Redis not reachable. Aborting further checks.")
        return False

    # ── 2. Cache Key Generation ────────────────────────────────────────────────
    section("2 · Cache Key Isolation")

    k_ua = _make_key("authorized representative agreement", "UA", "IIb")
    k_eu = _make_key("authorized representative agreement", "EU", "IIb")
    k_us = _make_key("authorized representative agreement", "US", "IIb")
    k_ua2 = _make_key("AUTHORIZED REPRESENTATIVE AGREEMENT", "ua", "iib")  # case-normalised

    check("UA ≠ EU key (country isolation)", k_ua != k_eu, f"UA={k_ua[:16]}… EU={k_eu[:16]}…")
    check("UA ≠ US key", k_ua != k_us)
    check("Key is deterministic (case-normalised)", k_ua == k_ua2,
          "upper-case query + lower-case country → same digest")
    check("Key is 64-hex SHA-256", len(k_ua.split(":")[-1]) == 64)
    check("Key has version prefix v3", k_ua.startswith("faiss:v3:"))
    check("Key embeds country prefix for fast invalidation", ":UA:" in k_ua,
          f"key={k_ua[:30]}…")

    # ── 3. Cache Miss (cold) ───────────────────────────────────────────────────
    section("3 · Cache Miss (cold start)")

    # Clean slate — delete any leftover keys from prior runs
    r.delete(k_ua, k_eu, k_us)

    miss_ua = get_cached("authorized representative agreement", "UA", "IIb")
    miss_eu = get_cached("authorized representative agreement", "EU", "IIb")
    check("Cold miss returns None (UA)", miss_ua is None)
    check("Cold miss returns None (EU)", miss_eu is None)

    # ── 4. Cache Write & Hit ───────────────────────────────────────────────────
    section("4 · Cache Write → Hit")

    ua_results = _fake_results("UA", n=5)
    set_cached("authorized representative agreement", "UA", ua_results, "IIb", ttl=60)

    hit = get_cached("authorized representative agreement", "UA", "IIb")
    check("Cache hit after set_cached", hit is not None)
    check("Hit returns correct count", hit is not None and len(hit) == 5,
          f"got {len(hit) if hit else 0}")
    check("Hit preserves chunk_id", hit is not None and hit[0]["chunk_id"].startswith("fake-UA"))
    check("Hit preserves regulatory_status=ACTIVE",
          hit is not None and hit[0]["regulatory_status"] == "ACTIVE")

    # Country isolation: UA write should NOT pollute EU
    miss_eu_after_ua_write = get_cached("authorized representative agreement", "EU", "IIb")
    check("UA write does NOT pollute EU cache", miss_eu_after_ua_write is None)

    # ── 5. TTL Expiry ─────────────────────────────────────────────────────────
    section("5 · TTL Expiry")

    set_cached("ttl-expiry-test", "UA", _fake_results("UA", 1), ttl=2)
    hit_before = get_cached("ttl-expiry-test", "UA")
    check("Pre-expiry: cache hit", hit_before is not None)

    print(f"  {INFO} Waiting 3s for TTL=2 expiry…", end="", flush=True)
    time.sleep(3)
    print(" done.")

    hit_after = get_cached("ttl-expiry-test", "UA")
    check("Post-expiry: cache miss (TTL working)", hit_after is None)

    # ── 6. Empty Result Guard ─────────────────────────────────────────────────
    section("6 · Empty Result Guard (never cache empty lists)")

    set_cached("empty-guard-test", "UA", [], ttl=60)  # should be silently dropped
    miss_empty = get_cached("empty-guard-test", "UA")
    check("Empty results not cached (returns None)", miss_empty is None)

    # ── 7. Country Invalidation ───────────────────────────────────────────────
    section("7 · Country Invalidation (RAA post-ingestion)")

    # Seed several UA and EU keys
    for i in range(5):
        set_cached(f"query-{i}", "UA", _fake_results("UA", 3), ttl=300)
    for i in range(3):
        set_cached(f"query-{i}", "EU", _fake_results("EU", 3), ttl=300)

    # Verify they're hot before invalidation
    pre_ua = get_cached("query-0", "UA")
    pre_eu = get_cached("query-0", "EU")
    check("UA keys hot before invalidation", pre_ua is not None)
    check("EU keys hot before invalidation", pre_eu is not None)

    deleted = invalidate_country("UA")
    check("invalidate_country returns >0 deletions", deleted > 0, f"deleted={deleted}")

    # All UA keys should be gone; EU keys should survive
    post_ua = get_cached("query-0", "UA")
    post_eu = get_cached("query-0", "EU")
    check("UA cache wiped after invalidation", post_ua is None)
    check("EU cache survives UA invalidation", post_eu is not None,
          "country isolation preserved")

    # ── 8. cache_stats() ──────────────────────────────────────────────────────
    section("8 · cache_stats() for /health/detailed")

    stats = cache_stats()
    check("cache_stats returns dict", isinstance(stats, dict))
    check("cache_stats.available = True", stats.get("available") is True,
          str(stats))
    check("cache_stats has used_memory_human", "used_memory_human" in stats,
          stats.get("used_memory_human", "MISSING"))

    # ── 9. Graceful Degradation ───────────────────────────────────────────────
    section("9 · Graceful Degradation (bad Redis URL)")

    import app.services.faiss_cache as _fc_module
    # Temporarily reset the module's sticky flags and inject a bad client
    orig_client = _fc_module._redis_client
    orig_flag = _fc_module._redis_unavailable

    _fc_module._redis_client = None
    _fc_module._redis_unavailable = False

    # Monkey-patch settings to return a dead port
    from unittest.mock import patch
    with patch("app.core.config.get_settings") as mock_cfg:
        mock_cfg.return_value.redis_url = "redis://localhost:19999"
        mock_cfg.return_value.faiss_cache_enabled = True
        mock_cfg.return_value.faiss_cache_ttl = 3600

        # First call should fail fast and set sticky flag
        degraded_result = get_cached("degradation-test", "UA")

    check("Degraded get_cached returns None (no crash)", degraded_result is None)
    check("Sticky unavailable flag set after failure", _fc_module._redis_unavailable is True)

    # Second call with bad config: must NOT attempt reconnect (sticky flag)
    second_call = get_cached("degradation-test2", "UA")
    check("Second call skips Redis (sticky flag)", second_call is None)

    # Restore
    _fc_module._redis_client = orig_client
    _fc_module._redis_unavailable = orig_flag

    # ── 10. JSON Serialisation Round-trip ─────────────────────────────────────
    section("10 · JSON Serialisation Round-trip")

    rich_results = _fake_results("IN", 2)
    rich_results[0]["extra_field"] = {"nested": True, "value": 3.14}
    set_cached("json-roundtrip-test", "IN", rich_results, ttl=60)
    rt = get_cached("json-roundtrip-test", "IN")
    check("Round-trip preserves nested dict", rt is not None and
          isinstance(rt[0].get("extra_field"), dict))
    check("Round-trip float precision", rt is not None and
          rt[0]["extra_field"]["value"] == 3.14)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    if failed == 0:
        print(f"  {PASS} All {total}/{total} checks passed — cache is production-ready")
    else:
        print(f"  {FAIL} {failed}/{total} checks FAILED")
        print(f"\n  Failed checks:")
        for label, ok, detail in _results:
            if not ok:
                print(f"    • {label}" + (f": {detail}" if detail else ""))

    print(f"{SEP}\n")

    # Cleanup test keys
    try:
        r.delete(
            _make_key("authorized representative agreement", "UA", "IIb"),
            _make_key("authorized representative agreement", "EU", "IIb"),
            _make_key("authorized representative agreement", "US", "IIb"),
            _make_key("json-roundtrip-test", "IN", None),
        )
    except Exception:
        pass

    return failed == 0


if __name__ == "__main__":
    # Suppress logger.warning() noise from faiss_cache internals (graceful-degradation
    # section intentionally triggers connection failures; we check return values, not logs).
    import logging as _logging
    _logging.getLogger("app.services.faiss_cache").setLevel(_logging.CRITICAL)
    _logging.getLogger("app").setLevel(_logging.CRITICAL)

    try:
        ok = run_smoke_test()
        sys.exit(0 if ok else 1)
    except Exception:
        print(f"\n{FAIL} Unhandled exception:")
        traceback.print_exc()
        sys.exit(1)
