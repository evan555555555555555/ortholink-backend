"""
OrthoLink Health Check Routes
"""
import logging

from fastapi import APIRouter, Depends

from app.core.config import get_settings
from app.middleware.auth import AuthenticatedUser, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])
_AGENT_COUNT = 12


@router.get("/health")
async def health_check():
    """Basic health check — no auth required."""
    settings = get_settings()
    chunk_count = 0
    faiss_vectors = 0
    try:
        from app.tools.vector_store import get_vector_store
        store = get_vector_store()
        chunk_count = store.get_chunk_count()
        if store.index is not None:
            faiss_vectors = store.index.ntotal
    except Exception:
        pass
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "agents": _AGENT_COUNT,
        "chunks": chunk_count,
        "faiss_vectors": faiss_vectors,
    }


@router.get("/health/faiss-debug")
async def faiss_debug():
    """FAISS diagnostic — no auth. Tests each step of search pipeline."""
    import numpy as np
    diag: dict = {}
    try:
        from app.tools.vector_store import get_vector_store
        import faiss as faiss_lib
        store = get_vector_store()
        store._ensure_loaded()
        diag["index_loaded"] = store.index is not None
        diag["index_ntotal"] = store.index.ntotal if store.index else 0
        diag["index_dimension"] = store.index.d if store.index else 0
        diag["use_db"] = store._use_db
        diag["sqlite_count"] = store._metadata_db.count() if store._metadata_db else 0

        if not store.index or store.index.ntotal == 0:
            diag["abort"] = "index has 0 vectors"
            return diag

        # Step 1: Test embed_text
        try:
            from app.tools.embeddings import embed_text
            emb = embed_text("medical device registration")
            diag["embed_ok"] = True
            diag["embed_shape"] = list(emb.shape)
            diag["embed_dtype"] = str(emb.dtype)
            diag["embed_norm"] = float(np.linalg.norm(emb))
        except Exception as e:
            diag["embed_ok"] = False
            diag["embed_error"] = str(e)[:300]
            return diag

        # Step 2: Raw FAISS search (no country filter)
        try:
            qvec = emb.reshape(1, -1).copy()
            faiss_lib.normalize_L2(qvec)
            scores, indices = store.index.search(qvec, 10)
            raw_indices = [int(idx) for idx in indices[0] if idx != -1]
            raw_scores = [float(s) for s, idx in zip(scores[0], indices[0]) if idx != -1]
            diag["raw_faiss_hits"] = len(raw_indices)
            diag["raw_faiss_top_indices"] = raw_indices[:5]
            diag["raw_faiss_top_scores"] = raw_scores[:5]
        except Exception as e:
            diag["raw_faiss_error"] = str(e)[:300]
            return diag

        # Step 3: Check metadata for those raw hits
        if raw_indices:
            try:
                countries_found = []
                for idx in raw_indices[:5]:
                    chunk = store._get_chunk(idx)
                    if chunk:
                        countries_found.append(chunk.country)
                    else:
                        countries_found.append(f"MISSING_IDX_{idx}")
                diag["raw_hit_countries"] = countries_found
            except Exception as e:
                diag["metadata_lookup_error"] = str(e)[:200]

        # Step 4: Large raw FAISS search — count countries in top 1000
        try:
            qvec2 = emb.reshape(1, -1).copy()
            faiss_lib.normalize_L2(qvec2)
            scores2, indices2 = store.index.search(qvec2, 1000)
            country_counts: dict = {}
            for idx in indices2[0]:
                if idx == -1:
                    continue
                chunk = store._get_chunk(int(idx))
                if chunk:
                    c = chunk.country
                    country_counts[c] = country_counts.get(c, 0) + 1
                else:
                    country_counts["MISSING"] = country_counts.get("MISSING", 0) + 1
            diag["top1000_country_distribution"] = dict(sorted(country_counts.items(), key=lambda x: -x[1])[:10])
            diag["top1000_US_count"] = country_counts.get("US", 0)
        except Exception as e:
            diag["top1000_error"] = str(e)[:300]

        # Step 5: Redis cache status
        try:
            from app.services.faiss_cache import _get_redis, get_cached
            r = _get_redis()
            diag["redis_available"] = r is not None
            if r is not None:
                # Check if there's a cached result for our test query
                cached = get_cached("medical device registration", "US")
                diag["redis_cached_test_query"] = cached is not None
                if cached is not None:
                    diag["redis_cached_result_len"] = len(cached)
        except Exception as e:
            diag["redis_error"] = str(e)[:200]

        # Step 6: Manual full pipeline using pre-computed embedding (same as step 1)
        try:
            qvec3 = emb.reshape(1, -1).copy()
            faiss_lib.normalize_L2(qvec3)
            # Use same oversample as store.search for US
            search_k_full = min(12000, store.index.ntotal)
            scores3, indices3 = store.index.search(qvec3, search_k_full)
            manual_results = []
            skipped_none = 0
            skipped_country = 0
            skipped_inactive = 0
            skipped_revoked = 0
            for sc, ix in zip(scores3[0], indices3[0]):
                if ix == -1:
                    continue
                chunk = store._get_chunk(int(ix))
                if chunk is None:
                    skipped_none += 1
                    continue
                if chunk.country.upper() != "US":
                    skipped_country += 1
                    continue
                if not chunk.is_active:
                    skipped_inactive += 1
                    continue
                if chunk.regulatory_status in ("REVOKED", "SUPERSEDED"):
                    skipped_revoked += 1
                    continue
                manual_results.append({
                    "score": float(sc),
                    "regulation": chunk.regulation_name[:60],
                    "article": chunk.article[:40] if chunk.article else "",
                })
                if len(manual_results) >= 5:
                    break
            diag["manual_search_results"] = len(manual_results)
            diag["manual_search_skipped"] = {
                "none_chunk": skipped_none,
                "wrong_country": skipped_country,
                "inactive": skipped_inactive,
                "revoked": skipped_revoked,
            }
            if manual_results:
                diag["manual_search_top3"] = manual_results[:3]
        except Exception as e:
            diag["manual_search_error"] = str(e)[:300]

        # Step 7: Second embed_text call to test if it fails
        try:
            emb2 = embed_text("medical device registration")
            diag["embed2_ok"] = True
            diag["embed2_norm"] = float(np.linalg.norm(emb2))
            # Check if embeddings are identical
            diag["embed_match"] = bool(np.allclose(emb, emb2, atol=1e-5))
        except Exception as e:
            diag["embed2_ok"] = False
            diag["embed2_error"] = str(e)[:500]

        # Step 8: Test store.search() (calls embed_text internally — 3rd call)
        try:
            results = store.search("medical device registration", "US", top_k=3)
            diag["full_search_results"] = len(results)
            if results:
                diag["full_search_top"] = {
                    "score": results[0].get("score"),
                    "regulation": results[0].get("regulation_name", "")[:80],
                    "country": results[0].get("country"),
                }
        except Exception as e:
            diag["full_search_error"] = str(e)[:500]

    except Exception as e:
        diag["load_error"] = str(e)[:200]
    return diag


@router.get("/health/detailed")
async def detailed_health_check(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Detailed health check — requires authentication."""
    settings = get_settings()
    checks: dict = {
        "app": True,
        "supabase_configured": bool(settings.supabase_url and settings.supabase_anon_key),
        "openai_configured": bool(settings.openai_api_key),
        "sentry_configured": bool(settings.sentry_dsn),
    }
    try:
        from app.tools.vector_store import get_vector_store
        store = get_vector_store()
        checks["vector_store"] = True
        checks["vector_store_chunks"] = store.get_chunk_count()
        checks["vector_store_countries"] = store.get_countries()
    except Exception as e:
        logger.warning("Health check: vector store error: %s", e)
        checks["vector_store"] = False
        checks["vector_store_error"] = "unavailable"
    all_healthy = all(v for k, v in checks.items() if isinstance(v, bool))
    return {"status": "ok" if all_healthy else "degraded", "checks": checks}
