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

        # Step 4: Full search with country filter
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
            diag["full_search_error"] = str(e)[:300]

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
