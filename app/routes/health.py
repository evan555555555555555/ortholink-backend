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
    """FAISS diagnostic — no auth. Returns index stats + test search."""
    diag: dict = {}
    try:
        from app.tools.vector_store import get_vector_store
        store = get_vector_store()
        store._ensure_loaded()
        diag["index_loaded"] = store.index is not None
        diag["index_ntotal"] = store.index.ntotal if store.index else 0
        diag["index_dimension"] = store.index.d if store.index else 0
        diag["use_db"] = store._use_db
        diag["sqlite_count"] = store._metadata_db.count() if store._metadata_db else 0

        # Test: try a search to see if embed_text + FAISS work end-to-end
        if store.index and store.index.ntotal > 0:
            try:
                results = store.search("medical device registration", "US", top_k=3)
                diag["test_search_results"] = len(results)
                if results:
                    diag["test_search_top_score"] = results[0].get("score")
                    diag["test_search_top_reg"] = results[0].get("regulation_name", "")[:80]
                else:
                    diag["test_search_results"] = 0
                    diag["test_search_error"] = "empty results despite index.ntotal > 0"
            except Exception as e:
                diag["test_search_error"] = str(e)[:200]
        else:
            diag["test_search_skipped"] = "index has 0 vectors"
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
