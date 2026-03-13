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
    try:
        from app.tools.vector_store import get_vector_store
        chunk_count = get_vector_store().get_chunk_count()
    except Exception:
        pass
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "agents": _AGENT_COUNT,
        "chunks": chunk_count,
    }


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
