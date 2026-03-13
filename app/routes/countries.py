"""
OrthoLink Countries (public)
GET /api/v1/countries — list of supported countries in the vector store.
"""

from fastapi import APIRouter

from app.tools.vector_store import get_vector_store

router = APIRouter(prefix="/countries", tags=["countries"])


@router.get("")
async def get_countries():
    """Return list of countries available in the vector store. No auth required for discovery."""
    store = get_vector_store()
    countries = store.get_countries()
    return {
        "countries": sorted(countries),
        "count": len(countries),
    }
