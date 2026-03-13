"""
OrthoLink History Routes
GET /api/v1/history — Paginated analysis history for the current user/org.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.middleware.auth import AuthenticatedUser, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["history"])


class HistoryItem(BaseModel):
    """Single analysis history entry."""
    id: str
    agent: str  # dva | cra | roa | rsa | raa
    country: str
    summary: str = ""
    status: str = "completed"
    created_at: str
    updated_at: Optional[str] = None


class HistoryResponse(BaseModel):
    """Paginated history response."""
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int
    has_more: bool


@router.get("", response_model=HistoryResponse)
async def get_history(
    agent: Optional[str] = Query(None),
    country: Optional[str] = Query(None, max_length=4, pattern=r"^[A-Za-z]{2,4}$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Return paginated analysis history for the authenticated user's org.

    PRD: GET /api/v1/history | JWT | All | ?agent=&country=&page=
    """
    try:
        from app.services.supabase_client import get_supabase_client

        client = get_supabase_client()

        query_builder = (
            client.table("analysis_results")
            .select("*", count="exact")
            .eq("org_id", user.org_id)
            .order("created_at", desc=True)
        )

        if agent:
            query_builder = query_builder.eq("agent", agent)
        if country:
            query_builder = query_builder.ilike("country", f"%{country}%")

        offset = (page - 1) * page_size
        query_builder = query_builder.range(offset, offset + page_size - 1)

        result = query_builder.execute()
        total = result.count if result.count is not None else len(result.data or [])
        items = []

        for row in (result.data or []):
            items.append(HistoryItem(
                id=row.get("id", ""),
                agent=row.get("agent", ""),
                country=row.get("country", ""),
                summary=row.get("summary", ""),
                status=row.get("status", "completed"),
                created_at=row.get("created_at", ""),
                updated_at=row.get("updated_at"),
            ))

        return HistoryResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            has_more=(offset + page_size) < total,
        )

    except Exception as e:
        logger.error(f"Failed to fetch history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch history. Please try again.")
