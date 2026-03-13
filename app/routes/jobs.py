"""
OrthoLink Job Polling
GET /api/v1/jobs/{job_id}

Long-running crew runs (DVA, RSA) return job_id immediately; client polls for result.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from app.middleware.auth import AuthenticatedUser, get_current_user
from app.services.job_store import get_job_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}")
async def get_job_status(
    job_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Poll for job result. Returns status (pending | running | completed | failed)
    and result when completed.
    """
    try:
        UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id")

    response = get_job_response(job_id)
    if not response:
        return {
            "status": "NOT_FOUND",
            "message": "Job not found. Server may have restarted. Please resubmit your request.",
        }

    return response
