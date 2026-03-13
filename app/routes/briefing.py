"""
Briefing Routes — Daily Regulatory Intelligence Brief + Coverage Audit

GET  /api/v1/briefing/latest          → most recent daily brief
GET  /api/v1/briefing/coverage        → latest FAISS coverage audit
POST /api/v1/briefing/run             → trigger on-demand (no scheduler needed)
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.middleware.auth import AuthenticatedUser, get_current_user
from app.services.job_store import get_latest_job, create_job, set_completed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/briefing", tags=["Briefing"])


@router.get("/latest")
async def get_latest_brief(user: AuthenticatedUser = Depends(get_current_user)):
    """Return the most recently generated daily regulatory intelligence brief."""
    job = get_latest_job("daily_brief")
    if not job:
        raise HTTPException(
            status_code=404,
            detail="No daily brief generated yet. The brief runs automatically every 24h, or POST /api/v1/briefing/run to trigger now.",
        )
    return job["result"]


@router.get("/coverage")
async def get_coverage_audit(user: AuthenticatedUser = Depends(get_current_user)):
    """Return the latest FAISS coverage audit (Reality Checker verdict)."""
    job = get_latest_job("daily_brief")
    if not job or not job.get("result", {}).get("coverage_audit"):
        # Run a quick coverage-only check on demand
        from app.services.daily_brief import _audit_coverage
        try:
            coverage = _audit_coverage()
            return coverage
        except Exception as e:
            logger.error("Coverage audit failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Coverage audit failed. Please try again.")
    return job["result"]["coverage_audit"]


@router.post("/run")
async def run_brief_now(user: AuthenticatedUser = Depends(get_current_user)):
    """
    Trigger an on-demand daily brief immediately.
    Returns the brief result synchronously (runs in ~1-2s, no LLM required).
    """
    from app.services.daily_brief import generate_daily_brief
    try:
        brief = generate_daily_brief()

        # Sign it
        try:
            from app.services.crypto_signer import sign_payload
            brief = sign_payload(brief)
        except Exception as exc:
            logger.warning("Briefing crypto-sign failed: %s", exc)

        # Store so /latest also picks it up
        job_id = create_job(agent="daily_brief")
        set_completed(job_id, brief)

        return brief
    except Exception as e:
        logger.error("on-demand brief failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Brief generation failed. Please try again.")
