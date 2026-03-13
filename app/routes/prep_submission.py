"""
OrthoLink ROA — Full Market Entry / Submission Prep Route
POST /api/v1/prep-submission

Generates a full market entry plan for a country + device, including:
  - Phased timeline (pre-submission → registration → PMS)
  - Technical File section checklist
  - Clinical evaluation and PMS requirements
  - Apostille / notarisation requirements
  - Pre-submission meeting recommendation

Supports async_mode (default True): returns 202 + job_id.
Poll GET /api/v1/jobs/{job_id} for the completed FullMarketEntryPlan.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException
from fastapi.responses import JSONResponse

from app.crews.full_market_entry import run_full_market_entry
from app.middleware.auth import AuthenticatedUser
from app.middleware.rbac import require_reviewer
from app.services.job_store import create_job, set_completed, set_failed, set_running
from app.services.usage_metering import get_usage_meter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prep-submission", tags=["ROA"])


async def _run_prep_background(
    job_id: str,
    country: str,
    device_class: str,
    device_type: Optional[str],
    org_id: str,
    user_id: str,
) -> None:
    """Background task: run full market entry crew and store FullMarketEntryPlan."""
    set_running(job_id)
    try:
        plan = await asyncio.to_thread(
            run_full_market_entry,
            country=country,
            device_class=device_class,
            device_type=device_type,
        )
        get_usage_meter().record_usage(org_id=org_id, user_id=user_id, agent_type="roa_prep")
        payload = plan.model_dump()
        payload["job_id"] = job_id
        payload["status"] = "completed"
        set_completed(job_id, payload)
    except Exception as e:
        logger.error(f"prep-submission job {job_id} failed: {e}", exc_info=True)
        set_failed(job_id, str(e))


@router.post("")
async def prep_submission(
    background_tasks: BackgroundTasks,
    country: str = Form(..., description="Target market country code (e.g. US, UA, IN)"),
    device_class: str = Form(..., description="Device class (e.g. IIb, III)"),
    device_type: Optional[str] = Form(None, description="Device type (e.g. orthopedic_implant)"),
    async_mode: bool = Form(True, description="True (default): 202 + job_id. False: synchronous."),
    user: AuthenticatedUser = Depends(require_reviewer),
):
    """
    Full Market Entry Plan — PRD ROA extended.

    Returns a phased timeline and complete submission checklist for entering
    a market with a specific device class. Includes clinical evaluation and
    PMS requirements, apostille flags, and pre-submission meeting guidance.
    """
    meter = get_usage_meter()
    usage = meter.check_trial_limit(user.org_id or "")
    if usage["exceeded"]:
        raise HTTPException(status_code=402, detail="Free trial limit reached.")

    country_code = country.strip().upper()
    if not country_code:
        raise HTTPException(status_code=422, detail="country is required.")

    if async_mode:
        job_id = create_job(agent="roa_prep")
        background_tasks.add_task(
            _run_prep_background,
            job_id,
            country_code,
            device_class,
            device_type,
            user.org_id or "",
            user.user_id,
        )
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "pending",
                "country": country_code,
                "message": f"Submission prep running. Poll GET /api/v1/jobs/{job_id} for result.",
            },
        )

    # Synchronous path
    job_id = create_job(agent="roa_prep")
    await _run_prep_background(job_id, country_code, device_class, device_type, user.org_id or "", user.user_id)
    from app.services.job_store import get_job_response
    response = get_job_response(job_id)
    return response or {"status": "failed", "job_id": job_id}
