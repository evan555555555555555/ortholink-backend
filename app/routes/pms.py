"""
OrthoLink PMS Route — Post-Market Surveillance Plan Generator
POST /api/v1/pms-plan
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException
from fastapi.responses import JSONResponse

from app.crews.pms_plan import PMSPlan, run_pms_plan
from app.middleware.auth import AuthenticatedUser
from app.middleware.rbac import require_reviewer
from app.services.job_store import create_job, set_completed, set_failed, set_running
from app.services.usage_metering import get_usage_meter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pms-plan", tags=["PMS"])


async def _run_pms_background(
    job_id: str,
    country: str,
    device_class: str,
    device_type: Optional[str],
    org_id: str,
    user_id: str,
) -> None:
    set_running(job_id)
    try:
        plan: PMSPlan = await asyncio.to_thread(
            run_pms_plan,
            country=country,
            device_class=device_class,
            device_type=device_type,
        )
        get_usage_meter().record_usage(org_id=org_id, user_id=user_id, agent_type="pms")
        payload = plan.model_dump()
        payload["job_id"] = job_id
        payload["status"] = "completed"
        set_completed(job_id, payload)
    except Exception as e:
        logger.error(f"PMS job {job_id} failed: {e}", exc_info=True)
        set_failed(job_id, str(e))


@router.post("")
async def pms_plan(
    background_tasks: BackgroundTasks,
    country: str = Form(..., description="Target market country code"),
    device_class: str = Form(..., description="Device class"),
    device_type: Optional[str] = Form(None),
    async_mode: bool = Form(True),
    user: AuthenticatedUser = Depends(require_reviewer),
):
    """
    Post-Market Surveillance Plan Generator.

    Generates a PMS plan per EU MDR Art. 83-86, FDA 21 CFR 803/820.198,
    ISO 13485 §8.2. Includes surveillance activities, reporting deadlines,
    and key data sources.
    """
    meter = get_usage_meter()
    usage = meter.check_trial_limit(user.org_id or "")
    if usage["exceeded"]:
        raise HTTPException(status_code=402, detail="Free trial limit reached.")

    country_code = country.strip().upper()
    if not country_code:
        raise HTTPException(status_code=422, detail="country is required.")

    if async_mode:
        job_id = create_job(agent="pms")
        background_tasks.add_task(
            _run_pms_background,
            job_id, country_code, device_class, device_type, user.org_id or "", user.user_id,
        )
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "pending",
                "country": country_code,
                "message": f"PMS plan generating. Poll GET /api/v1/jobs/{job_id}",
            },
        )

    job_id = create_job(agent="pms")
    await _run_pms_background(
        job_id, country_code, device_class, device_type, user.org_id or "", user.user_id
    )
    from app.services.job_store import get_job_response
    return get_job_response(job_id) or {"status": "failed", "job_id": job_id}
