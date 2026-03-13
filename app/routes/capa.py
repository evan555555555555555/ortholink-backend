"""
OrthoLink CAPA Route — Corrective and Preventive Action Analysis
POST /api/v1/capa
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException
from fastapi.responses import JSONResponse

from app.crews.capa_analysis import CAPAAnalysis, run_capa_analysis
from app.middleware.auth import AuthenticatedUser
from app.middleware.rbac import require_reviewer
from app.services.job_store import create_job, set_completed, set_failed, set_running
from app.services.usage_metering import get_usage_meter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/capa", tags=["CAPA"])


async def _run_capa_background(
    job_id: str,
    problem_statement: str,
    country: str,
    device_class: str,
    device_type: Optional[str],
    org_id: str,
    user_id: str,
) -> None:
    set_running(job_id)
    try:
        analysis: CAPAAnalysis = await asyncio.to_thread(
            run_capa_analysis,
            problem_statement=problem_statement,
            country=country,
            device_class=device_class,
            device_type=device_type,
        )
        get_usage_meter().record_usage(org_id=org_id, user_id=user_id, agent_type="capa")
        payload = analysis.model_dump()
        payload["job_id"] = job_id
        payload["status"] = "completed"
        set_completed(job_id, payload)
    except Exception as e:
        logger.error(f"CAPA job {job_id} failed: {e}", exc_info=True)
        set_failed(job_id, str(e))


@router.post("")
async def capa_analysis(
    background_tasks: BackgroundTasks,
    problem_statement: str = Form(
        ...,
        max_length=2000,
        description="Description of the nonconformity, complaint, or adverse event",
    ),
    country: str = Form(..., description="Country code for regulatory context"),
    device_class: str = Form(..., description="Device class"),
    device_type: Optional[str] = Form(None),
    async_mode: bool = Form(True),
    user: AuthenticatedUser = Depends(require_reviewer),
):
    """
    CAPA Analysis Agent — Corrective and Preventive Action.

    Generates root cause categories, investigation questions, corrective actions,
    and maps each to regulatory obligations (ISO 13485 §8.5, 21 CFR 820.100).
    Flags whether regulatory notification (MDR, FSCA) is required.
    """
    meter = get_usage_meter()
    usage = meter.check_trial_limit(user.org_id or "")
    if usage["exceeded"]:
        raise HTTPException(status_code=402, detail="Free trial limit reached.")

    country_code = country.strip().upper()
    if not problem_statement.strip():
        raise HTTPException(status_code=422, detail="problem_statement is required.")

    if async_mode:
        job_id = create_job(agent="capa")
        background_tasks.add_task(
            _run_capa_background,
            job_id, problem_statement, country_code, device_class,
            device_type, user.org_id or "", user.user_id,
        )
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "pending",
                "message": f"CAPA analysis running. Poll GET /api/v1/jobs/{job_id}",
            },
        )

    job_id = create_job(agent="capa")
    await _run_capa_background(
        job_id, problem_statement, country_code, device_class,
        device_type, user.org_id or "", user.user_id
    )
    from app.services.job_store import get_job_response
    return get_job_response(job_id) or {"status": "failed", "job_id": job_id}
