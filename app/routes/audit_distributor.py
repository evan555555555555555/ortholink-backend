"""
OrthoLink DVA — Batch Audit Route
POST /api/v1/audit-distributor

Runs the DVA 7-step pipeline across multiple countries for a single CSV.
Returns a BatchReport with per-country gap analysis and aggregate statistics.
Always async (returns 202 + job_id; poll GET /api/v1/jobs/{job_id}).
"""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.crews.batch_analysis import run_batch_analysis
from app.middleware.auth import AuthenticatedUser
from app.middleware.rbac import require_reviewer
from app.services.job_store import create_job, set_completed, set_failed, set_running
from app.services.usage_metering import get_usage_meter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit-distributor", tags=["DVA"])

# All 15 supported countries (PRD §2.1)
ALL_COUNTRIES = [
    "US", "EU", "UK", "CA", "AU", "JP", "CN", "BR", "IN",
    "UA", "CH", "MX", "KR", "RU", "SA",
]


async def _run_batch_background(
    job_id: str,
    csv_content: str,
    countries: list[str],
    device_class: str,
    device_name: str,
    device_type: Optional[str],
    org_id: str,
    user_id: str,
) -> None:
    """Background task: run batch DVA and store aggregated BatchReport."""
    set_running(job_id)
    try:
        report = await run_batch_analysis(
            csv_content=csv_content,
            countries=countries,
            device_class=device_class,
            device_name=device_name,
            device_type=device_type,
        )
        get_usage_meter().record_usage(org_id=org_id, user_id=user_id, agent_type="dva_batch")
        payload = report.model_dump()
        payload["job_id"] = job_id
        payload["status"] = "completed"
        set_completed(job_id, payload)
    except Exception as e:
        logger.error(f"Batch DVA job {job_id} failed: {e}", exc_info=True)
        set_failed(job_id, str(e))


@router.post("")
async def audit_distributor(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV of distributor document items"),
    device_class: str = Form(..., description="Device class (e.g. IIb)"),
    device_name: str = Form("", description="Device name (for report header)"),
    device_type: Optional[str] = Form(None, description="Device type (e.g. orthopedic_implant)"),
    countries: Optional[str] = Form(
        None,
        description=(
            "Comma-separated country codes to audit. "
            "Omit or pass 'ALL' to run all 15 supported countries."
        ),
    ),
    user: AuthenticatedUser = Depends(require_reviewer),
):
    """
    DVA Batch Audit — PRD §4.2 extended.

    Runs the full DVA gap analysis across multiple countries for a single
    distributor CSV. Always async: returns 202 + job_id.
    Poll GET /api/v1/jobs/{job_id} for the completed BatchReport.
    """
    meter = get_usage_meter()
    usage = meter.check_trial_limit(user.org_id or "")
    if usage["exceeded"]:
        raise HTTPException(status_code=402, detail="Free trial limit reached.")

    # Parse CSV
    content_bytes = await file.read()
    try:
        csv_content = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        csv_content = content_bytes.decode("latin-1")

    if not csv_content.strip():
        raise HTTPException(status_code=422, detail="CSV file is empty.")

    # Parse countries
    if not countries or countries.strip().upper() == "ALL":
        target_countries = ALL_COUNTRIES
    else:
        target_countries = [c.strip().upper() for c in countries.split(",") if c.strip()]
        invalid = [c for c in target_countries if c not in ALL_COUNTRIES]
        if invalid:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown country codes: {invalid}. Valid: {ALL_COUNTRIES}",
            )

    job_id = create_job(agent="dva_batch")
    background_tasks.add_task(
        _run_batch_background,
        job_id,
        csv_content,
        target_countries,
        device_class,
        device_name,
        device_type,
        user.org_id or "",
        user.user_id,
    )

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "pending",
            "countries": target_countries,
            "message": f"Batch audit running for {len(target_countries)} countries. "
                       f"Poll GET /api/v1/jobs/{job_id} for result.",
        },
    )
