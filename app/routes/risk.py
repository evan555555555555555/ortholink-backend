"""
OrthoLink Risk Management Agent (RMA) Route
POST /api/v1/risk-analysis

ISO 14971:2019 risk analysis — async (202 + job poll) or sync.
Input:  device_description, intended_use, device_class, country, hazards_hint (optional)
Output: RiskManagementReport (hazard table, risk matrix, controls, residual risk, verdict)
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException
from fastapi.responses import JSONResponse

from app.crews.risk_analysis import RiskManagementReport, run_risk_analysis
from app.middleware.auth import AuthenticatedUser
from app.middleware.rbac import require_reviewer
from app.services.job_store import create_job, set_completed, set_failed, set_running
from app.services.usage_metering import get_usage_meter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk-analysis", tags=["Risk Management"])


async def _run_rma_background(
    job_id: str,
    device_description: str,
    intended_use: str,
    device_class: str,
    country: str,
    hazards_hint: Optional[str],
    org_id: str,
    user_id: str,
) -> None:
    set_running(job_id)
    try:
        report: RiskManagementReport = await asyncio.to_thread(
            run_risk_analysis,
            device_description=device_description,
            intended_use=intended_use,
            device_class=device_class,
            country=country,
            hazards_hint=hazards_hint,
        )
        get_usage_meter().record_usage(org_id=org_id, user_id=user_id, agent_type="rma")
        payload = report.model_dump()
        payload["job_id"] = job_id
        payload["status"] = "completed"
        set_completed(job_id, payload)
    except Exception as e:
        logger.error("RMA job %s failed: %s", job_id, e, exc_info=True)
        set_failed(job_id, str(e))


@router.post("")
async def risk_analysis(
    background_tasks: BackgroundTasks,
    device_description: str = Form(
        ...,
        max_length=2000,
        description="Short description of the medical device (e.g. 'tibial knee implant with UHMWPE bearing')",
    ),
    intended_use: str = Form(
        ...,
        max_length=2000,
        description="Intended use statement per ISO 14971 §4.2 (e.g. 'total knee replacement in adults')",
    ),
    country: str = Form(..., description="Primary regulatory market (ISO 3166-1 alpha-2)"),
    device_class: str = Form(
        ..., description="Regulatory classification (I / IIa / IIb / III / II)"
    ),
    hazards_hint: Optional[str] = Form(
        None,
        max_length=2000,
        description="Optional: known hazards or concerns to prime analysis (e.g. 'wear debris, bone resorption')",
    ),
    async_mode: bool = Form(True),
    user: AuthenticatedUser = Depends(require_reviewer),
):
    """
    Risk Management Agent — ISO 14971:2019 Compliant Risk Analysis.

    Generates:
    - Hazard analysis table (hazard → hazardous situation → harm → P × S matrix)
    - Risk acceptability verdict per ISO 14971 Annex D matrix (ACCEPTABLE | ALARP | UNACCEPTABLE)
    - Risk control measures with ISO 14971 §6.2 hierarchy labeling
    - Residual risk assessment (post-control P × S)
    - Overall verdict + §7.4 benefit-risk justification

    Backed by 24,258-chunk FAISS vector store (official regulatory sources only).
    Cryptographically signed + auto fact-checked via IntegrityGuard.
    """
    meter = get_usage_meter()
    usage = meter.check_trial_limit(user.org_id or "")
    if usage["exceeded"]:
        raise HTTPException(status_code=402, detail="Free trial limit reached.")

    country_code = country.strip().upper()
    if not device_description.strip():
        raise HTTPException(status_code=422, detail="device_description is required.")
    if not intended_use.strip():
        raise HTTPException(status_code=422, detail="intended_use is required.")

    if async_mode:
        job_id = create_job(agent="rma")
        background_tasks.add_task(
            _run_rma_background,
            job_id, device_description, intended_use, device_class,
            country_code, hazards_hint, user.org_id or "", user.user_id,
        )
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "pending",
                "country": country_code,
                "device_class": device_class,
                "message": f"ISO 14971 risk analysis running. Poll GET /api/v1/jobs/{job_id}",
            },
        )

    # Synchronous path
    job_id = create_job(agent="rma")
    await _run_rma_background(
        job_id, device_description, intended_use, device_class,
        country_code, hazards_hint, user.org_id or "", user.user_id,
    )
    from app.services.job_store import get_job_response
    return get_job_response(job_id) or {"status": "failed", "job_id": job_id}
