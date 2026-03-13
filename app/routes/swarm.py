"""
OrthoLink GCO Route — Global Compliance Orchestrator
POST /api/v1/gco-analysis

Runs TDA + PMS + ROA + (optional CAPA) in parallel.
Returns aggregated regulatory readiness report.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException
from fastapi.responses import JSONResponse

from app.crews.swarm_analysis import GcoReport, run_gco_analysis
from app.middleware.auth import AuthenticatedUser
from app.middleware.rbac import require_reviewer
from app.services.job_store import create_job, set_completed, set_failed, set_running
from app.services.usage_metering import get_usage_meter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gco-analysis", tags=["GCO"])


async def _run_gco_background(
    job_id: str,
    country: str,
    device_class: str,
    device_type: Optional[str],
    problem_statement: Optional[str],
    org_id: str,
    user_id: str,
) -> None:
    set_running(job_id)
    try:
        report: GcoReport = await asyncio.to_thread(
            run_gco_analysis,
            country=country,
            device_class=device_class,
            device_type=device_type,
            problem_statement=problem_statement,
        )
        # Count as 3-4 agent runs for metering
        meter = get_usage_meter()
        for _ in range(report.agents_run):
            meter.record_usage(org_id=org_id, user_id=user_id, agent_type="gco")
        payload = report.model_dump()
        payload["job_id"] = job_id
        payload["status"] = "completed"
        set_completed(job_id, payload)
    except Exception as e:
        logger.error("GCO job %s failed: %s", job_id, e, exc_info=True)
        set_failed(job_id, str(e))


@router.post("")
async def gco_analysis(
    background_tasks: BackgroundTasks,
    country: str = Form(..., description="Target market country code"),
    device_class: str = Form(..., description="Device class (e.g. IIb, III)"),
    device_type: Optional[str] = Form(None, description="Device type"),
    problem_statement: Optional[str] = Form(
        None, description="Optional: active quality problem to trigger CAPA agent"
    ),
    async_mode: bool = Form(True),
    user: AuthenticatedUser = Depends(require_reviewer),
):
    """
    Global Compliance Orchestrator — runs multiple regulatory agents in parallel.

    Agents: Technical Documentation (TDA) + Post-Market Surveillance (PMS) +
    Regulatory Obligations (ROA) + optionally CAPA if problem_statement provided.

    Returns a unified regulatory readiness report with cross-agent synthesis.
    This is 'Department Mode' — one request = full regulatory department review.
    """
    meter = get_usage_meter()
    usage = meter.check_trial_limit(user.org_id or "")
    if usage["exceeded"]:
        raise HTTPException(status_code=402, detail="Free trial limit reached.")

    country_code = country.strip().upper()
    if not country_code:
        raise HTTPException(status_code=422, detail="country is required.")

    if async_mode:
        job_id = create_job(agent="gco")
        background_tasks.add_task(
            _run_gco_background,
            job_id, country_code, device_class, device_type,
            problem_statement, user.org_id or "", user.user_id,
        )
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "pending",
                "country": country_code,
                "agents": ["TDA", "PMS", "ROA"] + (["CAPA"] if problem_statement else []),
                "message": f"GCO analysis running. Poll GET /api/v1/jobs/{job_id}",
            },
        )

    job_id = create_job(agent="gco")
    await _run_gco_background(
        job_id, country_code, device_class, device_type,
        problem_statement, user.org_id or "", user.user_id
    )
    from app.services.job_store import get_job_response
    return get_job_response(job_id) or {"status": "failed", "job_id": job_id}
