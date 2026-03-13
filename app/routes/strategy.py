"""
OrthoLink Strategy (RSA) Routes
POST /api/v1/plan-strategy — Regulatory Strategy Agent.

Supports async_mode: when True, returns 202 + job_id; crew runs in background.
Poll GET /api/v1/jobs/{job_id} for result.
"""

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Form
from fastapi.responses import JSONResponse

from app.middleware.auth import AuthenticatedUser
from app.middleware.rbac import require_reviewer
from app.services.job_store import create_job, set_completed, set_failed, set_running

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/plan-strategy", tags=["RSA"])


async def _run_rsa_background(
    job_id: str,
    device_name: str,
    target_markets: list[str],
    org_id: str,
    user_id: str,
):
    """Background task: run RSA crew (Planner/Retriever/Analyzer/Critic) and store StrategyReport.

    run_rsa_strategy is synchronous (CrewAI + LLM calls). We offload to a thread via
    asyncio.to_thread() so the FastAPI event loop is never blocked — even when called
    from the inline (sync_mode) path on line 84.
    """
    set_running(job_id)
    try:
        from app.crews.plan_strategy import run_rsa_strategy, StrategyReport

        # Offload blocking CrewAI execution to a thread pool thread
        report: StrategyReport = await asyncio.to_thread(
            run_rsa_strategy,
            device_name=device_name,
            target_markets=target_markets,
        )
        result = report.model_dump()
        result["job_id"] = job_id
        result["status"] = "completed"
        set_completed(job_id, result)
    except Exception as e:
        logger.error(f"RSA background job {job_id} failed: {e}", exc_info=True)
        set_failed(job_id, str(e))


@router.post("")
async def plan_strategy(
    background_tasks: BackgroundTasks,
    device_name: str = Form(..., description="Device name"),
    target_markets: str = Form(..., description="Comma-separated country codes (e.g. US,UA,IN)"),
    async_mode: bool = Form(True, description="If true, return 202 + job_id; poll GET /api/v1/jobs/{job_id}"),
    user: AuthenticatedUser = Depends(require_reviewer),
):
    """
    Regulatory Strategy Agent (RSA). Returns optimal entry sequence and document reuse.
    When async_mode=True, returns 202 with job_id; poll GET /api/v1/jobs/{job_id} for result.
    """
    markets = [m.strip() for m in target_markets.split(",") if m.strip()]

    if async_mode:
        job_id = create_job(agent="rsa")
        background_tasks.add_task(
            _run_rsa_background,
            job_id,
            device_name,
            markets,
            user.org_id or "",
            user.user_id,
        )
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "pending",
                "message": f"Poll GET /api/v1/jobs/{job_id} for result.",
            },
        )

    # Sync: run inline (blocking — use async_mode under load)
    job_id = create_job(agent="rsa")
    await _run_rsa_background(job_id, device_name, markets, user.org_id or "", user.user_id)
    from app.services.job_store import get_job_response
    response = get_job_response(job_id)
    return response or {"status": "failed", "job_id": job_id}
