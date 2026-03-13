"""
Ingestion Routes — Trigger and monitor the global regulatory data ingestion pipeline.
Router prefix: /ingestion, tags: ["Ingestion"]

POST /ingestion/run/full          — run full ingestion (all sources, all countries)
POST /ingestion/run/enforcement   — run enforcement scrapers only
POST /ingestion/run/registries    — run registry adapters only
POST /ingestion/run/country/{cc}  — run ingestion for one country
GET  /ingestion/status            — current pipeline status + last run report
GET  /ingestion/jobs/{job_id}     — poll async ingestion job

All run endpoints are async (202 + job_id). Admin-only.
"""

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.middleware.auth import AuthenticatedUser, get_current_user
from app.middleware.rbac import require_admin
from app.services.job_store import create_job, get_job, set_completed, set_failed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


# ─────────────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────────────


class IngestionJobResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str


class PipelineStatusResponse(BaseModel):
    pipeline_available: bool
    last_run: Optional[dict[str, Any]] = None
    status_summary: Optional[dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_pipeline():
    """Lazy import — pipeline has heavy optional deps (adapters, scrapers)."""
    try:
        from app.ingestion.pipeline import get_pipeline
        return get_pipeline()
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="Ingestion pipeline not available. Contact system administrator.",
        )


def _run_ingestion_background(job_id: str, mode: str, country: Optional[str] = None):
    """Run ingestion in a background thread and write result to job_store."""
    try:
        pipeline = _get_pipeline()

        async def _run():
            try:
                if mode == "full":
                    report = await pipeline.run_full_ingestion()
                elif mode == "enforcement":
                    report = await pipeline.run_enforcement_ingestion()
                elif mode == "registries":
                    countries = None  # all countries
                    report = await pipeline.run_registry_ingestion(countries)
                elif mode == "country" and country:
                    report = await pipeline.run_country_ingestion(country)
                else:
                    raise ValueError(f"Unknown ingestion mode: {mode!r}")

                payload = report.model_dump() if hasattr(report, "model_dump") else dict(report)
                set_completed(job_id, {"result": payload, "status": "completed", "agent": "ingestion"})
            except Exception as exc:
                logger.exception("Ingestion job %s failed: %s", job_id, exc)
                set_failed(job_id, str(exc))

        asyncio.run(_run())
    except Exception as exc:
        logger.exception("Ingestion background thread failed: %s", exc)
        set_failed(job_id, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/run/full", response_model=IngestionJobResponse, status_code=202)
async def run_full_ingestion(
    user: AuthenticatedUser = Depends(get_current_user),
    _admin=Depends(require_admin),
):
    """
    Trigger a full ingestion run across all 8 registry adapters and 5 enforcement scrapers.
    Returns a job_id — poll /ingestion/jobs/{job_id} or GET /jobs/{job_id} for progress.
    Admin-only.
    """
    job_id = create_job(agent="ingestion")
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_ingestion_background, job_id, "full")
    return IngestionJobResponse(
        job_id=job_id,
        status="queued",
        message="Full ingestion queued. Poll /api/v1/jobs/" + job_id + " for progress.",
    )


@router.post("/run/enforcement", response_model=IngestionJobResponse, status_code=202)
async def run_enforcement_ingestion(
    user: AuthenticatedUser = Depends(get_current_user),
    _admin=Depends(require_admin),
):
    """
    Trigger enforcement-only ingestion (FDA Warning Letters, TGA Alerts,
    EUDAMED FSNs, Health Canada Incidents, Market Surveillance).
    Admin-only.
    """
    job_id = create_job(agent="ingestion_enforcement")
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_ingestion_background, job_id, "enforcement")
    return IngestionJobResponse(
        job_id=job_id,
        status="queued",
        message="Enforcement ingestion queued. Poll /api/v1/jobs/" + job_id,
    )


@router.post("/run/registries", response_model=IngestionJobResponse, status_code=202)
async def run_registry_ingestion(
    user: AuthenticatedUser = Depends(get_current_user),
    _admin=Depends(require_admin),
):
    """
    Trigger registry-adapter-only ingestion (GUDID, EUDAMED, Swissdamed,
    ANVISA, ARTG, MDALL, SUGAM, GMDN). Admin-only.
    """
    job_id = create_job(agent="ingestion_registries")
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_ingestion_background, job_id, "registries")
    return IngestionJobResponse(
        job_id=job_id,
        status="queued",
        message="Registry ingestion queued. Poll /api/v1/jobs/" + job_id,
    )


@router.post("/run/country/{country_code}", response_model=IngestionJobResponse, status_code=202)
async def run_country_ingestion(
    country_code: str = Path(..., min_length=2, max_length=4, pattern=r"^[A-Za-z]{2,4}$", description="ISO country code, e.g. US, AU, EU"),
    user: AuthenticatedUser = Depends(get_current_user),
    _admin=Depends(require_admin),
):
    """
    Trigger ingestion for a single country — monitored docs + registry adapter.
    Admin-only.
    """
    cc = country_code.upper()
    job_id = create_job(agent=f"ingestion_{cc}")
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_ingestion_background, job_id, "country", cc)
    return IngestionJobResponse(
        job_id=job_id,
        status="queued",
        message=f"{cc} ingestion queued. Poll /api/v1/jobs/{job_id}",
    )


@router.get("/status", response_model=PipelineStatusResponse)
async def get_ingestion_status(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Return current pipeline availability + status from the last ingestion run.
    Readable by all authenticated users.
    """
    try:
        pipeline = _get_pipeline()
        status = pipeline.get_ingestion_status()
        return PipelineStatusResponse(
            pipeline_available=True,
            status_summary=status,
        )
    except HTTPException:
        return PipelineStatusResponse(
            pipeline_available=False,
            last_run=None,
            status_summary={"error": "Pipeline not initialised"},
        )
    except Exception as exc:
        logger.warning("Pipeline status check failed: %s", exc)
        return PipelineStatusResponse(
            pipeline_available=False,
            status_summary={"error": "Pipeline status unavailable."},
        )


@router.get("/jobs/{job_id}")
async def get_ingestion_job(
    job_id: str = Path(..., description="Job ID returned from a run endpoint"),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Poll the status of an ingestion job by ID.
    Returns the IngestionReport when status == 'completed'.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JSONResponse(content=job)
