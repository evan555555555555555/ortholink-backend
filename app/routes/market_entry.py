"""
OrthoLink Multi-Agent Pipeline — Full Analysis & Batch Analysis.

POST /api/v1/full-analysis — Manager delegates RSA → DVA → ROA → MarketEntryPackage.
POST /api/v1/batch-analysis — Parallel DVA over multiple {csv, country, class} payloads.
"""

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.crews.generate_checklist import RoleSplitChecklist, run_roa_checklist
from app.crews.plan_strategy import run_rsa_strategy, StrategyReport
from app.crews.verify_distributor import run_dva_analysis
from app.middleware.auth import AuthenticatedUser
from app.middleware.rbac import require_reviewer
from app.services.usage_metering import get_usage_meter

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Market Entry"])

DISCLAIMER = "Reference tool only. Verify with official sources."

# Limit concurrent DVA runs so we never hit OpenAI rate limits (429). Process control:
# asyncio.Semaphore caps how many DVA analyses run at once; the rest wait. Without this,
# a burst of batch requests would fire N parallel LLM/embed calls and trigger rate-limit
# failures. The semaphore makes the batch pipeline predictable and stable under load.
BATCH_SEMAPHORE_LIMIT = 3


# ─── Full Analysis: unified output bridging RSA + DVA + ROA ───────────────────

class MarketEntryPackage(BaseModel):
    """Unified output for POST /api/v1/full-analysis."""

    job_id: str = Field(default="")
    strategy: Optional[StrategyReport] = None
    gap_report: Optional[Any] = None  # GapAnalysisReport
    checklist: Optional[RoleSplitChecklist] = None
    disclaimer: str = Field(default=DISCLAIMER)


@router.post("/full-analysis", response_model=MarketEntryPackage)
async def full_analysis(
    device_name: str = Form(..., description="Device name"),
    target_markets: str = Form(..., description="Comma-separated country codes"),
    device_class: str = Form(..., description="Device class (e.g. IIb)"),
    device_type: Optional[str] = Form(None),
    csv_file: Optional[UploadFile] = File(None),
    country_for_dva: Optional[str] = Form(None, description="Single country for DVA if CSV provided"),
    country_for_roa: Optional[str] = Form(None, description="Country for ROA checklist"),
    user: AuthenticatedUser = Depends(require_reviewer),
) -> MarketEntryPackage:
    """
    Master pipeline: run RSA (strategy), then optionally DVA (if CSV + country_for_dva),
    then ROA checklist for country_for_roa (or first target market). All runs offloaded to thread.
    """
    meter = get_usage_meter()
    if meter.check_trial_limit(user.org_id or "").get("exceeded"):
        raise HTTPException(status_code=402, detail="Free trial limit reached.")

    markets = [m.strip() for m in target_markets.split(",") if m.strip()]
    if not markets:
        raise HTTPException(status_code=400, detail="At least one target market required.")

    package = MarketEntryPackage(job_id="")

    try:
        # 1. RSA — strategy report
        strategy_report = await asyncio.to_thread(
            run_rsa_strategy,
            device_name=device_name,
            target_markets=markets,
            device_class=device_class or "IIb",
        )
        package.strategy = strategy_report
        package.job_id = strategy_report.job_id

        # 2. DVA — if CSV and country provided
        if csv_file and country_for_dva:
            content = await csv_file.read()
            csv_content = content.decode("utf-8", errors="replace")
            gap_report = await run_dva_analysis(
                csv_content=csv_content,
                country=country_for_dva,
                device_class=device_class or "IIb",
                device_type=device_type,
            )
            package.gap_report = gap_report

        # 3. ROA — checklist for one country
        roa_country = country_for_roa or (markets[0] if markets else "US")
        checklist = await asyncio.to_thread(
            run_roa_checklist,
            country=roa_country,
            device_class=device_class or "IIb",
            device_type=device_type,
        )
        package.checklist = checklist
        package.disclaimer = DISCLAIMER

        meter.record_usage(org_id=user.org_id or "", user_id=user.user_id, agent_type="full_analysis")
        return package
    except Exception as e:
        logger.exception("full_analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=f"System Integrity Check: full-analysis failed. Reference: full_analysis.")


# ─── Batch Analysis: concurrent DVA over multiple distributor CSVs ─────────────

class BatchItemRequest(BaseModel):
    """Single item in batch-analysis request."""

    country: str = Field(..., min_length=1)
    device_class: str = Field(..., min_length=1)
    device_type: Optional[str] = None
    csv_content: str = Field(..., min_length=1)


class BatchItemResult(BaseModel):
    """Result for one batch item."""

    country: str = ""
    device_class: str = ""
    job_id: str = ""
    gap_report: Optional[Any] = None
    error: Optional[str] = None


class BatchReport(BaseModel):
    """Consolidated output for POST /api/v1/batch-analysis."""

    job_id: str = Field(default="")
    total: int = 0
    completed: int = 0
    failed: int = 0
    results: list[BatchItemResult] = Field(default_factory=list)
    disclaimer: str = Field(default=DISCLAIMER)


_semaphore: Optional[asyncio.Semaphore] = None


def _get_batch_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(BATCH_SEMAPHORE_LIMIT)
    return _semaphore


async def _run_one_dva(item: BatchItemRequest) -> BatchItemResult:
    """Run DVA for one item; semaphore limits concurrency."""
    sem = _get_batch_semaphore()
    async with sem:
        try:
            report = await run_dva_analysis(
                csv_content=item.csv_content,
                country=item.country,
                device_class=item.device_class,
                device_type=item.device_type,
            )
            return BatchItemResult(
                country=item.country,
                device_class=item.device_class,
                job_id=report.analysis_id,
                gap_report=report,
            )
        except Exception as e:
            logger.warning("Batch DVA failed for %s: %s", item.country, e)
            return BatchItemResult(
                country=item.country,
                device_class=item.device_class,
                error=str(e),
            )


@router.post("/batch-analysis", response_model=BatchReport)
async def batch_analysis(
    body: list[BatchItemRequest],
    user: AuthenticatedUser = Depends(require_reviewer),
) -> BatchReport:
    """
    Run DVA in parallel for multiple {country, device_class, csv_content} payloads.
    Concurrency is capped by an asyncio.Semaphore (BATCH_SEMAPHORE_LIMIT) so we never
    exceed a fixed number of in-flight DVA runs—eliminating the API rate-limit bottleneck
    before it can cause 429s. Each batch item acquires the semaphore, runs DVA, then
    releases; excess tasks wait. Keeps the pipeline stable under load.
    """
    meter = get_usage_meter()
    if meter.check_trial_limit(user.org_id or "").get("exceeded"):
        raise HTTPException(status_code=402, detail="Free trial limit reached.")

    if not body or len(body) > 20:
        raise HTTPException(status_code=400, detail="Provide 1–20 batch items.")

    import uuid
    job_id = str(uuid.uuid4())
    tasks = [_run_one_dva(item) for item in body]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out_results: list[BatchItemResult] = []
    completed = 0
    failed = 0
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            out_results.append(
                BatchItemResult(
                    country=body[i].country,
                    device_class=body[i].device_class,
                    error=str(r),
                )
            )
            failed += 1
        else:
            out_results.append(r)
            if r.error:
                failed += 1
            else:
                completed += 1

    meter.record_usage(org_id=user.org_id or "", user_id=user.user_id, agent_type="batch_analysis")
    return BatchReport(
        job_id=job_id,
        total=len(body),
        completed=completed,
        failed=failed,
        results=out_results,
        disclaimer=DISCLAIMER,
    )
