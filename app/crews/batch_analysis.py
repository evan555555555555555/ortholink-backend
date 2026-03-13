"""
OrthoLink DVA Crew — Batch Analysis
POST /api/v1/audit-distributor (called via audit route)

PRD §4.2 extended: Batch Analysis runs the DVA 7-step pipeline across
multiple distributor CSV submissions or across multiple countries for
a single submission, aggregating results into a BatchReport.

Uses asyncio.gather to run multiple run_dva_analysis calls concurrently
(each on a different country or CSV). Respects the same confidence gate,
set-theory logic, and soft-delete rules as the single-submission flow.
"""

import asyncio
import logging
import uuid
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Batch reference analysis only. Verify all findings with official sources. "
    "Does not constitute legal advice."
)


# ─────────────────────────────────────────────────────────────────────────────
# Output models
# ─────────────────────────────────────────────────────────────────────────────

class BatchItem(BaseModel):
    """Single DVA result within a batch run."""

    country: str
    device_class: str
    total_submitted: int = 0
    required: int = 0
    extra: int = 0
    missing: int = 0
    optional: int = 0
    unverifiable: int = 0
    fraud_risk_score: float = 0.0
    status: str = "completed"   # completed | failed | skipped
    error: Optional[str] = None


class BatchReport(BaseModel):
    """Aggregated DVA batch analysis result."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_name: str = ""
    device_class: str = ""
    countries_analysed: list[str] = Field(default_factory=list)
    results: list[BatchItem] = Field(default_factory=list)
    highest_risk_country: Optional[str] = None
    average_fraud_risk: float = 0.0
    total_items_analysed: int = 0
    disclaimer: str = Field(default=DISCLAIMER)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

async def _run_single_dva(
    csv_content: str,
    country: str,
    device_class: str,
    device_type: Optional[str],
) -> BatchItem:
    """
    Run DVA pipeline for a single (csv, country, device_class) tuple.
    Returns a BatchItem summary; never raises — captures errors in .error field.
    """
    from app.crews.verify_distributor import run_dva_analysis

    try:
        report = await run_dva_analysis(
            csv_content=csv_content,
            country=country,
            device_class=device_class,
            device_type=device_type,
        )
        s = report.summary
        return BatchItem(
            country=country,
            device_class=device_class,
            total_submitted=s.total_submitted,
            required=s.required,
            extra=s.extra,
            missing=s.missing,
            optional=s.optional,
            unverifiable=s.unverifiable,
            fraud_risk_score=s.fraud_risk_score,
            status="completed",
        )
    except Exception as e:
        logger.error("batch_analysis DVA failed for country=%s: %s", country, e, exc_info=True)
        return BatchItem(
            country=country,
            device_class=device_class,
            status="failed",
            error=str(e),
        )


async def run_batch_analysis(
    csv_content: str,
    countries: list[str],
    device_class: str,
    device_name: str = "",
    device_type: Optional[str] = None,
    max_concurrent: int = 5,
) -> BatchReport:
    """
    Run DVA analysis concurrently across multiple countries for a single CSV.

    Args:
        csv_content:     CSV text (same format as POST /api/v1/verify-distributor)
        countries:       List of ISO country codes to check against
        device_class:    Device class (e.g. "IIb")
        device_name:     Optional device name for the report
        device_type:     Optional device type (e.g. "orthopedic_implant")
        max_concurrent:  Max parallel DVA runs (default 5 to avoid OpenAI rate limits)

    Returns:
        BatchReport aggregating all individual DVA results.
    """
    if not countries:
        return BatchReport(
            device_name=device_name,
            device_class=device_class,
            disclaimer=DISCLAIMER,
        )

    # Deduplicate and normalise
    unique_countries = list(dict.fromkeys(c.strip().upper() for c in countries if c.strip()))

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _with_semaphore(country: str) -> BatchItem:
        async with semaphore:
            return await _run_single_dva(csv_content, country, device_class, device_type)

    tasks = [_with_semaphore(c) for c in unique_countries]
    results: list[BatchItem] = list(await asyncio.gather(*tasks))

    # Aggregate statistics
    completed = [r for r in results if r.status == "completed"]
    total_items = sum(r.total_submitted for r in completed)

    avg_risk = 0.0
    highest_risk_country: Optional[str] = None
    if completed:
        avg_risk = round(sum(r.fraud_risk_score for r in completed) / len(completed), 3)
        highest = max(completed, key=lambda r: r.fraud_risk_score)
        if highest.fraud_risk_score > 0.0:
            highest_risk_country = highest.country

    return BatchReport(
        job_id=str(uuid.uuid4()),
        device_name=device_name,
        device_class=device_class,
        countries_analysed=unique_countries,
        results=results,
        highest_risk_country=highest_risk_country,
        average_fraud_risk=avg_risk,
        total_items_analysed=total_items,
        disclaimer=DISCLAIMER,
    )
