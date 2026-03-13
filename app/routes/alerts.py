"""
RAA — Regulatory Alert Agent endpoints.
GET  /api/v1/alerts               — list recent alerts (per org)
POST /api/v1/alerts/subscribe     — subscribe org to country
GET  /api/v1/alerts/subscriptions — list subscribed countries
GET  /api/v1/alerts/{country}     — alerts for one country
POST /api/v1/alerts/check-changes — admin: run RAA for a country, emit alerts (PRD line 339)
GET  /api/v1/alerts/live/fda-recalls — live FDA recall feed (openFDA API)
GET  /api/v1/alerts/live/maude      — live MAUDE adverse events (openFDA API)
GET  /api/v1/alerts/live/eu-dhpcs   — live EMA DHPC safety communications
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from app.middleware.auth import AuthenticatedUser, get_current_user
from app.middleware.rbac import require_admin
from app.services.alert_store import add_alert, get_alerts, get_subscribed_orgs, get_subscriptions, subscribe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["RAA"])


# ─────────────────────────────────────────────────────────────────────────────
# Request / response schemas
# ─────────────────────────────────────────────────────────────────────────────

class SubscribeBody(BaseModel):
    country: str


class CheckChangesBody(BaseModel):
    """Body for POST /api/v1/alerts/check-changes — PRD line 339.

    When document_id is provided, checks only that specific document.
    Otherwise runs the full country sweep (all monitored_docs for the country).
    The RAA crew scrapes source URLs directly — document_text is not accepted.
    """
    country: str
    document_id: Optional[str] = None
    async_mode: bool = False


class CheckChangesResult(BaseModel):
    alerts_emitted: int = 0
    changed_chunks: int = 0
    country: str = ""
    document_id: Optional[str] = None
    job_id: Optional[str] = None
    status: str = "completed"
    message: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_check_changes_sync(
    country: str,
    document_id: Optional[str],
) -> CheckChangesResult:
    """
    Run RAA for a country/document; emit alerts for any changed chunks.

    RAA scrapes source URLs from the monitored_docs registry — it does NOT
    accept raw document text.  Each document is fetched live, hashed, and
    compared against the stored FAISS chunk hashes.

    Alerts are persisted via add_alert() and notified orgs are subscribed ones.
    """
    from app.agents.raa_agent import run_raa_for_country, run_raa_for_document
    from app.ingestion.monitored_docs import get_monitored_doc, get_monitored_docs

    alerts_emitted = 0
    changed_chunks = 0

    try:
        if document_id:
            # Single-document check: look up source URL from the registry
            doc = get_monitored_doc(country, document_id)
            if not doc:
                return CheckChangesResult(
                    country=country,
                    document_id=document_id,
                    status="failed",
                    message=(
                        f"Document '{document_id}' is not in the monitored registry "
                        f"for country '{country}'. Use GET /api/v1/monitored-docs to "
                        f"see registered documents."
                    ),
                )
            event = run_raa_for_document(
                country=country,
                document_id=doc["document_id"],
                source_url=doc["source_url"],
                regulation_name=doc["regulation_name"],
            )
            events = [event] if event is not None else []
        else:
            # Full country sweep: run RAA for every monitored document
            documents = get_monitored_docs(country)
            if not documents:
                return CheckChangesResult(
                    country=country,
                    status="failed",
                    message=(
                        f"No monitored documents registered for country '{country}'. "
                        f"Supported countries: US, EU, UK, UA, IN, CA, AU, JP, "
                        f"CN, BR, KR, CH, MX, RU, SA."
                    ),
                )
            events = run_raa_for_country(country=country, documents=documents)

        notified_orgs = get_subscribed_orgs(country)

        for event in events:
            changed_chunks += 1
            if not isinstance(event, dict):
                try:
                    event = event.model_dump() if hasattr(event, "model_dump") else dict(event)
                except Exception as ser_exc:
                    logger.warning("CRITICAL: Alert event serialization failed: %s — using minimal dict", ser_exc)
                    event = {"country": country, "document_id": document_id}

            event["notified_orgs"] = notified_orgs
            event.setdefault("country", country)
            add_alert(event)
            alerts_emitted += 1

        logger.info(
            f"RAA check-changes: country={country} changed={changed_chunks} "
            f"alerts={alerts_emitted} notified_orgs={len(notified_orgs)}"
        )

    except Exception as e:
        logger.error(f"RAA check-changes failed for country={country}: {e}", exc_info=True)
        return CheckChangesResult(
            country=country,
            document_id=document_id,
            status="failed",
            message=str(e),
        )

    return CheckChangesResult(
        alerts_emitted=alerts_emitted,
        changed_chunks=changed_chunks,
        country=country,
        document_id=document_id,
        status="completed",
        message=(
            f"{alerts_emitted} alert(s) emitted for {country}."
            if alerts_emitted
            else f"No regulation changes detected for {country}."
        ),
    )


async def _run_check_changes_background(
    job_id: str,
    country: str,
    document_id: Optional[str],
) -> None:
    """Background wrapper: runs sync RAA crew in thread, stores result as job."""
    from app.services.job_store import set_completed, set_failed, set_running
    set_running(job_id)
    try:
        result = await asyncio.to_thread(
            _run_check_changes_sync, country, document_id
        )
        payload = result.model_dump()
        payload["job_id"] = job_id
        set_completed(job_id, payload)
    except Exception as e:
        logger.error(f"RAA background job {job_id} failed: {e}", exc_info=True)
        set_failed(job_id, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("")
def list_alerts(
    limit: int = 50,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """List recent alerts for the current user's org. Most recent first."""
    org_id = user.org_id or ""
    items = get_alerts(org_id=org_id, limit=limit)
    return {"alerts": items, "count": len(items)}


@router.post("/subscribe")
def subscribe_org(
    body: SubscribeBody,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Subscribe the current org to regulatory change alerts for a country."""
    org_id = user.org_id or ""
    if not org_id:
        return {"subscribed": False, "message": "Org not set"}
    subscribe(org_id, body.country)
    return {"subscribed": True, "country": body.country.upper(), "org_id": org_id}


@router.get("/subscriptions")
def list_subscriptions(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """List countries the current org is subscribed to."""
    org_id = user.org_id or ""
    countries = get_subscriptions(org_id)
    return {"countries": countries, "count": len(countries)}


@router.post("/check-changes", response_model=CheckChangesResult)
async def check_changes(
    body: CheckChangesBody,
    background_tasks: BackgroundTasks,
    user: AuthenticatedUser = Depends(require_admin),
):
    """
    Admin endpoint — PRD line 339.

    Trigger the RAA crew to check for regulatory changes for a country
    (or a specific document when document_id + document_text provided).
    Detected changes are stored as alert events; subscribed orgs are notified.

    When async_mode=True, returns 202 + job_id; poll GET /api/v1/jobs/{job_id}.
    When async_mode=False (default), runs synchronously and returns result.
    """
    country = body.country.strip().upper()
    if not country:
        raise HTTPException(status_code=422, detail="country is required")

    if body.async_mode:
        from app.services.job_store import create_job
        from fastapi.responses import JSONResponse
        job_id = create_job(agent="raa")
        background_tasks.add_task(
            _run_check_changes_background,
            job_id,
            country,
            body.document_id,
        )
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "pending",
                "country": country,
                "message": f"Poll GET /api/v1/jobs/{job_id} for result.",
            },
        )

    # Synchronous path: offload blocking RAA crew to thread
    result = await asyncio.to_thread(
        _run_check_changes_sync,
        country,
        body.document_id,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Live government data feeds
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/live/fda-recalls")
async def live_fda_recalls(
    device_name: Optional[str] = None,
    product_code: Optional[str] = None,
    days_back: int = 90,
    limit: int = 25,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Live FDA device recall feed via openFDA API.
    Returns recent Class I/II/III device recalls directly from FDA.
    """
    from app.services.openfda_client import fda_client
    recalls = await fda_client.get_recent_recalls(
        device_name=device_name,
        product_code=product_code,
        days_back=days_back,
        limit=limit,
    )
    summary = fda_client.format_recall_summary(recalls)
    return {"source": "openFDA", "count": len(recalls), "recalls": recalls, "summary": summary}


@router.get("/live/maude")
async def live_maude_events(
    device_name: Optional[str] = None,
    product_code: Optional[str] = None,
    days_back: int = 180,
    limit: int = 25,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Live MAUDE adverse event reports via openFDA API.
    Returns recent FDA medical device adverse event reports.
    """
    from app.services.openfda_client import fda_client
    events = await fda_client.get_adverse_events(
        device_name=device_name,
        product_code=product_code,
        days_back=days_back,
        limit=limit,
    )
    summary = fda_client.format_maude_summary(events)
    return {"source": "MAUDE/openFDA", "count": len(events), "events": events, "summary": summary}


@router.get("/live/eu-dhpcs")
async def live_eu_dhpcs(
    limit: int = 20,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Live EMA Direct Healthcare Professional Communications (DHPCs).
    EU safety alerts equivalent to FDA Class I recalls.
    """
    from app.services.ema_client import ema_client
    dhpcs = await ema_client.get_dhpcs(limit=limit)
    summary = ema_client.format_dhpc_summary(dhpcs)
    return {"source": "EMA", "count": len(dhpcs), "dhpcs": dhpcs, "summary": summary}


@router.get("/{country}")
def alerts_by_country(
    country: str,
    limit: int = 50,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """List recent alerts for a specific country."""
    org_id = user.org_id or ""
    items = get_alerts(org_id=org_id, country=country, limit=limit)
    return {"alerts": items, "country": country.upper(), "count": len(items)}
