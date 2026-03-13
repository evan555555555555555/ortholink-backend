"""
Enforcement Routes — Cross-source regulatory enforcement actions dashboard.
Router prefix: /enforcement, tags: ["Enforcement"]

GET  /enforcement/overview                     — dashboard stats + recent actions from all sources
GET  /enforcement/warning-letters              — FDA Warning Letters
GET  /enforcement/warning-letters/patterns     — citation pattern analysis across warning letters
GET  /enforcement/tga-alerts                   — TGA Safety Alerts (Australia)
GET  /enforcement/fsn                          — EUDAMED Field Safety Notices (EU)
GET  /enforcement/fsn/{device_name}/risk-profile — FSN risk profile for a device
GET  /enforcement/hc-incidents                 — Health Canada Incident Reports
GET  /enforcement/device/{device_name}         — cross-source enforcement history for a device
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.middleware.auth import AuthenticatedUser, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enforcement", tags=["Enforcement"])

# ─────────────────────────────────────────────────────────────────────────────
# Scraper imports — graceful degradation if modules not yet built
# ─────────────────────────────────────────────────────────────────────────────

try:
    from app.scrapers.fda_warning_letters import FDAWarningLettersScraper
except ImportError:
    FDAWarningLettersScraper = None  # type: ignore[assignment, misc]

try:
    from app.scrapers.tga_alerts import TGASafetyAlertsScraper
except ImportError:
    TGASafetyAlertsScraper = None  # type: ignore[assignment, misc]

try:
    from app.scrapers.eudamed_fsn import EUDAMEDFSNScraper
except ImportError:
    EUDAMEDFSNScraper = None  # type: ignore[assignment, misc]

try:
    from app.scrapers.hc_incidents import HCIncidentsScraper
except ImportError:
    HCIncidentsScraper = None  # type: ignore[assignment, misc]


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class EnforcementAction(BaseModel):
    """A single enforcement action from any source."""
    action_id: str = ""
    source: str = ""
    country: str = ""
    device_name: str = ""
    manufacturer: str = ""
    action_type: str = ""
    severity: str = ""
    date_issued: Optional[str] = None
    summary: str = ""
    url: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EnforcementOverview(BaseModel):
    """Dashboard-level summary of enforcement actions across all sources."""
    total_actions: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    by_severity: dict[str, int] = Field(default_factory=dict)
    recent: list[dict[str, Any]] = Field(default_factory=list)
    period_days: int = 30
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CitationPattern(BaseModel):
    """Recurring CFR / regulation citation found in warning letters."""
    citation: str
    count: int
    regulation: str = ""
    device_types: list[str] = Field(default_factory=list)
    recent_letters: list[str] = Field(default_factory=list)


class CitationPatternsResponse(BaseModel):
    total_letters_analyzed: int = 0
    top_citations: list[CitationPattern] = Field(default_factory=list)
    analysis_period_days: int = 90
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class FSNRiskProfile(BaseModel):
    """Field Safety Notice risk profile for a device."""
    device_name: str
    total_fsns: int = 0
    severity_breakdown: dict[str, int] = Field(default_factory=dict)
    most_common_hazards: list[str] = Field(default_factory=list)
    affected_countries: list[str] = Field(default_factory=list)
    earliest_fsn: Optional[str] = None
    latest_fsn: Optional[str] = None
    risk_level: str = "UNKNOWN"
    notices: list[dict[str, Any]] = Field(default_factory=list)


class CrossSourceHistory(BaseModel):
    """Enforcement history for a device across all data sources."""
    device_name: str
    total_actions: int = 0
    sources_checked: list[str] = Field(default_factory=list)
    actions_by_source: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scraper_available(cls: Any) -> bool:
    """Return True if a scraper class was successfully imported."""
    return cls is not None


def _normalise_actions(raw: Any, source: str, country: str) -> list[dict[str, Any]]:
    """
    Normalise whatever a scraper returns into a flat list of dicts.
    Scrapers may return lists of dicts, Pydantic models, or structured objects.
    """
    items: list[dict[str, Any]] = []
    if raw is None:
        return items
    iterable = raw if isinstance(raw, list) else [raw]
    for item in iterable:
        if hasattr(item, "model_dump"):
            d = item.model_dump()
        elif isinstance(item, dict):
            d = dict(item)
        else:
            d = {"raw": str(item)}
        d.setdefault("source", source)
        d.setdefault("country", country)
        items.append(d)
    return items


async def _run_scraper(
    scraper_cls: Any,
    method: str,
    kwargs: dict[str, Any],
    source_label: str,
    country: str,
) -> tuple[str, list[dict[str, Any]], Optional[str]]:
    """
    Instantiate a scraper, call the named method with kwargs, return
    (source_label, normalised_actions, error_or_None).
    Runs blocking scrapers in a thread so they never block the event loop.
    """
    if scraper_cls is None:
        return source_label, [], f"{source_label} scraper not available"
    try:
        scraper = scraper_cls()
        fn = getattr(scraper, method, None)
        if fn is None:
            return source_label, [], f"{source_label}.{method}() not implemented"
        if asyncio.iscoroutinefunction(fn):
            raw = await fn(**kwargs)
        else:
            raw = await asyncio.to_thread(fn, **kwargs)
        return source_label, _normalise_actions(raw, source_label, country), None
    except Exception as exc:
        logger.warning("Scraper %s.%s failed: %s", source_label, method, exc)
        return source_label, [], str(exc)


def _classify_severity(action: dict[str, Any]) -> str:
    """Derive a normalised severity label from an action dict."""
    for key in ("severity", "class", "risk_level", "action_type"):
        val = str(action.get(key, "")).upper()
        if any(w in val for w in ("CRITICAL", "CLASS I", "HIGH", "URGENT")):
            return "CRITICAL"
        if any(w in val for w in ("CLASS II", "MEDIUM", "MODERATE", "WARNING")):
            return "MODERATE"
        if any(w in val for w in ("CLASS III", "LOW", "ADVISORY", "RECALL")):
            return "LOW"
    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/overview", response_model=EnforcementOverview)
async def enforcement_overview(
    days: int = Query(30, ge=1, le=365, description="Lookback window in days"),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Dashboard: aggregated enforcement stats across FDA Warning Letters, TGA Safety
    Alerts, EUDAMED Field Safety Notices, and Health Canada Incidents.

    Returns total action count, breakdown by source and severity, and the 20
    most recent actions sorted newest-first.
    """
    tasks = [
        _run_scraper(
            FDAWarningLettersScraper, "get_recent",
            {"days": days},
            "FDA Warning Letters", "US",
        ),
        _run_scraper(
            TGASafetyAlertsScraper, "get_recent",
            {"days": days},
            "TGA Safety Alerts", "AU",
        ),
        _run_scraper(
            EUDAMEDFSNScraper, "get_recent",
            {"days": days},
            "EUDAMED FSN", "EU",
        ),
        _run_scraper(
            HCIncidentsScraper, "get_recent",
            {"days": days},
            "HC Incidents", "CA",
        ),
    ]

    gathered = await asyncio.gather(*tasks)

    by_source: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    all_actions: list[dict[str, Any]] = []

    for source_label, actions, _error in gathered:
        by_source[source_label] = len(actions)
        for action in actions:
            sev = _classify_severity(action)
            by_severity[sev] = by_severity.get(sev, 0) + 1
            all_actions.append(action)

    # Sort newest-first by any date field present
    def _sort_key(a: dict[str, Any]) -> str:
        for k in ("date_issued", "date", "issued_date", "created_at"):
            v = a.get(k)
            if v:
                return str(v)
        return ""

    all_actions.sort(key=_sort_key, reverse=True)

    return EnforcementOverview(
        total_actions=len(all_actions),
        by_source=by_source,
        by_severity=by_severity,
        recent=all_actions[:20],
        period_days=days,
    )


@router.get("/warning-letters")
async def fda_warning_letters(
    days: int = Query(90, ge=1, le=730, description="Lookback window in days"),
    device_type: str = Query("", description="Optional device type filter"),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    FDA Warning Letters for medical devices.

    Fetches recent letters from the FDA Warning Letters scraper. Supports
    optional device_type filter to narrow results.
    """
    if not _scraper_available(FDAWarningLettersScraper):
        raise HTTPException(
            status_code=503,
            detail="FDAWarningLettersScraper not available. Install app.scrapers.fda_warning_letters.",
        )

    _, actions, error = await _run_scraper(
        FDAWarningLettersScraper, "get_recent",
        {"days": days, "device_type": device_type} if device_type else {"days": days},
        "FDA Warning Letters", "US",
    )

    if error and not actions:
        raise HTTPException(status_code=502, detail=f"FDA Warning Letters scraper error: {error}")

    payload: dict[str, Any] = {
        "source": "FDA Warning Letters",
        "country": "US",
        "days": days,
        "count": len(actions),
        "warning_letters": actions,
    }
    if error:
        payload["warning"] = error

    try:
        from app.services.crypto_signer import sign_payload
        return JSONResponse(content=sign_payload(payload))
    except Exception:
        return payload


@router.get("/warning-letters/patterns", response_model=CitationPatternsResponse)
async def warning_letter_citation_patterns(
    days: int = Query(90, ge=1, le=730, description="Lookback window in days"),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Analyse CFR citation patterns across FDA Warning Letters.

    Returns the most frequently cited regulations, device types affected, and
    example letter references — useful for identifying systemic compliance gaps.
    """
    if not _scraper_available(FDAWarningLettersScraper):
        raise HTTPException(
            status_code=503,
            detail="FDAWarningLettersScraper not available.",
        )

    _, actions, error = await _run_scraper(
        FDAWarningLettersScraper, "get_recent",
        {"days": days},
        "FDA Warning Letters", "US",
    )

    if error and not actions:
        raise HTTPException(status_code=502, detail=f"Scraper error: {error}")

    # Count citations
    citation_counts: dict[str, dict[str, Any]] = {}

    for letter in actions:
        citations = letter.get("citations", [])
        if isinstance(citations, str):
            citations = [citations]
        device_type = letter.get("device_type", letter.get("subject", ""))
        letter_id = letter.get("action_id", letter.get("id", letter.get("url", "")))

        for citation in citations:
            if not citation:
                continue
            if citation not in citation_counts:
                citation_counts[citation] = {
                    "count": 0,
                    "regulation": letter.get("regulation", ""),
                    "device_types": set(),
                    "recent_letters": [],
                }
            citation_counts[citation]["count"] += 1
            if device_type:
                citation_counts[citation]["device_types"].add(str(device_type))
            if letter_id and len(citation_counts[citation]["recent_letters"]) < 5:
                citation_counts[citation]["recent_letters"].append(str(letter_id))

    top_citations = sorted(
        citation_counts.items(), key=lambda x: x[1]["count"], reverse=True
    )[:20]

    patterns = [
        CitationPattern(
            citation=cite,
            count=data["count"],
            regulation=data["regulation"],
            device_types=sorted(data["device_types"])[:10],
            recent_letters=data["recent_letters"][:5],
        )
        for cite, data in top_citations
    ]

    return CitationPatternsResponse(
        total_letters_analyzed=len(actions),
        top_citations=patterns,
        analysis_period_days=days,
    )


@router.get("/tga-alerts")
async def tga_safety_alerts(
    days: int = Query(90, ge=1, le=730, description="Lookback window in days"),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    TGA (Therapeutic Goods Administration) Safety Alerts — Australia.

    Returns medical device hazard alerts and safety-related product actions
    issued by Australia's TGA.
    """
    if not _scraper_available(TGASafetyAlertsScraper):
        raise HTTPException(
            status_code=503,
            detail="TGASafetyAlertsScraper not available. Install app.scrapers.tga_alerts.",
        )

    _, actions, error = await _run_scraper(
        TGASafetyAlertsScraper, "get_recent",
        {"days": days},
        "TGA Safety Alerts", "AU",
    )

    if error and not actions:
        raise HTTPException(status_code=502, detail=f"TGA scraper error: {error}")

    payload: dict[str, Any] = {
        "source": "TGA Safety Alerts",
        "country": "AU",
        "days": days,
        "count": len(actions),
        "alerts": actions,
    }
    if error:
        payload["warning"] = error

    try:
        from app.services.crypto_signer import sign_payload
        return JSONResponse(content=sign_payload(payload))
    except Exception:
        return payload


@router.get("/fsn")
async def eudamed_fsn(
    days: int = Query(90, ge=1, le=730, description="Lookback window in days"),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    EUDAMED Field Safety Notices (FSNs) — European Union.

    Field Safety Notices are manufacturer-issued communications about safety
    corrections or removals of medical devices placed on the EU market.
    """
    if not _scraper_available(EUDAMEDFSNScraper):
        raise HTTPException(
            status_code=503,
            detail="EUDAMEDFSNScraper not available. Install app.scrapers.eudamed_fsn.",
        )

    _, notices, error = await _run_scraper(
        EUDAMEDFSNScraper, "get_recent",
        {"days": days},
        "EUDAMED FSN", "EU",
    )

    if error and not notices:
        raise HTTPException(status_code=502, detail=f"EUDAMED FSN scraper error: {error}")

    payload: dict[str, Any] = {
        "source": "EUDAMED Field Safety Notices",
        "country": "EU",
        "days": days,
        "count": len(notices),
        "notices": notices,
    }
    if error:
        payload["warning"] = error

    try:
        from app.services.crypto_signer import sign_payload
        return JSONResponse(content=sign_payload(payload))
    except Exception:
        return payload


@router.get("/fsn/{device_name}/risk-profile", response_model=FSNRiskProfile)
async def fsn_risk_profile(
    device_name: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    FSN risk profile for a specific device name.

    Aggregates all EUDAMED Field Safety Notices for the given device and
    derives a risk level (CRITICAL / HIGH / MODERATE / LOW) based on the
    severity distribution and total FSN count.
    """
    if not _scraper_available(EUDAMEDFSNScraper):
        raise HTTPException(
            status_code=503,
            detail="EUDAMEDFSNScraper not available.",
        )

    _, notices, error = await _run_scraper(
        EUDAMEDFSNScraper, "get_for_device",
        {"device_name": device_name},
        "EUDAMED FSN", "EU",
    )

    if error and not notices:
        raise HTTPException(status_code=502, detail=f"EUDAMED FSN scraper error: {error}")

    severity_breakdown: dict[str, int] = {}
    hazards: list[str] = []
    countries: set[str] = set()
    dates: list[str] = []

    for notice in notices:
        sev = _classify_severity(notice)
        severity_breakdown[sev] = severity_breakdown.get(sev, 0) + 1
        for hk in ("hazard", "hazard_type", "issue_type"):
            h = notice.get(hk)
            if h and str(h) not in hazards:
                hazards.append(str(h))
        for ck in ("country", "affected_countries", "market"):
            c = notice.get(ck)
            if isinstance(c, list):
                countries.update(c)
            elif c:
                countries.add(str(c))
        for dk in ("date_issued", "date", "fsn_date"):
            d = notice.get(dk)
            if d:
                dates.append(str(d))
                break

    dates.sort()

    # Risk level: any critical → CRITICAL; >5 notices → HIGH; >1 → MODERATE; else LOW
    if severity_breakdown.get("CRITICAL", 0) > 0:
        risk_level = "CRITICAL"
    elif len(notices) > 5:
        risk_level = "HIGH"
    elif len(notices) > 1:
        risk_level = "MODERATE"
    elif len(notices) == 1:
        risk_level = "LOW"
    else:
        risk_level = "NONE"

    return FSNRiskProfile(
        device_name=device_name,
        total_fsns=len(notices),
        severity_breakdown=severity_breakdown,
        most_common_hazards=hazards[:10],
        affected_countries=sorted(countries),
        earliest_fsn=dates[0] if dates else None,
        latest_fsn=dates[-1] if dates else None,
        risk_level=risk_level,
        notices=notices,
    )


@router.get("/hc-incidents")
async def health_canada_incidents(
    days: int = Query(90, ge=1, le=730, description="Lookback window in days"),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Health Canada Medical Device Incident Reports.

    Returns mandatory problem reports submitted to Health Canada by
    manufacturers and importers of medical devices sold in Canada.
    """
    if not _scraper_available(HCIncidentsScraper):
        raise HTTPException(
            status_code=503,
            detail="HCIncidentsScraper not available. Install app.scrapers.hc_incidents.",
        )

    _, incidents, error = await _run_scraper(
        HCIncidentsScraper, "get_recent",
        {"days": days},
        "HC Incidents", "CA",
    )

    if error and not incidents:
        raise HTTPException(status_code=502, detail=f"Health Canada scraper error: {error}")

    payload: dict[str, Any] = {
        "source": "Health Canada Incident Reports",
        "country": "CA",
        "days": days,
        "count": len(incidents),
        "incidents": incidents,
    }
    if error:
        payload["warning"] = error

    try:
        from app.services.crypto_signer import sign_payload
        return JSONResponse(content=sign_payload(payload))
    except Exception:
        return payload


@router.get("/device/{device_name}", response_model=CrossSourceHistory)
async def device_enforcement_history(
    device_name: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Cross-source enforcement history for a specific device.

    Queries FDA Warning Letters, TGA Safety Alerts, EUDAMED FSNs, and Health
    Canada Incidents in parallel and returns a unified view of all enforcement
    actions involving the specified device name.
    """
    tasks = [
        _run_scraper(
            FDAWarningLettersScraper, "get_for_device",
            {"device_name": device_name},
            "FDA Warning Letters", "US",
        ),
        _run_scraper(
            TGASafetyAlertsScraper, "get_for_device",
            {"device_name": device_name},
            "TGA Safety Alerts", "AU",
        ),
        _run_scraper(
            EUDAMEDFSNScraper, "get_for_device",
            {"device_name": device_name},
            "EUDAMED FSN", "EU",
        ),
        _run_scraper(
            HCIncidentsScraper, "get_for_device",
            {"device_name": device_name},
            "HC Incidents", "CA",
        ),
    ]

    gathered = await asyncio.gather(*tasks)

    sources_checked: list[str] = []
    actions_by_source: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    total = 0

    for source_label, actions, error in gathered:
        sources_checked.append(source_label)
        if error:
            errors[source_label] = error
        if actions:
            actions_by_source[source_label] = actions
            total += len(actions)

    return CrossSourceHistory(
        device_name=device_name,
        total_actions=total,
        sources_checked=sources_checked,
        actions_by_source=actions_by_source,
        errors=errors,
    )
