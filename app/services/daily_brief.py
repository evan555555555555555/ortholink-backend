"""
Daily Regulatory Intelligence Brief — Agency Pattern: ComplianceAuditor + Legal Compliance Checker

Runs on a cron (every 24 h) via APScheduler.
Philosophy borrowed from agency-agents/specialized/compliance-auditor.md:
  "Automate evidence collection — manual evidence is fragile; systems are reliable."
  "Substance over checkboxes."

No LLM required — pure data aggregation from:
  1. RAA alert store (recent 24 h of detected regulatory changes)
  2. FAISS vector store (live chunk counts per country)
  3. Integrity job store (recent agent run quality metrics)

Output stored in job_store as agent="daily_brief" and signed by CryptoSigner.
Retrieve via GET /api/v1/briefing/latest
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Minimum acceptable chunk counts per country (Reality Checker thresholds)
# Below WARN = flagged as LOW_COVERAGE; below CRITICAL = flagged as CRITICAL_GAP
_COVERAGE_THRESHOLDS: dict[str, dict[str, int]] = {
    "US":  {"warn": 5000, "critical": 2000},
    "EU":  {"warn": 2000, "critical": 500},
    "AU":  {"warn": 3000, "critical": 1000},
    "JP":  {"warn": 2000, "critical": 500},
    "KR":  {"warn": 1500, "critical": 300},
    "IN":  {"warn": 800,  "critical": 200},
    "UK":  {"warn": 600,  "critical": 150},
    "UA":  {"warn": 300,  "critical": 50},
    "MX":  {"warn": 500,  "critical": 100},
    "SA":  {"warn": 300,  "critical": 50},
    "CA":  {"warn": 100,  "critical": 20},
    "BR":  {"warn": 80,   "critical": 15},
    "CH":  {"warn": 100,  "critical": 20},
    "CN":  {"warn": 150,  "critical": 30},
    "RU":  {"warn": 30,   "critical": 5},
}

# Recommended scrape sources per low-coverage country (ComplianceAuditor remediation roadmap)
_REMEDIATION_SOURCES: dict[str, list[str]] = {
    "BR": ["https://www.gov.br/anvisa/pt-br/assuntos/noticias-anvisa/publicacoes/resolucoes-da-diretoria-colegiada-rdc"],
    "RU": ["https://roszdravnadzor.gov.ru/services/licenses"],
    "CH": ["https://www.swissmedic.ch/swissmedic/en/home/medical-devices/market-authorisation.html"],
    "CN": ["https://www.nmpa.gov.cn/ylqx/index.html"],
    "CA": ["https://www.canada.ca/en/health-canada/services/drugs-health-products/medical-devices/application-information/guidance-documents.html"],
    "SA": ["https://www.sfda.gov.sa/en/medical-devices"],
    "UA": ["https://www.dec.gov.ua/"],
    "MX": ["https://www.gob.mx/cofepris"],
}


def generate_daily_brief() -> dict[str, Any]:
    """
    Aggregate 24-hour regulatory intelligence brief.

    ComplianceAuditor workflow:
      1. Assessment — count alerts by country / drift severity
      2. Gap Analysis — FAISS coverage vs. thresholds
      3. Risk Level — aggregate risk rating
      4. Remediation — prioritised fix list for coverage gaps
      5. Sign & Store — CryptoSigner injects _signed block
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=24)

    # ── Step 1: Collect RAA alerts from last 24 h ─────────────────────────────
    alert_stats = _collect_alert_stats(window_start)

    # ── Step 2: FAISS coverage audit (Reality Checker: default GAPS_FOUND) ───
    coverage = _audit_coverage()

    # ── Step 3: Integrity job quality metrics ────────────────────────────────
    quality = _collect_quality_metrics()

    # ── Step 4: Compute overall risk level ────────────────────────────────────
    risk_level = _compute_risk_level(alert_stats, coverage, quality)

    # ── Step 5: Build brief ───────────────────────────────────────────────────
    brief: dict[str, Any] = {
        "generated_at": now.isoformat(),
        "window_hours": 24,
        "risk_level": risk_level,          # CRITICAL | HIGH | MEDIUM | LOW
        "alert_summary": alert_stats,
        "coverage_audit": coverage,
        "quality_metrics": quality,
        "remediation_priorities": _build_remediation_list(coverage),
        "agent": "daily_brief",
        "disclaimer": (
            "Auto-generated regulatory intelligence brief. "
            "Verify all alerts and coverage gaps with official sources. "
            "Not a substitute for qualified regulatory affairs professional review."
        ),
    }

    return brief


def _collect_alert_stats(window_start: datetime) -> dict[str, Any]:
    """Pull recent alerts from the RAA alert store."""
    try:
        from app.services.alert_store import get_alerts
        all_alerts = get_alerts(limit=500)

        recent = [
            a for a in all_alerts
            if _parse_dt(a.get("detected_at")) >= window_start
        ]

        by_country: dict[str, int] = {}
        drift_counts = {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0, "TRIVIAL": 0, "UNKNOWN": 0}

        for a in recent:
            country = a.get("country", "UNKNOWN")
            by_country[country] = by_country.get(country, 0) + 1
            drift = a.get("drift_label", "UNKNOWN") or "UNKNOWN"
            drift_counts[drift] = drift_counts.get(drift, 0) + 1

        return {
            "total_alerts_24h": len(recent),
            "by_country": by_country,
            "drift_breakdown": drift_counts,
            "critical_drift_count": drift_counts.get("CRITICAL", 0),
        }
    except Exception as e:
        logger.warning("daily_brief: alert collection failed: %s", e)
        return {"total_alerts_24h": 0, "by_country": {}, "drift_breakdown": {}, "critical_drift_count": 0}


def _audit_coverage() -> dict[str, Any]:
    """
    Reality Checker pattern: default verdict is GAPS_FOUND.
    Requires overwhelming proof (all countries above warn threshold) for COVERAGE_OK.
    """
    try:
        from app.tools.vector_store import get_vector_store
        store = get_vector_store()
        countries_in_store = store.get_countries()

        # Get chunk count per country
        counts: dict[str, int] = {}
        for country in countries_in_store:
            try:
                counts[country] = _get_country_chunk_count(store, country)
            except Exception:
                counts[country] = 0

        # Evaluate against thresholds
        critical_gaps: list[dict] = []
        low_coverage: list[dict] = []
        healthy: list[str] = []

        for country, threshold in _COVERAGE_THRESHOLDS.items():
            count = counts.get(country, 0)
            if count < threshold["critical"]:
                critical_gaps.append({
                    "country": country,
                    "chunks": count,
                    "threshold": threshold["critical"],
                    "gap_pct": round((threshold["critical"] - count) / max(threshold["critical"], 1) * 100, 1),
                    "sources": _REMEDIATION_SOURCES.get(country, []),
                })
            elif count < threshold["warn"]:
                low_coverage.append({
                    "country": country,
                    "chunks": count,
                    "threshold": threshold["warn"],
                    "gap_pct": round((threshold["warn"] - count) / max(threshold["warn"], 1) * 100, 1),
                    "sources": _REMEDIATION_SOURCES.get(country, []),
                })
            else:
                healthy.append(country)

        # Reality Checker: COVERAGE_OK requires ALL countries healthy
        verdict = "COVERAGE_OK" if (not critical_gaps and not low_coverage) else (
            "CRITICAL_GAPS" if critical_gaps else "LOW_COVERAGE"
        )

        return {
            "verdict": verdict,           # COVERAGE_OK | LOW_COVERAGE | CRITICAL_GAPS
            "total_countries": len(_COVERAGE_THRESHOLDS),
            "healthy_count": len(healthy),
            "low_coverage_count": len(low_coverage),
            "critical_gap_count": len(critical_gaps),
            "critical_gaps": critical_gaps,
            "low_coverage": low_coverage,
            "healthy_countries": sorted(healthy),
            "chunk_counts": counts,
        }
    except Exception as e:
        logger.warning("daily_brief: coverage audit failed: %s", e)
        return {
            "verdict": "AUDIT_FAILED",
            "error": str(e),
            "critical_gaps": [],
            "low_coverage": [],
        }


def _get_country_chunk_count(store: Any, country: str) -> int:
    """Get chunk count for a country from vector store metadata."""
    try:
        if hasattr(store, "get_chunk_count"):
            return store.get_chunk_count(country)
        # Fallback: iterate metadata list directly
        if hasattr(store, "metadata") and store.metadata:
            return sum(
                1 for m in store.metadata
                if getattr(m, "country", "").upper() == country.upper()
            )
        return 0
    except Exception:
        return 0


def _collect_quality_metrics() -> dict[str, Any]:
    """Check recent job quality from job store integrity reports."""
    try:
        from app.services.job_store import get_all_recent_jobs
        jobs = get_all_recent_jobs(limit=50)

        total = len(jobs)
        reliable = sum(
            1 for j in jobs
            if j.get("result", {}) and
               isinstance(j["result"].get("_integrity"), dict) and
               j["result"]["_integrity"].get("overall_verdict") == "RELIABLE"
        )
        review = sum(
            1 for j in jobs
            if j.get("result", {}) and
               isinstance(j["result"].get("_integrity"), dict) and
               j["result"]["_integrity"].get("overall_verdict") == "REVIEW_REQUIRED"
        )
        unreliable = sum(
            1 for j in jobs
            if j.get("result", {}) and
               isinstance(j["result"].get("_integrity"), dict) and
               j["result"]["_integrity"].get("overall_verdict") == "UNRELIABLE"
        )
        signed = sum(1 for j in jobs if j.get("result", {}) and j["result"].get("_signed"))

        return {
            "jobs_sampled": total,
            "signed": signed,
            "integrity_reliable": reliable,
            "integrity_review_required": review,
            "integrity_unreliable": unreliable,
            "quality_rate": round(reliable / max(total, 1) * 100, 1),
        }
    except Exception as e:
        logger.warning("daily_brief: quality metrics failed: %s", e)
        return {"jobs_sampled": 0, "quality_rate": 0.0}


def _compute_risk_level(
    alerts: dict, coverage: dict, quality: dict
) -> str:
    """
    Aggregate risk: CRITICAL > HIGH > MEDIUM > LOW.
    Reality Checker philosophy: conservative — any critical signal = CRITICAL.
    """
    if (
        alerts.get("critical_drift_count", 0) > 0
        or coverage.get("verdict") == "CRITICAL_GAPS"
        or quality.get("integrity_unreliable", 0) > 0
    ):
        return "CRITICAL"

    if (
        alerts.get("total_alerts_24h", 0) > 5
        or coverage.get("verdict") == "LOW_COVERAGE"
        or quality.get("integrity_review_required", 0) > 2
    ):
        return "HIGH"

    if (
        alerts.get("total_alerts_24h", 0) > 0
        or quality.get("quality_rate", 100) < 90
    ):
        return "MEDIUM"

    return "LOW"


def _build_remediation_list(coverage: dict) -> list[dict]:
    """ComplianceAuditor: prioritised remediation roadmap."""
    items = []

    for gap in coverage.get("critical_gaps", []):
        items.append({
            "priority": "P0_CRITICAL",
            "country": gap["country"],
            "action": f"Scrape missing regulatory content — {gap['chunks']} chunks vs {gap['threshold']} required",
            "sources": gap.get("sources", []),
            "effort": "high",
        })

    for gap in coverage.get("low_coverage", []):
        items.append({
            "priority": "P1_HIGH",
            "country": gap["country"],
            "action": f"Expand coverage — {gap['chunks']} chunks vs {gap['threshold']} recommended",
            "sources": gap.get("sources", []),
            "effort": "medium",
        })

    return items


def _parse_dt(val: Any) -> datetime:
    """Parse ISO datetime string to aware datetime, fallback to epoch."""
    try:
        if isinstance(val, datetime):
            return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
        if isinstance(val, str):
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return datetime.fromtimestamp(0, tz=timezone.utc)


def run_daily_brief_cron() -> None:
    """
    Entry point for APScheduler cron job.
    Generates brief → signs it → stores it in job_store.
    """
    logger.info("daily_brief: starting scheduled run")
    try:
        brief = generate_daily_brief()

        # Sign with CryptoSigner
        try:
            from app.services.crypto_signer import sign_payload
            brief = sign_payload(brief)
        except Exception:
            pass

        # Store in job_store so frontend can poll GET /api/v1/briefing/latest
        from app.services.job_store import create_job, set_completed
        job_id = create_job(agent="daily_brief")
        set_completed(job_id, brief)

        logger.info(
            "daily_brief: complete — risk=%s alerts=%d coverage=%s",
            brief.get("risk_level"),
            brief.get("alert_summary", {}).get("total_alerts_24h", 0),
            brief.get("coverage_audit", {}).get("verdict"),
        )
    except Exception as e:
        logger.error("daily_brief cron failed: %s", e, exc_info=True)
