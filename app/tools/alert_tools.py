"""
RAA (Regulatory Alert Agent) tools — PRD §4.6.

Pipeline: Scrape → Diff (SHA-256 vs chunk_hash) → Classify → Archive (soft-delete)
→ Re-Embed → Notify. NEVER delete old chunks; mark is_active=false, valid_to=today.
"""

import hashlib
import logging
from datetime import datetime, timezone

from app.ingestion.scraper import scrape_url
from app.ingestion.scraper_validator import validate_scraped_content
from app.services.email_service import get_email_service

logger = logging.getLogger(__name__)


def content_hash(text: str) -> str:
    """SHA-256 hash of cleaned text for change detection."""
    cleaned = text.strip().replace("\r\n", "\n")
    return hashlib.sha256(cleaned.encode()).hexdigest()


def scrape_and_hash(url: str) -> tuple[str, str, bool]:
    """
    Scrape URL and return (cleaned_text, sha256_hash, success).
    Validates: word count, legal keywords; rejects if clean_text contains browser noise.
    """
    result = scrape_url(url)
    if not result.success:
        return "", "", False
    text = result.text
    validation = validate_scraped_content(text)
    if not validation.is_valid:
        logger.warning(f"Scraper validation failed for {url}: {validation.errors}")
        return "", "", False
    # Noise check: PRD — if clean_text contains "Firefox" or "Google Chrome", scraper failed
    if "Firefox" in text or "Google Chrome" in text or "We use cookies" in text:
        logger.warning(f"Scraped text contains browser noise for {url}")
        return "", "", False
    return text, content_hash(text), True


def soft_deactivate_chunks(
    vector_store,
    chunk_ids: list[str],
) -> None:
    """
    Mark chunks as inactive; set valid_to=today. NEVER delete.
    Caller must pass the vector store instance (to avoid circular import).
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for meta in vector_store.metadata:
        if meta.chunk_id in chunk_ids:
            meta.is_active = False
            meta.valid_to = today
    vector_store.save()


def notify_subscribers(
    country: str,
    regulation_id: str,
    change_type: str,
    severity: str,
    summary: str,
) -> int:
    """
    Send email via Resend to users subscribed to this country/regulation.
    Returns number of emails sent. In-app notifications are separate (Supabase).

    Flow:
    1. Query raa_subscriptions for orgs subscribed to this country
    2. Query org_members to get admin/reviewer emails for each org
    3. Send alert notification email via Resend to each unique email
    """
    service = get_email_service()
    if not service.is_configured():
        logger.warning("Email service not configured; skipping notify")
        return 0

    # Resolve subscriber emails from Supabase
    try:
        from app.services.alert_store import get_subscribed_orgs
        org_ids = get_subscribed_orgs(country)
        if not org_ids:
            logger.info(f"RAA: no orgs subscribed to {country}; skipping notify")
            return 0

        # Gather recipient emails from org_members (admin + reviewer roles)
        import os
        from supabase import create_client

        url = os.getenv("SUPABASE_URL", "")
        key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
            or os.getenv("SUPABASE_SERVICE_KEY", "")
            or os.getenv("SUPABASE_KEY", "")
        )
        if not url or not key:
            logger.warning("RAA notify: Supabase not configured; skipping email send")
            return 0

        sb = create_client(url, key)
        emails_sent = 0
        seen_emails: set[str] = set()

        for org_id in org_ids:
            try:
                result = (
                    sb.table("org_members")
                    .select("email")
                    .eq("org_id", org_id)
                    .in_("role", ["admin", "reviewer"])
                    .execute()
                )
                for row in (result.data or []):
                    email = (row.get("email") or "").strip()
                    if email and email not in seen_emails:
                        seen_emails.add(email)
                        sent = service.send_alert_notification(
                            to_email=email,
                            country=country,
                            regulation_name=regulation_id,
                            severity=severity,
                            summary=summary,
                        )
                        if sent:
                            emails_sent += 1
            except Exception as e:
                logger.warning(f"RAA notify: failed to fetch members for org {org_id}: {e}")
                continue

        logger.info(
            f"RAA notify: sent {emails_sent} email(s) for {country} {regulation_id} ({severity})"
        )
        return emails_sent

    except Exception as e:
        logger.error(f"RAA notify_subscribers failed: {e}", exc_info=True)
        return 0


# Cron schedule constants (PRD §4.6.2)
SCHEDULE_DAILY = ["US", "EU"]
SCHEDULE_EVERY_3_DAYS = ["UK", "CA", "JP", "AU", "IN", "BR"]
SCHEDULE_WEEKLY = ["UA", "CN", "RU", "CH", "MX", "KR", "SA"]
