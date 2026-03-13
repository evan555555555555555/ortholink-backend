"""
RAA alert and subscription store.

Dual-layer persistence:
  1. In-memory cache (fast reads, survives within process lifetime)
  2. Supabase (persistent across restarts)

Tables required in Supabase:
  raa_alerts        (id uuid pk, country text, document_id text, payload jsonb,
                     notified_orgs text[], created_at timestamptz default now())
  raa_subscriptions (org_id text, country text, primary key(org_id, country))

When SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not set, the store operates
in pure in-memory mode (useful for local dev and tests).
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

AlertEventDict = dict  # AlertEvent.model_dump() or equivalent

# ── In-memory layer ──────────────────────────────────────────────────────────
_alerts: list[AlertEventDict] = []
_subscriptions: dict[str, set[str]] = {}  # org_id -> set of country codes
_MAX_ALERTS = 1000
_supabase_loaded = False  # have we done the initial load from Supabase?


# ── Supabase client factory ──────────────────────────────────────────────────

def _get_supabase():
    """Return a Supabase client or None if not configured."""
    url = os.getenv("SUPABASE_URL", "")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        or os.getenv("SUPABASE_SERVICE_KEY", "")  # legacy fallback
        or os.getenv("SUPABASE_KEY", "")
    )
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as e:
        logger.warning(f"alert_store: Supabase client creation failed: {e}")
        return None


# ── Bootstrap: load persisted alerts + subscriptions on first use ─────────────

def _bootstrap_from_supabase() -> None:
    """Load existing alerts and subscriptions from Supabase into memory cache."""
    global _alerts, _subscriptions, _supabase_loaded
    if _supabase_loaded:
        return
    _supabase_loaded = True
    sb = _get_supabase()
    if sb is None:
        return
    try:
        # Load alerts (most recent 1000)
        resp = (
            sb.table("raa_alerts")
            .select("*")
            .order("created_at", desc=True)
            .limit(_MAX_ALERTS)
            .execute()
        )
        rows = resp.data or []
        for row in reversed(rows):  # oldest first so in-memory list is chronological
            payload = row.get("payload") or {}
            payload.setdefault("country", row.get("country", ""))
            payload.setdefault("document_id", row.get("document_id"))
            payload.setdefault("notified_orgs", row.get("notified_orgs") or [])
            payload.setdefault("_alert_id", row.get("id"))
            _alerts.append(payload)
        logger.info(f"alert_store: loaded {len(rows)} alerts from Supabase")
    except Exception as e:
        logger.warning(f"alert_store: failed to load alerts from Supabase: {e}")

    try:
        # Load subscriptions
        resp2 = sb.table("raa_subscriptions").select("org_id, country").execute()
        for row in (resp2.data or []):
            org_id = row.get("org_id", "")
            country = row.get("country", "")
            if org_id and country:
                _subscriptions.setdefault(org_id, set()).add(country.upper())
        logger.info(f"alert_store: loaded subscriptions for {len(_subscriptions)} orgs")
    except Exception as e:
        logger.warning(f"alert_store: failed to load subscriptions from Supabase: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

def add_alert(event: AlertEventDict) -> None:
    """
    Append an alert event to the in-memory cache and persist to Supabase.
    Caps in-memory count at _MAX_ALERTS (oldest dropped).
    """
    global _alerts
    _bootstrap_from_supabase()

    _alerts.append(event)
    if len(_alerts) > _MAX_ALERTS:
        _alerts = _alerts[-_MAX_ALERTS:]

    logger.info(f"RAA alert stored: {event.get('country')} {event.get('document_id')}")

    # Persist to Supabase asynchronously (best-effort; never blocks the caller)
    sb = _get_supabase()
    if sb is not None:
        try:
            sb.table("raa_alerts").insert({
                "country": (event.get("country") or "").upper(),
                "document_id": event.get("document_id"),
                "payload": event,
                "notified_orgs": event.get("notified_orgs") or [],
            }).execute()
        except Exception as e:
            logger.warning(f"alert_store: Supabase insert failed (alert kept in memory): {e}")


def get_alerts(
    org_id: Optional[str] = None,
    country: Optional[str] = None,
    limit: int = 50,
) -> list[AlertEventDict]:
    """
    List recent alerts from the in-memory cache (backed by Supabase on first call).
    If org_id given, only alerts that notified that org.
    If country given, filter by country.
    Returns most recent first.
    """
    _bootstrap_from_supabase()
    out = _alerts[::-1]  # newest first
    if org_id:
        out = [a for a in out if org_id in (a.get("notified_orgs") or [])]
    if country:
        out = [a for a in out if (a.get("country") or "").upper() == (country or "").upper()]
    return out[:limit]


def subscribe(org_id: str, country: str) -> None:
    """Subscribe org to alerts for a country; persist to Supabase."""
    _bootstrap_from_supabase()
    _subscriptions.setdefault(org_id, set()).add(country.upper())
    logger.info(f"RAA subscribe: org={org_id} country={country}")

    sb = _get_supabase()
    if sb is not None:
        try:
            sb.table("raa_subscriptions").upsert(
                {"org_id": org_id, "country": country.upper()},
                on_conflict="org_id,country",
            ).execute()
        except Exception as e:
            logger.warning(f"alert_store: Supabase upsert subscription failed: {e}")


def get_subscribed_orgs(country: str) -> list[str]:
    """Return org_ids subscribed to this country."""
    _bootstrap_from_supabase()
    return [oid for oid, countries in _subscriptions.items() if country.upper() in countries]


def get_subscriptions(org_id: str) -> list[str]:
    """Return list of country codes the org is subscribed to."""
    _bootstrap_from_supabase()
    return list(_subscriptions.get(org_id, set()))


def clear_for_tests() -> None:
    """Reset store (tests only). Does NOT touch Supabase."""
    global _alerts, _subscriptions, _supabase_loaded
    _alerts = []
    _subscriptions = {}
    _supabase_loaded = True  # prevent bootstrap from overwriting test state
