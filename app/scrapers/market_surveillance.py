"""
OrthoLink -- EUDAMED Market Surveillance Scraper (EU)

Source: EUDAMED Market Surveillance REST API
  GET https://ec.europa.eu/tools/eudamed/api/market-surveillance/actions
  Params: status, pageSize, page, countryCode, deviceName

Parses: action_id, device_name, nbr_organization, action_type,
        member_states, start_date, end_date, status, description.

Usage:
    from app.scrapers.market_surveillance import market_surveillance_scraper
    actions = await market_surveillance_scraper.fetch_active_actions()
    history = await market_surveillance_scraper.get_device_enforcement_history("knee implant")
    summary = await market_surveillance_scraper.get_country_summary("DE")
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.scrapers.base_scraper import (
    BaseEnforcementScraper,
    EnforcementAction,
    Severity,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MS_API_BASE = "https://ec.europa.eu/tools/eudamed/api/market-surveillance"
_MS_ACTIONS_URL = f"{_MS_API_BASE}/actions"
_EUDAMED_BASE = "https://ec.europa.eu/tools/eudamed"

_DEFAULT_PAGE_SIZE = 100
_MAX_PAGES = 20

# Action type → severity mapping
_ACTION_TYPE_SEVERITY: dict[str, Severity] = {
    "restriction": Severity.HIGH,
    "withdrawal": Severity.CRITICAL,
    "recall": Severity.HIGH,
    "corrective_action": Severity.HIGH,
    "corrective action": Severity.HIGH,
    "safety_alert": Severity.HIGH,
    "safety alert": Severity.HIGH,
    "prohibition": Severity.CRITICAL,
    "suspension": Severity.CRITICAL,
    "ban": Severity.CRITICAL,
    "advisory": Severity.MEDIUM,
    "information": Severity.LOW,
    "monitoring": Severity.LOW,
    "investigation": Severity.MEDIUM,
}

# Canonical action type normalisation
_ACTION_TYPE_MAP: dict[str, str] = {
    "restriction": "restriction",
    "withdraw": "withdrawal",
    "removal": "withdrawal",
    "recall": "recall",
    "corrective": "corrective_action",
    "safety alert": "safety_alert",
    "prohibition": "restriction",
    "ban": "restriction",
    "suspension": "restriction",
    "advisory": "safety_alert",
    "information": "safety_alert",
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MarketSurveillanceAction(EnforcementAction):
    """EUDAMED Market Surveillance Action — extends unified EnforcementAction."""

    action_id: str = Field(default="", description="EUDAMED market surveillance action ID")
    nbr_organization: str = Field(
        default="", description="Notified Body or NCA organisation responsible"
    )
    member_states: list[str] = Field(
        default_factory=list,
        description="EU member states (ISO 3166-1 alpha-2) where action applies",
    )
    start_date: Optional[datetime] = Field(None, description="Date the action was initiated")
    end_date: Optional[datetime] = Field(None, description="Date the action was closed (None = ongoing)")
    status: str = Field(
        default="active",
        description="Action status: active | closed | pending",
    )
    action_category: str = Field(
        default="",
        description="restriction | withdrawal | recall | corrective_action | safety_alert",
    )


class CountrySurveillanceSummary(BaseModel):
    """Aggregated market surveillance summary for a single EU member state."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    total_actions: int = Field(default=0)
    active_actions: int = Field(default=0)
    by_action_type: dict[str, int] = Field(default_factory=dict)
    by_severity: dict[str, int] = Field(default_factory=dict)
    most_affected_devices: list[str] = Field(
        default_factory=list, description="Top 10 devices with most enforcement actions"
    )
    latest_action_date: Optional[datetime] = Field(None)
    actions: list[MarketSurveillanceAction] = Field(
        default_factory=list, description="All actions for this country (capped at 50)"
    )


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


class MarketSurveillanceScraper(BaseEnforcementScraper):
    """
    Scrapes EUDAMED Market Surveillance Actions via the public REST API.

    Covers restriction, withdrawal, recall, corrective action, and safety alert
    actions issued by EU National Competent Authorities (NCAs).
    """

    def __init__(self) -> None:
        super().__init__()
        self._api_headers = {
            **self._headers,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def get_source_name(self) -> str:
        return "EUDAMED Market Surveillance (EU)"

    # -- Classification helpers ---------------------------------------------

    @staticmethod
    def _normalise_action_type(raw: str) -> str:
        """Map raw action type string to a canonical value."""
        lowered = (raw or "").lower().strip()
        for keyword, canonical in _ACTION_TYPE_MAP.items():
            if keyword in lowered:
                return canonical
        return "corrective_action"

    @staticmethod
    def _classify_ms_severity(action_type: str, description: str = "") -> Severity:
        """Map normalised action type and description to Severity."""
        lowered_type = (action_type or "").lower()
        sev = _ACTION_TYPE_SEVERITY.get(lowered_type, Severity.MEDIUM)

        desc_lower = (description or "").lower()
        if any(kw in desc_lower for kw in ("death", "fatal", "life-threatening")):
            return Severity.CRITICAL
        if any(kw in desc_lower for kw in ("serious injury", "urgent")):
            if sev in (Severity.LOW, Severity.MEDIUM):
                return Severity.HIGH

        return sev

    # -- Record parsing -----------------------------------------------------

    def _parse_action_record(self, rec: dict[str, Any]) -> Optional[MarketSurveillanceAction]:
        """
        Convert a single EUDAMED market surveillance API record.

        Observed API fields:
          - id/actionId, deviceName, nbrOrganization, actionType,
            memberStates (list), startDate, endDate, status, description,
            noticeUrl
        """
        try:
            action_id = (
                rec.get("id")
                or rec.get("actionId")
                or rec.get("msId", "")
            )
            device_name = (
                rec.get("deviceName")
                or rec.get("device", {}).get("name", "")
                or rec.get("productName", "")
                or ""
            )
            nbr_org = (
                rec.get("nbrOrganization")
                or rec.get("nca")
                or rec.get("organization", "")
                or ""
            )
            if isinstance(nbr_org, dict):
                nbr_org = nbr_org.get("name", "")

            raw_action_type = (
                rec.get("actionType")
                or rec.get("type")
                or rec.get("msActionType", "")
                or ""
            )
            action_category = self._normalise_action_type(str(raw_action_type))

            description = (
                rec.get("description")
                or rec.get("summary")
                or rec.get("details", "")
                or ""
            )
            status = (rec.get("status") or "active").lower()

            # Member states — may be list of codes or list of dicts
            raw_states = rec.get("memberStates") or rec.get("countries") or []
            member_states: list[str] = []
            for s in raw_states:
                if isinstance(s, str):
                    member_states.append(s.upper())
                elif isinstance(s, dict):
                    code = s.get("code") or s.get("countryCode") or s.get("isoCode", "")
                    if code:
                        member_states.append(code.upper())

            # Dates
            start_raw = rec.get("startDate") or rec.get("start_date") or rec.get("date", "")
            end_raw = rec.get("endDate") or rec.get("end_date") or ""
            start_date = self._parse_date(str(start_raw)) if start_raw else None
            end_date = self._parse_date(str(end_raw)) if end_raw else None

            # Source URL
            notice_url = rec.get("noticeUrl") or rec.get("url") or ""
            if notice_url and not notice_url.startswith("http"):
                notice_url = f"{_EUDAMED_BASE}{notice_url}"

            severity = self._classify_ms_severity(action_category, description)

            return MarketSurveillanceAction(
                action_type=action_category,
                date=start_date,
                device_name=self._truncate(device_name, 300),
                company=str(nbr_org),
                description=self._truncate(description, 2000),
                severity=severity,
                source_url=notice_url or _MS_ACTIONS_URL,
                country="EU",
                raw_data=rec,
                # MS-specific fields
                action_id=str(action_id),
                nbr_organization=str(nbr_org),
                member_states=member_states,
                start_date=start_date,
                end_date=end_date,
                status=status,
                action_category=action_category,
            )
        except Exception as exc:
            logger.debug(
                "MS action parse error for %s: %s", rec.get("id", "?"), exc
            )
            return None

    # -- API fetch with pagination ------------------------------------------

    async def _fetch_page(
        self,
        page: int = 0,
        page_size: int = _DEFAULT_PAGE_SIZE,
        status: str = "active",
        country_code: Optional[str] = None,
        device_name: Optional[str] = None,
        from_date: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Fetch one page from the EUDAMED market surveillance API."""
        params: dict[str, Any] = {
            "pageSize": page_size,
            "page": page,
            "sort": "startDate,desc",
        }
        if status:
            params["status"] = status
        if country_code:
            params["countryCode"] = country_code
        if device_name:
            params["deviceName"] = device_name
        if from_date:
            params["fromDate"] = from_date

        try:
            data = await self._get_json(
                _MS_ACTIONS_URL,
                params=params,
                headers=self._api_headers,
            )
        except Exception as exc:
            logger.warning("EUDAMED MS API page=%d failed: %s", page, exc)
            return [], 0

        records: list[dict[str, Any]] = (
            data.get("content")
            or data.get("items")
            or data.get("results")
            or data.get("data")
            or (data if isinstance(data, list) else [])
        )
        total: int = int(
            data.get("totalElements")
            or data.get("total")
            or data.get("count")
            or len(records)
        )
        return records, total

    async def _fetch_all_pages(
        self,
        status: str = "active",
        country_code: Optional[str] = None,
        device_name: Optional[str] = None,
        from_date: Optional[str] = None,
        max_records: int = 500,
    ) -> list[MarketSurveillanceAction]:
        """Paginate through all matching market surveillance actions."""
        all_actions: list[MarketSurveillanceAction] = []
        page = 0
        fetched = 0

        while page < _MAX_PAGES and fetched < max_records:
            records, total = await self._fetch_page(
                page=page,
                page_size=min(_DEFAULT_PAGE_SIZE, max_records - fetched),
                status=status,
                country_code=country_code,
                device_name=device_name,
                from_date=from_date,
            )
            if not records:
                break

            for rec in records:
                action = self._parse_action_record(rec)
                if action:
                    all_actions.append(action)

            fetched += len(records)
            page += 1

            if len(records) < _DEFAULT_PAGE_SIZE:
                break
            if total and fetched >= total:
                break

        logger.info(
            "EUDAMED MS: fetched %d actions (pages=%d, status=%s)", len(all_actions), page, status
        )
        return self.deduplicate(all_actions)

    # -- Public API ---------------------------------------------------------

    async def fetch_active_actions(self) -> list[MarketSurveillanceAction]:
        """
        Fetch all currently active market surveillance actions from EUDAMED.

        Active = status is 'active' (ongoing restriction, withdrawal, recall, etc.)
        """
        return await self._fetch_all_pages(status="active")

    async def get_device_enforcement_history(
        self, device_name: str
    ) -> list[MarketSurveillanceAction]:
        """
        Fetch complete enforcement history for a device across all statuses.

        Queries both active and closed actions to provide full history.
        """
        # Fetch active + closed in parallel (sequential to stay polite to EUDAMED)
        active = await self._fetch_all_pages(
            status="active", device_name=device_name, max_records=200
        )
        closed = await self._fetch_all_pages(
            status="closed", device_name=device_name, max_records=200
        )

        # Merge and deduplicate
        all_actions = active + closed
        # Post-filter for device name precision
        dev_lower = device_name.lower()
        matched = [
            a for a in all_actions
            if dev_lower in a.device_name.lower() or dev_lower in a.description.lower()
        ]

        # Return matched if non-empty, else all fetched (API may have done the filtering)
        return self.deduplicate(matched if matched else all_actions)

    async def get_country_summary(
        self, country_code: str
    ) -> CountrySurveillanceSummary:
        """
        Build a CountrySurveillanceSummary for a specific EU member state.

        Queries both active and closed actions for the country, then aggregates.
        """
        country_upper = country_code.upper()

        active = await self._fetch_all_pages(
            status="active", country_code=country_upper, max_records=200
        )
        closed = await self._fetch_all_pages(
            status="closed", country_code=country_upper, max_records=200
        )

        all_actions = active + closed

        # Also filter by member_states in case country_code filtering is imprecise
        country_actions = [
            a for a in all_actions
            if country_upper in a.member_states or not a.member_states
        ]
        country_actions = country_actions if country_actions else all_actions

        if not country_actions:
            return CountrySurveillanceSummary(
                country_code=country_upper,
                total_actions=0,
                active_actions=0,
                by_action_type={},
                by_severity={},
                most_affected_devices=[],
                latest_action_date=None,
                actions=[],
            )

        type_counter: Counter[str] = Counter()
        sev_counter: Counter[str] = Counter()
        device_counter: Counter[str] = Counter()
        latest_date: Optional[datetime] = None

        for a in country_actions:
            type_counter[a.action_category] += 1
            sev_counter[a.severity.value] += 1
            if a.device_name:
                device_counter[a.device_name] += 1
            if a.start_date:
                if latest_date is None or a.start_date > latest_date:
                    latest_date = a.start_date

        return CountrySurveillanceSummary(
            country_code=country_upper,
            total_actions=len(country_actions),
            active_actions=sum(1 for a in country_actions if a.status == "active"),
            by_action_type=dict(type_counter),
            by_severity=dict(sev_counter),
            most_affected_devices=[
                device for device, _ in device_counter.most_common(10)
            ],
            latest_action_date=latest_date,
            actions=country_actions[:50],  # cap payload size
        )

    # -- BaseEnforcementScraper contract ------------------------------------

    async def fetch_recent(self, days: int = 30) -> list[EnforcementAction]:
        """
        Fetch market surveillance actions initiated within the last *days* days.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        from_date = cutoff.strftime("%Y-%m-%d")

        # Fetch active and recent-closed
        actions = await self._fetch_all_pages(
            status="active", from_date=from_date
        )
        # Post-filter by start_date
        filtered = [
            a for a in actions
            if not a.start_date or a.start_date >= cutoff
        ]
        return filtered  # type: ignore[return-value]

    async def fetch_by_device(self, device_name: str) -> list[EnforcementAction]:
        """Fetch enforcement history for a device name."""
        return await self.get_device_enforcement_history(device_name)  # type: ignore[return-value]

    async def search(self, query: str, limit: int = 50) -> list[EnforcementAction]:
        """Free-text search across EUDAMED market surveillance actions."""
        actions = await self._fetch_all_pages(
            device_name=query, max_records=limit, status=""
        )
        q_lower = query.lower()
        matched = [
            a for a in actions
            if (
                q_lower in a.device_name.lower()
                or q_lower in a.description.lower()
                or q_lower in a.nbr_organization.lower()
            )
        ]
        return (matched if matched else actions)[:limit]  # type: ignore[return-value]

    # -- Health check -------------------------------------------------------

    async def health_check(self) -> bool:
        """Verify EUDAMED market surveillance API is reachable."""
        try:
            records, _ = await self._fetch_page(page=0, page_size=1)
            return True
        except Exception as exc:
            logger.warning("EUDAMED MS health_check failed: %s", exc)
            return False


# Singleton
market_surveillance_scraper = MarketSurveillanceScraper()
