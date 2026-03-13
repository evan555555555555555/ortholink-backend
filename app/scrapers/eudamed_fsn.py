"""
OrthoLink -- EUDAMED Field Safety Notices Scraper (EU)

Source: EUDAMED REST API
  GET https://ec.europa.eu/tools/eudamed/api/vigilance/fsns
  Params: searchTerm, pageSize, fromDate (ISO 8601), page

Parses: fsn_reference, device_name, manufacturer, hazard_description,
        corrective_action, affected_countries, issue_date, fsn_type,
        nca_reference.

Usage:
    from app.scrapers.eudamed_fsn import eudamed_fsn_scraper
    fsns = await eudamed_fsn_scraper.fetch_fsn_list(days=30)
    profile = await eudamed_fsn_scraper.get_device_risk_profile("knee implant")
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

_EUDAMED_FSN_API = "https://ec.europa.eu/tools/eudamed/api/vigilance/fsns"
_EUDAMED_BASE = "https://ec.europa.eu/tools/eudamed"

_DEFAULT_PAGE_SIZE = 100
_MAX_PAGES = 20

# FSN type → severity mapping
_FSN_TYPE_SEVERITY: dict[str, Severity] = {
    "FSCA": Severity.HIGH,            # Field Safety Corrective Action
    "FSN": Severity.HIGH,             # Field Safety Notice (general)
    "RECALL": Severity.HIGH,          # Device recall
    "URGENT_FIELD_SAFETY": Severity.CRITICAL,   # Urgent FSN
    "ADVISORY": Severity.MEDIUM,
    "INFORMATION": Severity.LOW,
}

# Risk score weights for DeviceRiskProfile
_FSN_RISK_WEIGHT: dict[str, float] = {
    "URGENT_FIELD_SAFETY": 10.0,
    "RECALL": 8.0,
    "FSCA": 6.0,
    "FSN": 4.0,
    "ADVISORY": 2.0,
    "INFORMATION": 1.0,
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class FieldSafetyNotice(EnforcementAction):
    """EUDAMED Field Safety Notice — extends unified EnforcementAction."""

    fsn_reference: str = Field(default="", description="EUDAMED FSN reference number")
    manufacturer: str = Field(default="", description="Device manufacturer name")
    hazard_description: str = Field(default="", description="Clinical/technical hazard identified")
    corrective_action: str = Field(default="", description="Required corrective action from FSN")
    affected_countries: list[str] = Field(
        default_factory=list,
        description="ISO 3166-1 alpha-2 country codes affected by the FSN",
    )
    issue_date: Optional[datetime] = Field(None, description="Date the FSN was issued")
    fsn_type: str = Field(
        default="FSN",
        description="FSN type: FSN | FSCA | RECALL | URGENT_FIELD_SAFETY | ADVISORY | INFORMATION",
    )
    nca_reference: str = Field(
        default="", description="National Competent Authority reference number"
    )


class DeviceRiskProfile(BaseModel):
    """Aggregated FSN risk profile for a specific device."""

    device_name: str = Field(..., description="Device name used for search")
    total_fsns: int = Field(default=0, description="Total number of FSNs found")
    fsn_by_type: dict[str, int] = Field(
        default_factory=dict, description="Count of FSNs by type"
    )
    affected_countries: list[str] = Field(
        default_factory=list, description="Unique countries affected across all FSNs"
    )
    risk_score: float = Field(
        default=0.0,
        description="Weighted risk score (0–100). Higher = more FSN activity = higher risk.",
    )
    latest_fsn_date: Optional[datetime] = Field(
        None, description="Date of the most recent FSN for this device"
    )
    fsns: list[FieldSafetyNotice] = Field(
        default_factory=list, description="All FSNs matching the device name"
    )


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


class EUDAMEDFSNScraper(BaseEnforcementScraper):
    """
    Scrapes EUDAMED Field Safety Notices via the public REST API.

    EUDAMED exposes a paginated JSON endpoint for vigilance FSNs.
    Authentication is not required for read-only public access.
    """

    def __init__(self) -> None:
        super().__init__()
        # EUDAMED API requires JSON Accept header
        self._api_headers = {
            **self._headers,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def get_source_name(self) -> str:
        return "EUDAMED Field Safety Notices (EU)"

    # -- Severity classification --------------------------------------------

    @staticmethod
    def _classify_fsn_severity(fsn_type: str, description: str = "") -> Severity:
        """Map FSN type and description keywords to Severity."""
        upper_type = (fsn_type or "").strip().upper()
        sev = _FSN_TYPE_SEVERITY.get(upper_type, Severity.MEDIUM)

        # Escalate based on description keywords
        desc_lower = description.lower()
        if any(kw in desc_lower for kw in ("death", "fatal", "life-threatening", "urgent")):
            return Severity.CRITICAL
        if any(kw in desc_lower for kw in ("serious injury", "malfunction", "failure")):
            if sev == Severity.LOW or sev == Severity.MEDIUM:
                return Severity.HIGH

        return sev

    # -- Response parsing ---------------------------------------------------

    def _parse_fsn_record(self, rec: dict[str, Any]) -> Optional[FieldSafetyNotice]:
        """
        Convert a single EUDAMED FSN API record to a FieldSafetyNotice.

        EUDAMED API fields (observed from public API responses):
          - fsnReference, deviceName, manufacturerName, hazardDescription,
            correctiveAction, issuedDate, fsnType, affectedCountries (list),
            ncaReference, noticeUrl
        """
        try:
            fsn_ref = rec.get("fsnReference") or rec.get("reference") or rec.get("id", "")
            device_name = rec.get("deviceName") or rec.get("device", {}).get("name", "") or ""
            manufacturer = (
                rec.get("manufacturerName")
                or rec.get("manufacturer", {}).get("name", "")
                or ""
            )
            hazard = rec.get("hazardDescription") or rec.get("hazard") or rec.get("description", "") or ""
            corrective = rec.get("correctiveAction") or rec.get("action", "") or ""
            fsn_type = (
                rec.get("fsnType") or rec.get("type") or rec.get("noticeType", "FSN")
            ).upper()
            nca_ref = rec.get("ncaReference") or rec.get("nca", "") or ""

            # Affected countries — may be list of dicts or list of strings
            raw_countries = rec.get("affectedCountries") or rec.get("countries") or []
            affected_countries: list[str] = []
            for c in raw_countries:
                if isinstance(c, str):
                    affected_countries.append(c.upper())
                elif isinstance(c, dict):
                    code = c.get("code") or c.get("countryCode") or c.get("isoCode", "")
                    if code:
                        affected_countries.append(code.upper())

            # Source URL
            notice_url = rec.get("noticeUrl") or rec.get("url") or ""
            if notice_url and not notice_url.startswith("http"):
                notice_url = f"{_EUDAMED_BASE}{notice_url}"

            # Dates
            issue_date_raw = rec.get("issuedDate") or rec.get("issueDate") or rec.get("date", "")
            issue_date = self._parse_date(str(issue_date_raw)) if issue_date_raw else None

            severity = self._classify_fsn_severity(fsn_type, hazard)

            description_parts = []
            if hazard:
                description_parts.append(f"Hazard: {hazard}")
            if corrective:
                description_parts.append(f"Action: {corrective}")
            description = self._truncate(". ".join(description_parts), 2000)

            return FieldSafetyNotice(
                action_type="field_safety_notice",
                date=issue_date,
                device_name=self._truncate(device_name, 300),
                company=manufacturer,
                description=description,
                severity=severity,
                source_url=notice_url or _EUDAMED_FSN_API,
                country="EU",
                raw_data=rec,
                # FSN-specific fields
                fsn_reference=str(fsn_ref),
                manufacturer=manufacturer,
                hazard_description=self._truncate(hazard, 1000),
                corrective_action=self._truncate(corrective, 500),
                affected_countries=affected_countries,
                issue_date=issue_date,
                fsn_type=fsn_type,
                nca_reference=str(nca_ref),
            )
        except Exception as exc:
            logger.debug("FSN parse error for record %s: %s", rec.get("fsnReference", "?"), exc)
            return None

    # -- API fetch with pagination ------------------------------------------

    async def _fetch_page(
        self,
        page: int = 0,
        page_size: int = _DEFAULT_PAGE_SIZE,
        from_date: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Fetch one page of FSNs from EUDAMED API.

        Returns (records, total_count).
        """
        params: dict[str, Any] = {
            "pageSize": page_size,
            "page": page,
            "sort": "issuedDate,desc",
        }
        if from_date:
            params["fromDate"] = from_date
        if search_term:
            params["searchTerm"] = search_term

        try:
            data = await self._get_json(
                _EUDAMED_FSN_API,
                params=params,
                headers=self._api_headers,
            )
        except Exception as exc:
            logger.warning("EUDAMED FSN API page=%d failed: %s", page, exc)
            return [], 0

        # EUDAMED API may wrap results under different keys depending on version
        records: list[dict[str, Any]] = (
            data.get("content")
            or data.get("items")
            or data.get("results")
            or data.get("data")
            or (data if isinstance(data, list) else [])
        )
        total: int = (
            data.get("totalElements")
            or data.get("total")
            or data.get("count")
            or len(records)
        )
        return records, int(total)

    async def _fetch_all_pages(
        self,
        from_date: Optional[str] = None,
        search_term: Optional[str] = None,
        max_records: int = 500,
    ) -> list[FieldSafetyNotice]:
        """Paginate through EUDAMED FSN API, collecting all FSNs up to max_records."""
        all_fsns: list[FieldSafetyNotice] = []
        page = 0
        fetched = 0

        while page < _MAX_PAGES and fetched < max_records:
            records, total = await self._fetch_page(
                page=page,
                page_size=min(_DEFAULT_PAGE_SIZE, max_records - fetched),
                from_date=from_date,
                search_term=search_term,
            )

            if not records:
                break

            for rec in records:
                fsn = self._parse_fsn_record(rec)
                if fsn:
                    all_fsns.append(fsn)

            fetched += len(records)
            page += 1

            # If we got fewer than page_size records, we've exhausted the source
            if len(records) < _DEFAULT_PAGE_SIZE:
                break

            # If we've collected more than total reported, stop
            if total and fetched >= total:
                break

        logger.info(
            "EUDAMED FSN: fetched %d FSNs (total_reported=%d, pages=%d)",
            len(all_fsns),
            total if "total" in dir() else 0,
            page,
        )
        return self.deduplicate(all_fsns)

    # -- Public API ---------------------------------------------------------

    async def fetch_fsn_list(self, days: int = 30) -> list[FieldSafetyNotice]:
        """
        Fetch EUDAMED Field Safety Notices issued within the last *days* days.

        Returns list of FieldSafetyNotice ordered by issue date descending.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        from_date = cutoff.strftime("%Y-%m-%d")

        fsns = await self._fetch_all_pages(from_date=from_date)

        # Post-filter by date in case API from_date is inclusive but off-by-one
        filtered = [f for f in fsns if not f.issue_date or f.issue_date >= cutoff]
        return filtered

    async def get_device_risk_profile(self, device_name: str) -> DeviceRiskProfile:
        """
        Aggregate all FSNs for a device name and compute a risk profile.

        Risk score is a weighted sum of FSN types (0–100 scale), capped at 100.
        Higher score = more FSN activity = higher risk signal.
        """
        fsns = await self._fetch_all_pages(search_term=device_name, max_records=200)

        # Also filter by device_name in the parsed result for precision
        device_lower = device_name.lower()
        matched_fsns = [
            f for f in fsns
            if device_lower in f.device_name.lower() or device_lower in f.description.lower()
        ]

        if not matched_fsns:
            return DeviceRiskProfile(
                device_name=device_name,
                total_fsns=0,
                fsn_by_type={},
                affected_countries=[],
                risk_score=0.0,
                latest_fsn_date=None,
                fsns=[],
            )

        # Count by type
        type_counter: Counter[str] = Counter()
        all_countries: set[str] = set()
        latest_date: Optional[datetime] = None

        for f in matched_fsns:
            type_counter[f.fsn_type] += 1
            all_countries.update(f.affected_countries)
            if f.issue_date:
                if latest_date is None or f.issue_date > latest_date:
                    latest_date = f.issue_date

        # Compute risk score (weighted, capped at 100)
        raw_score = sum(
            _FSN_RISK_WEIGHT.get(fsn_type, 2.0) * count
            for fsn_type, count in type_counter.items()
        )
        risk_score = min(round(raw_score, 2), 100.0)

        return DeviceRiskProfile(
            device_name=device_name,
            total_fsns=len(matched_fsns),
            fsn_by_type=dict(type_counter),
            affected_countries=sorted(all_countries),
            risk_score=risk_score,
            latest_fsn_date=latest_date,
            fsns=matched_fsns[:50],  # cap at 50 for payload size
        )

    # -- BaseEnforcementScraper contract ------------------------------------

    async def fetch_recent(self, days: int = 30) -> list[EnforcementAction]:
        """Satisfy base class contract — delegates to fetch_fsn_list."""
        return await self.fetch_fsn_list(days=days)  # type: ignore[return-value]

    async def fetch_by_device(self, device_name: str) -> list[EnforcementAction]:
        """Fetch FSNs for a specific device name via EUDAMED search."""
        fsns = await self._fetch_all_pages(search_term=device_name)
        device_lower = device_name.lower()
        matched = [
            f for f in fsns
            if device_lower in f.device_name.lower() or device_lower in f.description.lower()
        ]
        return matched  # type: ignore[return-value]

    async def search(self, query: str, limit: int = 50) -> list[EnforcementAction]:
        """Free-text search against EUDAMED FSN API searchTerm parameter."""
        fsns = await self._fetch_all_pages(search_term=query, max_records=limit)
        return fsns[:limit]  # type: ignore[return-value]

    # -- Health check -------------------------------------------------------

    async def health_check(self) -> bool:
        """Verify EUDAMED FSN API endpoint is reachable."""
        try:
            records, _ = await self._fetch_page(page=0, page_size=1)
            # An empty records list is still a valid healthy response (no FSNs today)
            return True
        except Exception as exc:
            logger.warning("EUDAMED FSN health_check failed: %s", exc)
            return False


# Singleton
eudamed_fsn_scraper = EUDAMEDFSNScraper()
