"""
OrthoLink -- Health Canada Medical Device Incident Reports Scraper

Source: Health Canada Medical Devices Bureau Incident (MDBI) database
  Base: https://health-products.canada.ca/mdbi-bdim/en
  API:  https://health-products.canada.ca/api/v1/mdbi

Parses: report_number, device_name, manufacturer, incident_date,
        incident_description, outcome, report_type, mdall_licence.

Usage:
    from app.scrapers.hc_incidents import hc_incidents_scraper
    incidents = await hc_incidents_scraper.fetch_recent_incidents(days=30)
    by_licence = await hc_incidents_scraper.fetch_by_licence("12345")
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

from bs4 import BeautifulSoup
from pydantic import Field

from app.scrapers.base_scraper import (
    BaseEnforcementScraper,
    EnforcementAction,
    Severity,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HC_MDBI_API = "https://health-products.canada.ca/api/v1/mdbi"
_HC_MDBI_WEB = "https://health-products.canada.ca/mdbi-bdim/en"
_HC_BASE = "https://health-products.canada.ca"

_DEFAULT_PAGE_SIZE = 100
_MAX_PAGES = 15

# Outcome codes → severity mapping
_OUTCOME_SEVERITY: dict[str, Severity] = {
    "death": Severity.CRITICAL,
    "serious_injury": Severity.HIGH,
    "malfunction": Severity.MEDIUM,
    "other": Severity.LOW,
    "near_miss": Severity.MEDIUM,
}

# Outcome string → canonical outcome code
_OUTCOME_NORMALISE: dict[str, str] = {
    "death": "death",
    "died": "death",
    "fatality": "death",
    "fatal": "death",
    "serious injury": "serious_injury",
    "serious harm": "serious_injury",
    "injury": "serious_injury",
    "hospitali": "serious_injury",
    "malfunction": "malfunction",
    "device failure": "malfunction",
    "failure": "malfunction",
    "near miss": "near_miss",
    "near-miss": "near_miss",
    "other": "other",
    "unknown": "other",
}

OutcomeType = Literal["death", "serious_injury", "malfunction", "near_miss", "other"]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class HCIncidentReport(EnforcementAction):
    """Health Canada Medical Device Incident Report."""

    report_number: str = Field(default="", description="Health Canada incident report number")
    manufacturer: str = Field(default="", description="Device manufacturer name")
    incident_date: Optional[datetime] = Field(None, description="Date of the incident")
    incident_description: str = Field(default="", description="Full incident narrative")
    outcome: str = Field(
        default="other",
        description="Incident outcome: death | serious_injury | malfunction | near_miss | other",
    )
    report_type: str = Field(
        default="",
        description="Report type: mandatory or voluntary",
    )
    mdall_licence: str = Field(
        default="", description="MDALL device licence number (CA regulatory requirement)"
    )


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


class HCIncidentsScraper(BaseEnforcementScraper):
    """
    Scrapes Health Canada Medical Device Incident Reports.

    Uses the MDBI API when available; falls back to HTML scraping of the
    public search interface when the API is unavailable.
    """

    def __init__(self) -> None:
        super().__init__()
        self._api_headers = {
            **self._headers,
            "Accept": "application/json",
        }

    def get_source_name(self) -> str:
        return "Health Canada MDBI Incidents (CA)"

    # -- Outcome normalisation ----------------------------------------------

    @staticmethod
    def _normalise_outcome(raw: str) -> str:
        """Map a raw outcome string to a canonical OutcomeType value."""
        lowered = (raw or "").lower().strip()
        for keyword, code in _OUTCOME_NORMALISE.items():
            if keyword in lowered:
                return code
        return "other"

    @staticmethod
    def _outcome_severity(outcome: str) -> Severity:
        return _OUTCOME_SEVERITY.get(outcome, Severity.MEDIUM)

    # -- API-based parsing --------------------------------------------------

    def _parse_api_record(self, rec: dict[str, Any]) -> Optional[HCIncidentReport]:
        """
        Convert a single MDBI API record to an HCIncidentReport.

        Health Canada MDBI API fields (observed from public API):
          - reportNumber, deviceName, manufacturerName, incidentDate,
            incidentDescription, outcome, reportType, mdallLicenceNumber,
            sourceUrl
        """
        try:
            report_num = (
                rec.get("reportNumber")
                or rec.get("report_number")
                or rec.get("id", "")
            )
            device_name = (
                rec.get("deviceName")
                or rec.get("device_name")
                or rec.get("device", "")
                or ""
            )
            manufacturer = (
                rec.get("manufacturerName")
                or rec.get("manufacturer_name")
                or rec.get("manufacturer", "")
                or ""
            )
            incident_desc = (
                rec.get("incidentDescription")
                or rec.get("incident_description")
                or rec.get("description", "")
                or ""
            )
            outcome_raw = (
                rec.get("outcome")
                or rec.get("outcomeType")
                or rec.get("outcome_type", "other")
            )
            outcome = self._normalise_outcome(str(outcome_raw))
            report_type = (
                rec.get("reportType")
                or rec.get("report_type", "")
                or ""
            )
            licence = (
                rec.get("mdallLicenceNumber")
                or rec.get("mdall_licence")
                or rec.get("licence", "")
                or ""
            )

            # Dates
            incident_date_raw = (
                rec.get("incidentDate")
                or rec.get("incident_date")
                or rec.get("date", "")
            )
            incident_date = self._parse_date(str(incident_date_raw)) if incident_date_raw else None

            report_date_raw = (
                rec.get("reportDate")
                or rec.get("report_date")
                or incident_date_raw
                or ""
            )
            report_date = self._parse_date(str(report_date_raw)) if report_date_raw else incident_date

            # Source URL
            source_url = rec.get("sourceUrl") or rec.get("url") or ""
            if source_url and not source_url.startswith("http"):
                source_url = f"{_HC_BASE}{source_url}"
            if not source_url and report_num:
                source_url = f"{_HC_MDBI_WEB}/search?reportNumber={report_num}"

            severity = self._outcome_severity(outcome)

            return HCIncidentReport(
                action_type="incident_report",
                date=report_date,
                device_name=self._truncate(device_name, 300),
                company=manufacturer,
                description=self._truncate(incident_desc, 2000),
                severity=severity,
                source_url=source_url,
                country="CA",
                raw_data=rec,
                # HC-specific fields
                report_number=str(report_num),
                manufacturer=manufacturer,
                incident_date=incident_date,
                incident_description=self._truncate(incident_desc, 2000),
                outcome=outcome,
                report_type=str(report_type),
                mdall_licence=str(licence),
            )
        except Exception as exc:
            logger.debug("HC incident parse error for %s: %s", rec.get("reportNumber", "?"), exc)
            return None

    # -- API fetch ----------------------------------------------------------

    async def _fetch_api_page(
        self,
        page: int = 0,
        page_size: int = _DEFAULT_PAGE_SIZE,
        from_date: Optional[str] = None,
        device_name: Optional[str] = None,
        licence: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Fetch one page from the Health Canada MDBI API.

        Returns (records, total_count).
        """
        params: dict[str, Any] = {
            "pageSize": page_size,
            "page": page,
        }
        if from_date:
            params["fromDate"] = from_date
        if device_name:
            params["deviceName"] = device_name
        if licence:
            params["mdallLicenceNumber"] = licence

        try:
            data = await self._get_json(
                _HC_MDBI_API + "/incidents",
                params=params,
                headers=self._api_headers,
            )
        except Exception as exc:
            logger.warning("HC MDBI API page=%d failed: %s", page, exc)
            return [], 0

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

    async def _fetch_api_all_pages(
        self,
        from_date: Optional[str] = None,
        device_name: Optional[str] = None,
        licence: Optional[str] = None,
        max_records: int = 500,
    ) -> list[HCIncidentReport]:
        """Paginate through MDBI API and collect all matching incident reports."""
        all_incidents: list[HCIncidentReport] = []
        page = 0
        fetched = 0

        while page < _MAX_PAGES and fetched < max_records:
            records, total = await self._fetch_api_page(
                page=page,
                page_size=min(_DEFAULT_PAGE_SIZE, max_records - fetched),
                from_date=from_date,
                device_name=device_name,
                licence=licence,
            )
            if not records:
                break

            for rec in records:
                incident = self._parse_api_record(rec)
                if incident:
                    all_incidents.append(incident)

            fetched += len(records)
            page += 1

            if len(records) < _DEFAULT_PAGE_SIZE:
                break
            if total and fetched >= total:
                break

        logger.info("HC MDBI: fetched %d incidents (pages=%d)", len(all_incidents), page)
        return self.deduplicate(all_incidents)

    # -- HTML fallback scraping ---------------------------------------------

    async def _fetch_html_fallback(
        self, days: int
    ) -> list[HCIncidentReport]:
        """
        HTML fallback for when the MDBI API is unavailable.

        Scrapes the public search results page.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        from_date_str = cutoff.strftime("%Y-%m-%d")
        incidents: list[HCIncidentReport] = []

        try:
            params: dict[str, Any] = {
                "fromDate": from_date_str,
                "lang": "en",
            }
            html = await self._get_html(_HC_MDBI_WEB + "/search", params=params)
            soup = BeautifulSoup(html, "html.parser")

            table = soup.find("table")
            if not table:
                logger.warning("HC MDBI HTML fallback: no table found")
                return incidents

            rows = table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue

                date_raw = cells[0].get_text(strip=True)
                report_num = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                device = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                manufacturer = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                outcome_raw = cells[4].get_text(strip=True) if len(cells) > 4 else ""
                desc = cells[5].get_text(strip=True) if len(cells) > 5 else ""

                report_date = self._parse_date(date_raw)
                if report_date and report_date < cutoff:
                    continue

                outcome = self._normalise_outcome(outcome_raw)
                severity = self._outcome_severity(outcome)

                link_tag = cells[1].find("a") if len(cells) > 1 else None
                source_url = ""
                if link_tag and link_tag.get("href"):
                    href = link_tag["href"]
                    source_url = href if href.startswith("http") else f"{_HC_BASE}{href}"

                incidents.append(
                    HCIncidentReport(
                        action_type="incident_report",
                        date=report_date,
                        device_name=self._truncate(device, 300),
                        company=manufacturer,
                        description=self._truncate(desc, 2000),
                        severity=severity,
                        source_url=source_url or _HC_MDBI_WEB,
                        country="CA",
                        raw_data={
                            "date_raw": date_raw,
                            "report_num_raw": report_num,
                            "device_raw": device,
                            "manufacturer_raw": manufacturer,
                            "outcome_raw": outcome_raw,
                            "desc_raw": desc,
                        },
                        report_number=report_num,
                        manufacturer=manufacturer,
                        incident_date=report_date,
                        incident_description=self._truncate(desc, 2000),
                        outcome=outcome,
                        report_type="",
                        mdall_licence="",
                    )
                )
        except Exception as exc:
            logger.warning("HC MDBI HTML fallback failed: %s", exc)

        return incidents

    # -- Public API ---------------------------------------------------------

    async def fetch_recent_incidents(self, days: int = 30) -> list[HCIncidentReport]:
        """
        Fetch Health Canada incident reports submitted within the last *days* days.

        Tries the API first; falls back to HTML scraping if API is unavailable.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        from_date = cutoff.strftime("%Y-%m-%d")

        incidents = await self._fetch_api_all_pages(from_date=from_date)

        if not incidents:
            logger.info("HC MDBI API returned no results — trying HTML fallback")
            incidents = await self._fetch_html_fallback(days=days)

        # Post-filter: ensure all dates are within window
        filtered = [
            i for i in incidents
            if not i.incident_date or i.incident_date >= cutoff
        ]
        return filtered

    async def fetch_by_licence(self, licence_number: str) -> list[HCIncidentReport]:
        """
        Fetch all incident reports associated with a specific MDALL device licence.

        Licence numbers are the Canadian equivalent of a device registration.
        """
        incidents = await self._fetch_api_all_pages(
            licence=licence_number, max_records=200
        )
        # Also filter by licence in parsed results
        lic_lower = licence_number.lower()
        matched = [
            i for i in incidents
            if (
                lic_lower in i.mdall_licence.lower()
                or lic_lower in i.description.lower()
            )
        ]
        return matched if matched else incidents

    # -- BaseEnforcementScraper contract ------------------------------------

    async def fetch_recent(self, days: int = 30) -> list[EnforcementAction]:
        """Satisfy base class contract — delegates to fetch_recent_incidents."""
        return await self.fetch_recent_incidents(days=days)  # type: ignore[return-value]

    async def fetch_by_device(self, device_name: str) -> list[EnforcementAction]:
        """Fetch incidents related to a specific device name."""
        incidents = await self._fetch_api_all_pages(device_name=device_name, max_records=200)
        if not incidents:
            # Fallback: search within recent incidents
            all_recent = await self.fetch_recent_incidents(days=365)
            dev_lower = device_name.lower()
            incidents = [
                i for i in all_recent
                if dev_lower in i.device_name.lower() or dev_lower in i.description.lower()
            ]
        return incidents  # type: ignore[return-value]

    async def search(self, query: str, limit: int = 50) -> list[EnforcementAction]:
        """Free-text search across Health Canada incident reports."""
        incidents = await self._fetch_api_all_pages(device_name=query, max_records=limit)
        if not incidents:
            all_recent = await self.fetch_recent_incidents(days=730)
            q_lower = query.lower()
            incidents = [
                i for i in all_recent
                if (
                    q_lower in i.device_name.lower()
                    or q_lower in i.description.lower()
                    or q_lower in i.manufacturer.lower()
                )
            ]
        return incidents[:limit]  # type: ignore[return-value]

    # -- Health check -------------------------------------------------------

    async def health_check(self) -> bool:
        """Verify Health Canada MDBI API or web interface is reachable."""
        try:
            records, _ = await self._fetch_api_page(page=0, page_size=1)
            return True
        except Exception:
            # Try HTML fallback as secondary health probe
            try:
                html = await self._get_html(_HC_MDBI_WEB)
                return len(html) > 500
            except Exception as exc:
                logger.warning("HC MDBI health_check failed: %s", exc)
                return False


# Singleton
hc_incidents_scraper = HCIncidentsScraper()
