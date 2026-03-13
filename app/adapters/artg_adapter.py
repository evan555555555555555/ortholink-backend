"""
OrthoLink — TGA ARTG / AusUDID Registry Adapter (Australia)

Provides async access to Australia's Therapeutic Goods Administration (TGA)
Australian Register of Therapeutic Goods (ARTG) and AusUDID device
identification database.

ARTG: https://www.tga.gov.au/resources/artg
AusUDID: https://www.tga.gov.au/ausudid

Key regulatory context:
  - All medical devices sold in Australia must be listed on the ARTG
  - AusUDID mandatory deadlines: Class IIb/III by 2026-12-01, Class IIa by 2027-06-01
  - TGA classifications align with EU MDR risk classes (I, IIa, IIb, III)
  - ARTG entries include sponsor details, conditions, and classification

Usage:
    from app.adapters.artg_adapter import artg_adapter
    device = await artg_adapter.fetch_device("123456")
    results = await artg_adapter.search_devices("knee implant")
"""

import logging
import re
from datetime import date, datetime
from enum import Enum
from typing import Optional

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field

from app.adapters.base_adapter import BaseRegistryAdapter, RegistrationRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TGAClassification(str, Enum):
    """TGA/EU-aligned medical device risk classifications."""
    CLASS_I = "I"
    CLASS_IIA = "IIa"
    CLASS_IIB = "IIb"
    CLASS_III = "III"
    AIMD = "AIMD"
    IVD_A = "IVD-A"
    IVD_B = "IVD-B"
    IVD_C = "IVD-C"
    IVD_D = "IVD-D"


class AusUDIDStatus(str, Enum):
    """AusUDID migration status for a device."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLIANT = "compliant"
    OVERDUE = "overdue"


class ARTGDevice(BaseModel):
    """Parsed ARTG entry for an Australian-registered medical device."""
    artg_number: str = Field(..., description="ARTG ID number (6-digit)")
    device_name: str = Field(..., description="Device name / product description")
    sponsor: str = Field(default="", description="Australian sponsor company name")
    classification: Optional[str] = Field(
        default=None,
        description="TGA risk classification (I, IIa, IIb, III, AIMD, IVD-*)",
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="Conditions of inclusion on the ARTG",
    )
    aus_udi_di: Optional[str] = Field(
        default=None,
        description="AusUDID Device Identifier (if assigned)",
    )
    udi_status: AusUDIDStatus = Field(
        default=AusUDIDStatus.NOT_STARTED,
        description="AusUDID migration status",
    )
    udi_deadline: Optional[date] = Field(
        default=None,
        description="AusUDID compliance deadline based on classification",
    )
    gmdn_code: Optional[str] = Field(
        default=None,
        description="GMDN term code (if available in listing)",
    )
    start_date: Optional[date] = Field(
        default=None,
        description="Date the device was first included on the ARTG",
    )
    source_url: str = Field(default="", description="URL of the ARTG listing page")


class ARTGSearchResult(BaseModel):
    """Summary result from an ARTG search query."""
    artg_number: str
    device_name: str
    sponsor: str = ""
    classification: Optional[str] = None
    source_url: str = ""


# ---------------------------------------------------------------------------
# AusUDID transition deadlines (TGA Regulatory Framework)
# ---------------------------------------------------------------------------

_AUSUDID_DEADLINES: dict[str, date] = {
    "III": date(2026, 12, 1),
    "AIMD": date(2026, 12, 1),
    "IIb": date(2026, 12, 1),
    "IIa": date(2027, 6, 1),
    "I": date(2027, 12, 1),
    "IVD-D": date(2026, 12, 1),
    "IVD-C": date(2027, 6, 1),
    "IVD-B": date(2027, 12, 1),
    "IVD-A": date(2027, 12, 1),
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ARTGAdapter(BaseRegistryAdapter):
    """
    Async adapter for TGA ARTG and AusUDID (Australia).

    Scrapes the TGA public ARTG search pages using BeautifulSoup.
    TGA does not provide a structured JSON API, so HTML parsing is required.
    """

    ARTG_BASE = "https://www.tga.gov.au/resources/artg"
    AUSUDID_BASE = "https://www.tga.gov.au/how-we-regulate/manufacturing-and-registration/registration-medical-devices/unique-device-identification-udi"
    ARTG_SEARCH_URL = "https://www.tga.gov.au/resources/artg"

    BASE_URL = "https://www.tga.gov.au"
    REGISTRY_NAME = "TGA-ARTG"
    DEFAULT_TIMEOUT = 20.0

    def __init__(self) -> None:
        super().__init__()

    def get_source_url(self) -> str:
        """Return the canonical TGA ARTG registry URL."""
        return "https://www.tga.gov.au/resources/artg"

    def get_artg_url(self, identifier: str) -> str:
        """Return the direct ARTG listing URL for a given ARTG number."""
        return f"{self.ARTG_BASE}/{identifier}"

    # ------------------------------------------------------------------
    # Core abstract method implementations
    # ------------------------------------------------------------------

    async def fetch_device(self, identifier: str) -> Optional[ARTGDevice]:
        """
        Fetch a single ARTG entry by its ARTG number.

        Args:
            identifier: The ARTG number (e.g. "123456").

        Returns:
            Parsed ARTGDevice or None if not found.
        """
        artg_number = identifier.strip()
        url = self.get_artg_url(artg_number)
        logger.info("ARTG fetch_device: %s", artg_number)

        try:
            html = await self._get_html(url)
            if html is None:
                return None
            return self._parse_device_page(html, artg_number, url)

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "ARTG fetch_device %s HTTP %d: %s",
                artg_number,
                exc.response.status_code,
                exc,
            )
            return None
        except Exception as exc:
            logger.error("ARTG fetch_device %s failed: %s", artg_number, exc)
            return None

    async def _get_html(self, url: str) -> Optional[str]:
        """Fetch URL and return response HTML text; None on 404."""
        response = await self._get(url)
        if response.status_code == 404:
            logger.info("ARTG not found (404): %s", url)
            return None
        return response.text

    async def search_devices(
        self,
        query: str,
        limit: int = 25,
    ) -> list[ARTGSearchResult]:
        """
        Search the ARTG registry by device name, sponsor, or ARTG number.

        Args:
            query: Free-text search term.
            limit: Maximum results to return.

        Returns:
            List of ARTGSearchResult summaries.
        """
        logger.info("ARTG search_devices: query=%r limit=%d", query, limit)

        params = {
            "search_api_fulltext": query,
            "items_per_page": str(min(limit, 50)),
        }

        try:
            response = await self._get(
                self.ARTG_SEARCH_URL, params=params
            )
            return self._parse_search_results(response.text, limit)

        except Exception as exc:
            logger.warning("ARTG search_devices failed: %s", exc)
            return []

    async def fetch_registrations(
        self,
        country: str,
    ) -> list[RegistrationRecord]:
        """
        Fetch ARTG registrations. AU-only; delegates to search_devices.

        Args:
            country: ISO 3166-1 alpha-2. Only 'AU' returns results.

        Returns:
            Empty list for non-AU; RegistrationRecord list for AU.
        """
        if country.upper() != "AU":
            return []
        results = await self.search_devices(query="*", limit=50)
        return [
            RegistrationRecord(
                registry="TGA-ARTG",
                country="AU",
                device_id=r.artg_number,
                device_name=r.device_name,
                manufacturer=r.sponsor,
                status="",
                raw={"classification": r.classification, "source_url": r.source_url},
            )
            for r in results
        ]

    def _health_check_path(self) -> str:
        return "/resources/artg"

    # ------------------------------------------------------------------
    # AusUDID-specific methods
    # ------------------------------------------------------------------

    async def get_udi_status(self, artg_number: str) -> Optional[dict]:
        """
        Check AusUDID registration status for a device by ARTG number.

        Returns dict with udi_di, status, deadline, and days_remaining.
        """
        device = await self.fetch_device(artg_number)
        if device is None:
            return None

        deadline = device.udi_deadline
        days_remaining: Optional[int] = None
        if deadline:
            days_remaining = (deadline - date.today()).days

        return {
            "artg_number": device.artg_number,
            "device_name": device.device_name,
            "classification": device.classification,
            "aus_udi_di": device.aus_udi_di,
            "udi_status": device.udi_status.value,
            "udi_deadline": deadline.isoformat() if deadline else None,
            "days_remaining": days_remaining,
            "is_overdue": days_remaining is not None and days_remaining < 0,
        }

    def compute_udi_deadline(self, classification: Optional[str]) -> Optional[date]:
        """
        Determine the AusUDID compliance deadline based on device classification.
        """
        if not classification:
            return None
        return _AUSUDID_DEADLINES.get(classification)

    def compute_udi_status(
        self,
        classification: Optional[str],
        has_udi: bool,
    ) -> AusUDIDStatus:
        """
        Derive AusUDID migration status from classification and current UDI state.
        """
        if has_udi:
            return AusUDIDStatus.COMPLIANT

        deadline = self.compute_udi_deadline(classification)
        if deadline is None:
            return AusUDIDStatus.NOT_STARTED

        if date.today() > deadline:
            return AusUDIDStatus.OVERDUE

        return AusUDIDStatus.NOT_STARTED

    # ------------------------------------------------------------------
    # HTML parsing helpers
    # ------------------------------------------------------------------

    def _parse_device_page(
        self,
        html: str,
        artg_number: str,
        source_url: str,
    ) -> Optional[ARTGDevice]:
        """Parse a single ARTG device detail page."""
        soup = BeautifulSoup(html, "html.parser")

        device_name = self._extract_text(soup, "h1") or ""
        if not device_name:
            title_tag = soup.find("title")
            device_name = title_tag.get_text(strip=True) if title_tag else ""

        # Extract fields from the structured detail table/definition list
        fields = self._extract_detail_fields(soup)

        sponsor = fields.get("sponsor", fields.get("sponsor name", ""))
        classification_raw = fields.get("classification", fields.get("device class", ""))
        classification = self._normalize_classification(classification_raw)
        gmdn_code = fields.get("gmdn", fields.get("gmdn code", None))

        # Parse conditions
        conditions: list[str] = []
        conditions_section = soup.find(
            string=re.compile(r"conditions?\s+of\s+inclusion", re.IGNORECASE)
        )
        if conditions_section:
            parent = conditions_section.find_parent()
            if parent:
                next_list = parent.find_next("ul")
                if next_list:
                    conditions = [
                        li.get_text(strip=True) for li in next_list.find_all("li")
                    ]

        # Parse start date
        start_date = self._parse_date(fields.get("start date", ""))

        # AusUDID status
        aus_udi_di = fields.get("udi-di", fields.get("ausudid", None))
        has_udi = bool(aus_udi_di)
        udi_deadline = self.compute_udi_deadline(classification)
        udi_status = self.compute_udi_status(classification, has_udi)

        return ARTGDevice(
            artg_number=artg_number,
            device_name=device_name,
            sponsor=sponsor,
            classification=classification,
            conditions=conditions,
            aus_udi_di=aus_udi_di,
            udi_status=udi_status,
            udi_deadline=udi_deadline,
            gmdn_code=gmdn_code,
            start_date=start_date,
            source_url=source_url,
        )

    def _parse_search_results(
        self,
        html: str,
        limit: int,
    ) -> list[ARTGSearchResult]:
        """Parse ARTG search results page into structured summaries."""
        soup = BeautifulSoup(html, "html.parser")
        results: list[ARTGSearchResult] = []

        # TGA search results are rendered as a list of result items
        # Try multiple selectors used by TGA's frontend
        result_items = (
            soup.select(".view-content .views-row")
            or soup.select(".search-results li")
            or soup.select("table.views-table tbody tr")
            or soup.select("article")
        )

        for item in result_items[:limit]:
            artg_number = ""
            device_name = ""
            sponsor = ""
            classification = None
            link_url = ""

            if isinstance(item, Tag):
                # Try extracting from table row
                cells = item.find_all("td")
                if len(cells) >= 3:
                    artg_number = cells[0].get_text(strip=True)
                    device_name = cells[1].get_text(strip=True)
                    sponsor = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    classification_text = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                    classification = self._normalize_classification(classification_text)
                else:
                    # Fallback: link-based parsing
                    link = item.find("a")
                    if link:
                        device_name = link.get_text(strip=True)
                        href = link.get("href", "")
                        if href:
                            link_url = href if href.startswith("http") else f"https://www.tga.gov.au{href}"
                            # Extract ARTG number from URL path
                            match = re.search(r"/artg/(\d+)", href)
                            if match:
                                artg_number = match.group(1)

                    # Try extracting sponsor from a secondary element
                    sponsor_el = item.find(class_=re.compile(r"sponsor|company", re.IGNORECASE))
                    if sponsor_el:
                        sponsor = sponsor_el.get_text(strip=True)

            if artg_number or device_name:
                results.append(ARTGSearchResult(
                    artg_number=artg_number,
                    device_name=device_name,
                    sponsor=sponsor,
                    classification=classification,
                    source_url=link_url or self.get_artg_url(artg_number),
                ))

        logger.info("ARTG search: parsed %d results", len(results))
        return results

    def _extract_detail_fields(self, soup: BeautifulSoup) -> dict[str, str]:
        """
        Extract key-value pairs from TGA detail page.
        TGA uses <dl>/<dt>/<dd> or <table> layouts for device details.
        """
        fields: dict[str, str] = {}

        # Try definition lists first
        for dt in soup.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            if dd:
                key = dt.get_text(strip=True).lower().rstrip(":")
                value = dd.get_text(strip=True)
                fields[key] = value

        # Fallback: two-column table
        if not fields:
            for row in soup.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) == 2:
                    key = cells[0].get_text(strip=True).lower().rstrip(":")
                    value = cells[1].get_text(strip=True)
                    fields[key] = value

        return fields

    @staticmethod
    def _extract_text(soup: BeautifulSoup, tag: str) -> Optional[str]:
        """Extract text from the first occurrence of a tag."""
        el = soup.find(tag)
        return el.get_text(strip=True) if el else None

    @staticmethod
    def _normalize_classification(raw: str) -> Optional[str]:
        """Normalize a classification string to canonical form."""
        if not raw:
            return None
        cleaned = raw.strip().upper()
        mapping = {
            "CLASS I": "I",
            "CLASS IIA": "IIa",
            "CLASS IIB": "IIb",
            "CLASS III": "III",
            "I": "I",
            "IIA": "IIa",
            "IIB": "IIb",
            "III": "III",
            "AIMD": "AIMD",
        }
        for key, val in mapping.items():
            if cleaned == key or cleaned == f"CLASS {key}":
                return val
        # Check for IVD classes
        ivd_match = re.match(r"IVD[\s-]?([A-D])", cleaned)
        if ivd_match:
            return f"IVD-{ivd_match.group(1)}"
        return raw.strip() or None

    @staticmethod
    def _parse_date(raw: str) -> Optional[date]:
        """Attempt to parse a date string from TGA pages."""
        if not raw:
            return None
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(raw.strip(), fmt).date()
            except ValueError:
                continue
        return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

artg_adapter = ARTGAdapter()
