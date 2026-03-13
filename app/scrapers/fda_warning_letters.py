"""
OrthoLink -- FDA Warning Letters Scraper

Two data paths:
  1. openFDA drug/enforcement API  (structured JSON -- reliable)
  2. FDA warning-letters HTML page  (BeautifulSoup parse -- supplement)

Orthopedic-specific focus:
  - Filters for design controls, CAPA, process validation, 21 CFR 820.xxx
  - Aggregates most common citation patterns across time window

Usage:
    from app.scrapers.fda_warning_letters import fda_wl_scraper
    letters = await fda_wl_scraper.fetch_orthopedic_warnings(days=90)
    patterns = await fda_wl_scraper.analyze_citation_patterns()
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from app.scrapers.base_scraper import (
    BaseEnforcementScraper,
    EnforcementAction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPENFDA_ENFORCEMENT = "https://api.fda.gov/drug/enforcement.json"
_OPENFDA_DEVICE_ENFORCEMENT = "https://api.fda.gov/device/enforcement.json"

_FDA_WL_PAGE = (
    "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations"
    "/compliance-actions-and-activities/warning-letters"
)

# 21 CFR 820 design control citations relevant to orthopedic devices
ORTHO_KEYWORDS: list[str] = [
    "design validation",
    "design controls",
    "design verification",
    "design review",
    "design history file",
    "design input",
    "design output",
    "design transfer",
    "CAPA",
    "corrective and preventive action",
    "corrective action",
    "preventive action",
    "process validation",
    "production and process controls",
    "complaint handling",
    "medical device reporting",
    "purchasing controls",
    "quality system regulation",
    "quality management system",
    "sterilization",
    "biocompatibility",
    "implant",
    "orthopedic",
    "orthopaedic",
    "spine",
    "spinal",
    "hip",
    "knee",
    "joint",
    "bone",
    "trauma",
    "fixation",
    "prosthesis",
    "prosthetic",
]

# Regex to extract 21 CFR 820.xxx citations
_CFR_PATTERN = re.compile(r"21\s*CFR\s*(\d{3}(?:\.\d+)?)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class WarningLetter(BaseModel):
    """Single FDA Warning Letter record."""

    posting_date: Optional[datetime] = None
    company_name: str = ""
    subject: str = ""
    issuing_office: str = ""
    response_letter_url: str = ""
    source_url: str = ""
    cited_cfr_sections: list[str] = Field(default_factory=list)
    is_orthopedic_relevant: bool = False
    raw_data: dict[str, Any] = Field(default_factory=dict)


class CitationPattern(BaseModel):
    """Aggregated citation frequency."""

    cfr_section: str
    count: int
    percentage: float = Field(description="Percentage of letters citing this section")


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class FDAWarningLettersScraper(BaseEnforcementScraper):
    """
    Scrapes FDA Warning Letters from two sources:
      1. openFDA enforcement API (device + drug)
      2. FDA warning-letters HTML listing page
    """

    def __init__(self, api_key: str = "") -> None:
        super().__init__()
        self._api_key = api_key
        self._letters_cache: list[WarningLetter] = []

    def get_source_name(self) -> str:
        return "FDA Warning Letters"

    # -- openFDA helpers ----------------------------------------------------

    def _api_params(self, extra: dict[str, Any]) -> dict[str, Any]:
        params = dict(extra)
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    async def _fetch_enforcement_api(
        self,
        days: int,
        search_extra: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query openFDA device enforcement endpoint."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y%m%d")
        search = f"report_date:[{cutoff}+TO+99991231]"
        if search_extra:
            search = f"{search}+AND+{search_extra}"

        params = self._api_params({"search": search, "limit": min(limit, 100)})
        try:
            data = await self._get_json(_OPENFDA_DEVICE_ENFORCEMENT, params=params)
            return data.get("results", [])
        except Exception as exc:
            logger.warning("openFDA device enforcement fetch failed: %s", exc)
            return []

    # -- HTML scraping ------------------------------------------------------

    async def _fetch_wl_html(self) -> list[WarningLetter]:
        """
        Scrape the FDA warning-letters listing page.
        Extracts posting date, company name, subject, issuing office, and link.
        """
        letters: list[WarningLetter] = []
        try:
            html = await self._get_html(_FDA_WL_PAGE)
            soup = BeautifulSoup(html, "html.parser")

            # FDA renders WLs in a table or a list depending on redesigns.
            # Try table first, then structured div fallback.
            table = soup.find("table")
            if table:
                rows = table.find_all("tr")[1:]  # skip header
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) < 3:
                        continue
                    link_tag = cells[0].find("a")
                    letter_url = ""
                    if link_tag and link_tag.get("href"):
                        href = link_tag["href"]
                        if href.startswith("/"):
                            href = f"https://www.fda.gov{href}"
                        letter_url = href

                    date_text = cells[0].get_text(strip=True) if cells else ""
                    company = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                    subject = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    issuing = cells[3].get_text(strip=True) if len(cells) > 3 else ""

                    posting_date = self._parse_date(date_text)
                    cfr_sections = _CFR_PATTERN.findall(subject)
                    is_ortho = any(
                        kw.lower() in subject.lower() or kw.lower() in company.lower()
                        for kw in ORTHO_KEYWORDS
                    )

                    letters.append(
                        WarningLetter(
                            posting_date=posting_date,
                            company_name=company,
                            subject=subject,
                            issuing_office=issuing,
                            response_letter_url=letter_url,
                            source_url=letter_url or _FDA_WL_PAGE,
                            cited_cfr_sections=[f"21 CFR {s}" for s in cfr_sections],
                            is_orthopedic_relevant=is_ortho,
                        )
                    )
            else:
                # Fallback: look for a view-content region with article links
                articles = soup.select(".view-content .views-row, .view-content li")
                for article in articles:
                    link_tag = article.find("a")
                    if not link_tag:
                        continue
                    href = link_tag.get("href", "")
                    if href.startswith("/"):
                        href = f"https://www.fda.gov{href}"
                    title = link_tag.get_text(strip=True)
                    date_span = article.find("span", class_="date-display-single")
                    date_text = date_span.get_text(strip=True) if date_span else ""

                    cfr_sections = _CFR_PATTERN.findall(title)
                    is_ortho = any(kw.lower() in title.lower() for kw in ORTHO_KEYWORDS)

                    letters.append(
                        WarningLetter(
                            posting_date=self._parse_date(date_text),
                            company_name="",
                            subject=title,
                            source_url=href,
                            cited_cfr_sections=[f"21 CFR {s}" for s in cfr_sections],
                            is_orthopedic_relevant=is_ortho,
                        )
                    )

            logger.info("FDA WL HTML: parsed %d warning letters", len(letters))
        except Exception as exc:
            logger.warning("FDA WL HTML scrape failed: %s", exc)

        return letters

    # -- Public methods matching scraper spec --------------------------------

    async def fetch_recent(self, days: int = 30) -> list[EnforcementAction]:
        """
        Fetch recent FDA enforcement actions from openFDA API.

        Returns unified EnforcementAction list.
        """
        raw = await self._fetch_enforcement_api(days=days)
        actions: list[EnforcementAction] = []
        for rec in raw:
            date_str = rec.get("report_date") or rec.get("recall_initiation_date", "")
            actions.append(
                EnforcementAction(
                    action_type="warning_letter" if "warning" in rec.get("classification", "").lower() else "recall",
                    date=self._parse_date(date_str),
                    device_name=self._truncate(rec.get("product_description", ""), 200),
                    company=rec.get("recalling_firm", ""),
                    description=self._truncate(rec.get("reason_for_recall", ""), 2000),
                    severity=self.classify_severity(
                        f"class_{rec.get('classification', 'II').lower()}_recall"
                    ),
                    source_url=rec.get("openfda", {}).get("application_number", ""),
                    country="US",
                    raw_data=rec,
                )
            )
        return self.deduplicate(actions)

    async def fetch_by_device(self, device_name: str) -> list[EnforcementAction]:
        """Fetch enforcement actions for a specific device name."""
        raw = await self._fetch_enforcement_api(
            days=365,
            search_extra=f'product_description:"{device_name}"',
        )
        actions: list[EnforcementAction] = []
        for rec in raw:
            date_str = rec.get("report_date") or rec.get("recall_initiation_date", "")
            actions.append(
                EnforcementAction(
                    action_type="enforcement",
                    date=self._parse_date(date_str),
                    device_name=self._truncate(rec.get("product_description", ""), 200),
                    company=rec.get("recalling_firm", ""),
                    description=self._truncate(rec.get("reason_for_recall", ""), 2000),
                    severity=self.classify_severity(
                        f"class_{rec.get('classification', 'II').lower()}_recall"
                    ),
                    source_url="",
                    country="US",
                    raw_data=rec,
                )
            )
        return self.deduplicate(actions)

    async def search(self, query: str, limit: int = 50) -> list[EnforcementAction]:
        """Free-text search via openFDA enforcement API."""
        raw = await self._fetch_enforcement_api(
            days=730,
            search_extra=f'reason_for_recall:"{query}"',
            limit=limit,
        )
        actions: list[EnforcementAction] = []
        for rec in raw:
            date_str = rec.get("report_date") or rec.get("recall_initiation_date", "")
            actions.append(
                EnforcementAction(
                    action_type="enforcement",
                    date=self._parse_date(date_str),
                    device_name=self._truncate(rec.get("product_description", ""), 200),
                    company=rec.get("recalling_firm", ""),
                    description=self._truncate(rec.get("reason_for_recall", ""), 2000),
                    severity=self.classify_severity(
                        f"class_{rec.get('classification', 'II').lower()}_recall"
                    ),
                    source_url="",
                    country="US",
                    raw_data=rec,
                )
            )
        return self.deduplicate(actions)

    # -- Orthopedic-specific methods ----------------------------------------

    async def fetch_orthopedic_warnings(self, days: int = 90) -> list[WarningLetter]:
        """
        Fetch Warning Letters and filter to orthopedic-relevant citations.

        Combines openFDA enforcement results with HTML-scraped warning letters.
        Filters for citations to 21 CFR 820.xxx and orthopedic keywords.
        """
        # 1. HTML scrape for warning letter metadata
        html_letters = await self._fetch_wl_html()

        # 2. openFDA enforcement for additional structured data
        api_records = await self._fetch_enforcement_api(days=days)
        for rec in api_records:
            reason = rec.get("reason_for_recall", "")
            product = rec.get("product_description", "")
            combined_text = f"{reason} {product}"
            cfr_sections = _CFR_PATTERN.findall(combined_text)
            is_ortho = any(kw.lower() in combined_text.lower() for kw in ORTHO_KEYWORDS)

            if is_ortho or cfr_sections:
                html_letters.append(
                    WarningLetter(
                        posting_date=self._parse_date(
                            rec.get("report_date") or rec.get("recall_initiation_date", "")
                        ),
                        company_name=rec.get("recalling_firm", ""),
                        subject=self._truncate(reason, 300),
                        cited_cfr_sections=[f"21 CFR {s}" for s in cfr_sections],
                        is_orthopedic_relevant=is_ortho,
                        raw_data=rec,
                    )
                )

        # Filter: keep only letters posted within the requested window
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        filtered: list[WarningLetter] = []
        seen_subjects: set[str] = set()
        for letter in html_letters:
            if letter.posting_date and letter.posting_date < cutoff:
                continue
            dedup = f"{letter.company_name}|{letter.subject[:60]}"
            if dedup in seen_subjects:
                continue
            seen_subjects.add(dedup)

            # Re-check orthopedic relevance across all text
            full_text = f"{letter.subject} {letter.company_name} {' '.join(letter.cited_cfr_sections)}"
            if any(kw.lower() in full_text.lower() for kw in ORTHO_KEYWORDS):
                letter.is_orthopedic_relevant = True

            filtered.append(letter)

        self._letters_cache = filtered
        logger.info(
            "FDA WL orthopedic filter: %d / %d letters relevant",
            sum(1 for l in filtered if l.is_orthopedic_relevant),
            len(filtered),
        )
        return filtered

    async def analyze_citation_patterns(self) -> dict[str, Any]:
        """
        Aggregate most common CFR citations from cached warning letters.

        Returns:
            {
                "total_letters": int,
                "orthopedic_relevant": int,
                "top_citations": [CitationPattern, ...],
                "analyzed_at": str (ISO),
            }
        """
        if not self._letters_cache:
            await self.fetch_orthopedic_warnings(days=180)

        counter: Counter[str] = Counter()
        for letter in self._letters_cache:
            for section in letter.cited_cfr_sections:
                counter[section] += 1

        total = len(self._letters_cache)
        ortho_count = sum(1 for l in self._letters_cache if l.is_orthopedic_relevant)

        top_citations = [
            CitationPattern(
                cfr_section=section,
                count=count,
                percentage=round((count / total) * 100, 1) if total else 0.0,
            )
            for section, count in counter.most_common(20)
        ]

        return {
            "total_letters": total,
            "orthopedic_relevant": ortho_count,
            "top_citations": [c.model_dump() for c in top_citations],
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

    # -- Health check -------------------------------------------------------

    async def health_check(self) -> bool:
        """Verify openFDA device enforcement endpoint is reachable."""
        try:
            params = self._api_params({"search": "classification:I", "limit": 1})
            await self._get_json(_OPENFDA_DEVICE_ENFORCEMENT, params=params)
            return True
        except Exception as exc:
            logger.warning("FDA WL health_check failed: %s", exc)
            return False


# Singleton
fda_wl_scraper = FDAWarningLettersScraper()
