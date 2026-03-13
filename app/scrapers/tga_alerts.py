"""
OrthoLink -- TGA Safety Alerts Scraper (Australia)

Two data paths:
  1. TGA Safety Alerts listing page (BeautifulSoup HTML parse)
     https://www.tga.gov.au/safety/alerts-medicine-and-medical-devices
  2. TGA Recalls structured data page (BeautifulSoup HTML parse)
     https://www.tga.gov.au/recall-action/medical-devices

Parses: alert_date, device_name, sponsor, artg_number, hazard_description,
        action_taken, alert_type, severity.

Usage:
    from app.scrapers.tga_alerts import tga_alerts_scraper
    alerts = await tga_alerts_scraper.fetch_recent_alerts(days=30)
    recalls = await tga_alerts_scraper.fetch_recalls_only(days=30)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

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

_TGA_ALERTS_URL = (
    "https://www.tga.gov.au/safety/alerts-medicine-and-medical-devices"
)
_TGA_RECALLS_URL = "https://www.tga.gov.au/recall-action/medical-devices"
_TGA_BASE = "https://www.tga.gov.au"

# Alert type keyword mappings
_ALERT_TYPE_MAP: dict[str, str] = {
    "recall": "recall",
    "hazard": "recall",
    "safety alert": "safety_alert",
    "alert": "safety_alert",
    "advisory": "advisory",
    "counterfeit": "counterfeit",
    "not genuine": "counterfeit",
    "information": "advisory",
}

# Action keywords used for severity classification
_ACTION_SEVERITY_MAP: dict[str, Severity] = {
    "urgent": Severity.CRITICAL,
    "hazard alert": Severity.CRITICAL,
    "class i": Severity.CRITICAL,
    "class 1": Severity.CRITICAL,
    "mandatory": Severity.CRITICAL,
    "counterfeit": Severity.CRITICAL,
    "suspension": Severity.CRITICAL,
    "class ii": Severity.HIGH,
    "class 2": Severity.HIGH,
    "safety alert": Severity.HIGH,
    "recall": Severity.HIGH,
    "class iii": Severity.MEDIUM,
    "class 3": Severity.MEDIUM,
    "advisory": Severity.MEDIUM,
    "monitor": Severity.LOW,
    "information": Severity.LOW,
}

# ARTG number pattern
_ARTG_RE = re.compile(r"ARTG\s*[#:]?\s*(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TGASafetyAlert(EnforcementAction):
    """TGA Safety Alert or Recall — extends unified EnforcementAction."""

    artg_number: str = Field(default="", description="Australian Register of Therapeutic Goods entry number")
    sponsor: str = Field(default="", description="TGA registered sponsor (AU distributor/manufacturer rep)")
    hazard_description: str = Field(default="", description="Specific hazard identified by TGA")
    action_taken: str = Field(default="", description="Regulatory action taken (e.g. voluntary recall, safety alert issued)")
    alert_type: str = Field(
        default="safety_alert",
        description="One of: recall | safety_alert | counterfeit | advisory",
    )


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


class TGAAlertsScraper(BaseEnforcementScraper):
    """
    Scrapes TGA (Therapeutic Goods Administration) Safety Alerts and Recalls.

    Data comes from two HTML pages:
      - Alerts listing: /safety/alerts-medicine-and-medical-devices
      - Recalls listing: /recall-action/medical-devices

    Both are standard HTML tables/lists parsed with BeautifulSoup4.
    """

    def __init__(self) -> None:
        super().__init__()
        self._extra_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-AU,en;q=0.8",
            "Referer": _TGA_BASE,
        }

    def get_source_name(self) -> str:
        return "TGA Safety Alerts (AU)"

    # -- Severity classification for TGA-specific actions -------------------

    @staticmethod
    def _classify_tga_severity(action_text: str, alert_type: str) -> Severity:
        """Map TGA-specific action text to Severity."""
        combined = f"{action_text} {alert_type}".lower()
        for keyword, sev in _ACTION_SEVERITY_MAP.items():
            if keyword in combined:
                return sev
        return Severity.MEDIUM

    @staticmethod
    def _classify_alert_type(raw_text: str) -> str:
        """Classify raw heading/category text into canonical alert_type."""
        lowered = raw_text.lower()
        for keyword, atype in _ALERT_TYPE_MAP.items():
            if keyword in lowered:
                return atype
        return "safety_alert"

    # -- HTML parsing helpers -----------------------------------------------

    def _parse_alerts_page(
        self, html: str, cutoff: datetime
    ) -> list[TGASafetyAlert]:
        """
        Parse the TGA safety alerts listing HTML.

        The TGA alerts page renders a table or a list of div cards.
        Tries table first; falls back to div-based listing.
        """
        soup = BeautifulSoup(html, "html.parser")
        alerts: list[TGASafetyAlert] = []

        # Strategy 1: standard table
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")[1:]  # skip header
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue

                # Column order on TGA alerts page: Date | Product | Sponsor | Summary | Action
                date_cell = cells[0].get_text(strip=True)
                product_cell = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                sponsor_cell = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                summary_cell = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                action_cell = cells[4].get_text(strip=True) if len(cells) > 4 else ""

                alert_date = self._parse_date(date_cell)
                if alert_date and alert_date < cutoff:
                    continue

                link_tag = cells[1].find("a") if len(cells) > 1 else None
                source_url = ""
                if link_tag and link_tag.get("href"):
                    href = link_tag["href"]
                    source_url = href if href.startswith("http") else f"{_TGA_BASE}{href}"

                artg_match = _ARTG_RE.search(f"{product_cell} {summary_cell}")
                artg = artg_match.group(1) if artg_match else ""

                alert_type = self._classify_alert_type(f"{summary_cell} {action_cell}")
                severity = self._classify_tga_severity(action_cell, alert_type)

                alerts.append(
                    TGASafetyAlert(
                        action_type=alert_type,
                        date=alert_date,
                        device_name=self._truncate(product_cell, 300),
                        company=sponsor_cell,
                        description=self._truncate(f"{summary_cell}. {action_cell}".strip("."), 2000),
                        severity=severity,
                        source_url=source_url or _TGA_ALERTS_URL,
                        country="AU",
                        artg_number=artg,
                        sponsor=sponsor_cell,
                        hazard_description=self._truncate(summary_cell, 1000),
                        action_taken=self._truncate(action_cell, 500),
                        alert_type=alert_type,
                        raw_data={
                            "date_raw": date_cell,
                            "product_raw": product_cell,
                            "sponsor_raw": sponsor_cell,
                            "summary_raw": summary_cell,
                            "action_raw": action_cell,
                        },
                    )
                )
        else:
            # Strategy 2: div-card layout (TGA uses this for newer pages)
            articles = soup.select(
                "article, .views-row, .node--type-alert, "
                "[class*='alert-item'], [class*='recall-item'], .field-content"
            )
            for article in articles:
                title_el = article.find(["h2", "h3", "h4", ".field-title", "a"])
                title = title_el.get_text(strip=True) if title_el else ""

                date_el = article.find(
                    ["time", ".date-display-single", "[class*='date']", "span"]
                )
                date_raw = ""
                if date_el:
                    date_raw = date_el.get("datetime", date_el.get_text(strip=True))

                alert_date = self._parse_date(date_raw)
                if alert_date and alert_date < cutoff:
                    continue

                desc_el = article.find([".field-body", "p", "summary", "div"])
                desc = desc_el.get_text(strip=True) if desc_el else ""

                link_tag = article.find("a")
                source_url = ""
                if link_tag and link_tag.get("href"):
                    href = link_tag["href"]
                    source_url = href if href.startswith("http") else f"{_TGA_BASE}{href}"

                artg_match = _ARTG_RE.search(f"{title} {desc}")
                artg = artg_match.group(1) if artg_match else ""

                alert_type = self._classify_alert_type(title)
                severity = self._classify_tga_severity(title, alert_type)

                alerts.append(
                    TGASafetyAlert(
                        action_type=alert_type,
                        date=alert_date,
                        device_name=self._truncate(title, 300),
                        company="",
                        description=self._truncate(desc, 2000),
                        severity=severity,
                        source_url=source_url or _TGA_ALERTS_URL,
                        country="AU",
                        artg_number=artg,
                        sponsor="",
                        hazard_description=self._truncate(desc, 1000),
                        action_taken="",
                        alert_type=alert_type,
                        raw_data={"title_raw": title, "date_raw": date_raw, "desc_raw": desc},
                    )
                )

        logger.info("TGA alerts page: parsed %d alerts (cutoff=%s)", len(alerts), cutoff.date())
        return alerts

    def _parse_recalls_page(
        self, html: str, cutoff: datetime
    ) -> list[TGASafetyAlert]:
        """
        Parse the TGA medical devices recall listing HTML.

        The recalls page typically renders a filterable table with:
        Date | Product | Sponsor | ARTG | Recall Class | Hazard | Action
        """
        soup = BeautifulSoup(html, "html.parser")
        recalls: list[TGASafetyAlert] = []

        table = soup.find("table")
        if not table:
            logger.warning("TGA recalls page: no table found in HTML")
            return recalls

        rows = table.find_all("tr")[1:]
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            date_raw = cells[0].get_text(strip=True)
            product = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            sponsor = cells[2].get_text(strip=True) if len(cells) > 2 else ""
            artg_raw = cells[3].get_text(strip=True) if len(cells) > 3 else ""
            recall_class = cells[4].get_text(strip=True) if len(cells) > 4 else ""
            hazard = cells[5].get_text(strip=True) if len(cells) > 5 else ""
            action = cells[6].get_text(strip=True) if len(cells) > 6 else ""

            recall_date = self._parse_date(date_raw)
            if recall_date and recall_date < cutoff:
                continue

            # Extract bare ARTG number
            artg_match = _ARTG_RE.search(artg_raw) or _ARTG_RE.search(product)
            artg = artg_match.group(1) if artg_match else artg_raw.strip()

            link_tag = cells[1].find("a") if len(cells) > 1 else None
            source_url = ""
            if link_tag and link_tag.get("href"):
                href = link_tag["href"]
                source_url = href if href.startswith("http") else f"{_TGA_BASE}{href}"

            # Map recall class to severity
            class_str = recall_class.lower()
            if "i" in class_str and "ii" not in class_str:
                severity = Severity.CRITICAL
                action_type = "recall"
            elif "ii" in class_str and "iii" not in class_str:
                severity = Severity.HIGH
                action_type = "recall"
            elif "iii" in class_str:
                severity = Severity.MEDIUM
                action_type = "recall"
            else:
                severity = self._classify_tga_severity(f"{recall_class} {action}", "recall")
                action_type = "recall"

            description = self._truncate(
                f"Recall Class {recall_class}: {hazard}. Action: {action}".strip(), 2000
            )

            recalls.append(
                TGASafetyAlert(
                    action_type=action_type,
                    date=recall_date,
                    device_name=self._truncate(product, 300),
                    company=sponsor,
                    description=description,
                    severity=severity,
                    source_url=source_url or _TGA_RECALLS_URL,
                    country="AU",
                    artg_number=artg,
                    sponsor=sponsor,
                    hazard_description=self._truncate(hazard, 1000),
                    action_taken=self._truncate(action, 500),
                    alert_type="recall",
                    raw_data={
                        "date_raw": date_raw,
                        "product_raw": product,
                        "sponsor_raw": sponsor,
                        "artg_raw": artg_raw,
                        "recall_class": recall_class,
                        "hazard_raw": hazard,
                        "action_raw": action,
                    },
                )
            )

        logger.info("TGA recalls page: parsed %d recalls (cutoff=%s)", len(recalls), cutoff.date())
        return recalls

    # -- Pagination ---------------------------------------------------------

    async def _paginate_alerts(self, days: int) -> list[TGASafetyAlert]:
        """
        Paginate through TGA alerts listing.

        TGA uses Drupal-style ?page=N pagination.
        Stops when: no new items found, or all items are older than cutoff.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        all_alerts: list[TGASafetyAlert] = []
        page = 0
        max_pages = 20  # safety ceiling

        while page < max_pages:
            url = _TGA_ALERTS_URL
            params: dict[str, Any] = {}
            if page > 0:
                params["page"] = page

            try:
                html = await self._get_html(url, params=params if params else None)
            except Exception as exc:
                logger.warning("TGA alerts page=%d fetch failed: %s", page, exc)
                break

            batch = self._parse_alerts_page(html, cutoff)
            if not batch:
                break  # No results on this page — done

            all_alerts.extend(batch)

            # If every item in the batch is older than cutoff, stop paginating
            if all(a.date and a.date < cutoff for a in batch if a.date):
                break

            page += 1

        return self.deduplicate(all_alerts)

    # -- Public API ---------------------------------------------------------

    async def fetch_recent_alerts(self, days: int = 30) -> list[TGASafetyAlert]:
        """
        Fetch TGA safety alerts published within the last *days* days.

        Combines both alerts and recalls pages, deduplicates by (url, date, device).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Fetch from both pages concurrently (sequential to stay polite)
        try:
            alerts_html = await self._get_html(_TGA_ALERTS_URL)
            alerts = self._parse_alerts_page(alerts_html, cutoff)
        except Exception as exc:
            logger.warning("TGA alerts page fetch failed: %s", exc)
            alerts = []

        try:
            recalls_html = await self._get_html(_TGA_RECALLS_URL)
            recalls = self._parse_recalls_page(recalls_html, cutoff)
        except Exception as exc:
            logger.warning("TGA recalls page fetch failed: %s", exc)
            recalls = []

        combined = alerts + recalls
        return self.deduplicate(combined)

    async def fetch_recalls_only(self, days: int = 30) -> list[TGASafetyAlert]:
        """
        Fetch only TGA medical device recalls within the last *days* days.
        Uses the structured recalls listing page.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        try:
            html = await self._get_html(_TGA_RECALLS_URL)
            recalls = self._parse_recalls_page(html, cutoff)
            return self.deduplicate(recalls)
        except Exception as exc:
            logger.warning("TGA recalls fetch failed: %s", exc)
            return []

    # -- BaseEnforcementScraper contract ------------------------------------

    async def fetch_recent(self, days: int = 30) -> list[EnforcementAction]:
        """Satisfy base class contract — delegates to fetch_recent_alerts."""
        return await self.fetch_recent_alerts(days=days)  # type: ignore[return-value]

    async def fetch_by_device(self, device_name: str) -> list[EnforcementAction]:
        """
        Fetch TGA actions related to a specific device name.
        Searches within recently fetched alerts (last 365 days).
        """
        all_recent = await self.fetch_recent_alerts(days=365)
        lowered = device_name.lower()
        matched = [
            a for a in all_recent
            if lowered in a.device_name.lower() or lowered in a.description.lower()
        ]
        return matched  # type: ignore[return-value]

    async def search(self, query: str, limit: int = 50) -> list[EnforcementAction]:
        """
        Free-text search across recent TGA alerts (last 365 days).

        TGA does not expose a search API — we filter the scraped dataset.
        """
        all_recent = await self.fetch_recent_alerts(days=365)
        lowered = query.lower()
        matched = [
            a for a in all_recent
            if (
                lowered in a.device_name.lower()
                or lowered in a.description.lower()
                or lowered in a.hazard_description.lower()
                or lowered in a.sponsor.lower()
            )
        ]
        return matched[:limit]  # type: ignore[return-value]

    # -- Health check -------------------------------------------------------

    async def health_check(self) -> bool:
        """Verify TGA alerts page is reachable and returns parseable HTML."""
        try:
            html = await self._get_html(_TGA_ALERTS_URL)
            soup = BeautifulSoup(html, "html.parser")
            # Minimal signal: page has at least one anchor and some text
            has_content = bool(soup.find("a")) and len(soup.get_text(strip=True)) > 200
            if not has_content:
                logger.warning("TGA health_check: page returned but content looks empty")
            return has_content
        except Exception as exc:
            logger.warning("TGA health_check failed: %s", exc)
            return False


# Singleton
tga_alerts_scraper = TGAAlertsScraper()
