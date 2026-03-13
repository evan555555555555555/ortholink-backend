"""
OrthoLink — Live EMA (European Medicines Agency) Client

Provides real-time access to EMA device/medicine safety data:
  - Direct Healthcare Professional Communications (DHPCs) — EU safety alerts
  - Product shortages
  - EPAR document search (European Public Assessment Reports)

Used by RAA agent to supplement FAISS-based EU change monitoring with live EMA data.

Note: EMA API is publicly accessible. No auth required.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

_EMA_BASE = "https://www.ema.europa.eu/en/search/search"


class EMAClient:
    """Async client for EMA public data APIs relevant to medical devices."""

    async def get_dhpcs(self, limit: int = 20) -> list[dict]:
        """
        Fetch recent Direct Healthcare Professional Communications (DHPCs).
        DHPCs are urgent safety communications — the EU equivalent of FDA recalls.
        Returns parsed records with title, date, and URL.
        """
        url = "https://www.ema.europa.eu/en/human-regulatory-overview/post-authorisation/pharmacovigilance-post-authorisation/direct-healthcare-professional-communications"
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                r = await client.get(url)
                r.raise_for_status()
                # Parse the HTML response for DHPC links
                # EMA uses a structured table with date, product, and link
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(r.text, "html.parser")
                table = soup.find("table")
                results = []
                if table:
                    rows = table.find_all("tr")[1:]  # skip header
                    for row in rows[:limit]:
                        cells = row.find_all("td")
                        if len(cells) >= 2:
                            results.append({
                                "date": cells[0].get_text(strip=True) if cells else "",
                                "product": cells[1].get_text(strip=True) if len(cells) > 1 else "",
                                "link": cells[1].find("a")["href"] if cells[1].find("a") else "",
                                "source": "EMA DHPC",
                            })
                logger.info("EMA DHPCs: %d records parsed", len(results))
                return results
        except Exception as e:
            logger.warning("EMA DHPC fetch failed: %s", e)
            return []

    async def search_epar(
        self,
        search_term: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search EPAR (European Public Assessment Reports) for a device/product term.
        Useful for finding EU regulatory decisions on specific device categories.
        """
        url = "https://www.ema.europa.eu/en/medicines/search_api_render"
        params = {
            "q": search_term,
            "search_api_fulltext": search_term,
            "field_ema_medicine_type": "Human",
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json() if "application/json" in r.headers.get("content-type", "") else {}
                results = data.get("results", [])[:limit]
                logger.info("EMA EPAR search '%s': %d records", search_term, len(results))
                return results
        except Exception as e:
            logger.warning("EMA EPAR search failed: %s", e)
            return []

    async def get_product_shortages(self, limit: int = 20) -> list[dict]:
        """
        Fetch current EU medicine/device supply shortages.
        Relevant for combination products and PMS reports.
        """
        url = "https://www.ema.europa.eu/en/human-regulatory-overview/post-authorisation/availability-medicines/medicine-shortages"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url)
                r.raise_for_status()
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(r.text, "html.parser")
                table = soup.find("table")
                results = []
                if table:
                    rows = table.find_all("tr")[1:]
                    for row in rows[:limit]:
                        cells = row.find_all("td")
                        if len(cells) >= 1:
                            results.append({
                                "product": cells[0].get_text(strip=True),
                                "date": cells[1].get_text(strip=True) if len(cells) > 1 else "",
                                "source": "EMA Shortage",
                            })
                logger.info("EMA shortages: %d records parsed", len(results))
                return results
        except Exception as e:
            logger.warning("EMA shortages fetch failed: %s", e)
            return []

    def format_dhpc_summary(self, dhpcs: list[dict]) -> str:
        """Format DHPC records for RAA EU alert text."""
        if not dhpcs:
            return "No recent EMA DHPCs found."
        lines = [f"EMA Direct Healthcare Professional Communications: {len(dhpcs)} recent\n"]
        for d in dhpcs[:10]:
            lines.append(f"  [{d.get('date','?')}] {d.get('product','?')}")
        return "\n".join(lines)


# Singleton
ema_client = EMAClient()
