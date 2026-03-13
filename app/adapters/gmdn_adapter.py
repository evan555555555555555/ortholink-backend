"""
OrthoLink --- GMDN Agency Adapter (Global)

Provides async access to the Global Medical Device Nomenclature (GMDN) Agency
REST API for:
  - Term lookup by GMDN code
  - Free-text term search
  - GMDN-to-country-code mapping (FDA product code, EUDAMED code, etc.)

API reference: https://www.gmdnagency.org/Info.aspx?pageid=1002
Authentication: API key required. Read from settings.gmdn_api_key.
Rate limit: Varies by subscription tier; defaults to ~30 requests/minute.

Regulatory context:
  - GMDN codes are referenced by FDA (GUDID), EU (EUDAMED), AU (AusUDID),
    CA (MDALL), and many other national registries for device nomenclature
  - Each GMDN term has a unique 5-digit numeric code (e.g. "58266")
  - Collective terms group multiple specific GMDN codes under a broader category
  - Template numbers link GMDN terms to IVD and in-vitro test templates
  - Obsolete terms are retained with successor references for backwards compat

Usage:
    from app.adapters.gmdn_adapter import gmdn_adapter
    term = await gmdn_adapter.fetch_device("58266")  # generic lookup
    results = await gmdn_adapter.search_devices("knee prosthesis")
    mapping = await gmdn_adapter.map_to_country_code("58266", "US")
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.adapters.base_adapter import (
    BaseRegistryAdapter,
    DeviceRecord,
    RegistrationRecord,
)
from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GMDNCollectiveTerm(BaseModel):
    """A GMDN collective term that groups multiple specific GMDN codes."""

    collective_term_code: str = ""
    collective_term_name: str = ""
    description: str = ""


class GMDNTerm(BaseModel):
    """
    A single GMDN term record from the GMDN Agency API.

    GMDN terms are language-independent nomenclature entries that identify
    a specific category of medical device by function and technology.
    """

    gmdn_code: str = Field(..., description="5-digit GMDN term code")
    term_name: str = Field(default="", description="Preferred term name")
    definition: str = Field(default="", description="Full definition text")

    # Collective term grouping
    collective_term_code: str = ""
    collective_term_name: str = ""

    # Template reference
    template_number: str = ""        # Links to IVD or device template
    template_name: str = ""

    # Status
    is_obsolete: bool = False
    successor_code: str = ""         # If obsolete, the replacement GMDN code
    status: str = ""                 # Active | Obsolete | Reserved

    # Country-specific cross-references (populated by map_to_country_code)
    fda_product_code: str = ""       # US FDA product code (3-letter)
    eudamed_code: str = ""           # EU EUDAMED classification code
    tga_gmdn_code: str = ""          # AU TGA ARTG cross-reference
    iso_category: str = ""           # ISO 15223 / ISO 9999 category

    # Dates
    created_date: str = ""
    modified_date: str = ""

    raw: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class _GMDNDeviceRecord(DeviceRecord):
    """
    Internal DeviceRecord wrapper around a GMDNTerm.

    Required to satisfy BaseRegistryAdapter[T] interface. In practice,
    callers should use GMDNTerm directly via fetch_term() or search_terms().
    """
    source_registry: str = "GMDN"
    country: str = "GLOBAL"
    gmdn_code: str = ""
    term_name: str = ""
    definition: str = ""
    is_obsolete: bool = False


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


class GMDNAdapter(BaseRegistryAdapter[_GMDNDeviceRecord]):
    """
    Async adapter for the GMDN Agency REST API.

    API key is read from settings.gmdn_api_key and sent as Bearer token
    in the Authorization header (per GMDN Agency API v1 spec).

    Methods:
      fetch_term(gmdn_code) -- get a single GMDN term
      search_terms(query)   -- free-text term search
      map_to_country_code(gmdn_code, country) -- cross-reference mapping
      fetch_device(udi_di)  -- DeviceRecord interface (wraps fetch_term)
      search_devices(query) -- DeviceRecord interface (wraps search_terms)
    """

    BASE_URL = "https://www.gmdnagency.org/api"
    REGISTRY_NAME = "GMDN"
    DEFAULT_TIMEOUT = 15.0
    RATE_LIMIT_RPS = 0.5     # 30 req/min default tier
    RATE_LIMIT_BURST = 5

    def __init__(self) -> None:
        super().__init__()
        settings = get_settings()
        self._api_key: str = settings.gmdn_api_key
        if self._api_key:
            self._client_headers["Authorization"] = f"Bearer {self._api_key}"
        else:
            logger.warning(
                "[GMDN] No API key configured (settings.gmdn_api_key). "
                "Requests may be rejected with HTTP 401."
            )

    def get_source_url(self) -> str:
        return "https://www.gmdnagency.org"

    def _health_check_path(self) -> str:
        return "/v1/terms/search?q=test&pageSize=1"

    # ------------------------------------------------------------------
    # Primary GMDN-specific methods
    # ------------------------------------------------------------------

    async def fetch_term(self, gmdn_code: str) -> Optional[GMDNTerm]:
        """
        Look up a single GMDN term by its 5-digit code.

        Endpoint: GET /v1/terms/{gmdn_code}

        Args:
            gmdn_code: GMDN code (e.g. "58266"). Leading zeros are preserved.

        Returns:
            GMDNTerm or None if not found.
        """
        if not gmdn_code or not gmdn_code.strip():
            logger.warning("[GMDN] fetch_term called with empty gmdn_code")
            return None

        code = gmdn_code.strip()
        try:
            response = await self._get(f"/v1/terms/{code}")
            data = response.json()
            term = self._parse_term(data)
            if term:
                logger.info(
                    "[GMDN] Fetched term %s: %s",
                    code,
                    term.term_name,
                )
            return term

        except Exception as exc:
            logger.warning("[GMDN] fetch_term(%s) failed: %s", code, exc)
            return None

    async def search_terms(
        self, query: str, limit: int = 20
    ) -> list[GMDNTerm]:
        """
        Free-text search across GMDN terms by name or definition keywords.

        Endpoint: GET /v1/terms/search?q={query}&pageSize={limit}

        Args:
            query: Term name or keyword.
            limit: Maximum results (1–50).

        Returns:
            List of GMDNTerm records. Empty list on failure.
        """
        if not query or not query.strip():
            return []

        clamped = max(1, min(limit, 50))
        try:
            response = await self._get(
                "/v1/terms/search",
                params={"q": query.strip(), "pageSize": clamped},
            )
            data = response.json()

            raw_list = data if isinstance(data, list) else data.get(
                "terms", data.get("data", data.get("results", []))
            )
            if not isinstance(raw_list, list):
                raw_list = []

            results: list[GMDNTerm] = []
            for entry in raw_list:
                term = self._parse_term(entry)
                if term:
                    results.append(term)

            logger.info("[GMDN] search '%s': %d terms", query, len(results))
            return results

        except Exception as exc:
            logger.warning("[GMDN] search_terms('%s') failed: %s", query, exc)
            return []

    async def map_to_country_code(
        self, gmdn_code: str, country: str
    ) -> dict[str, str]:
        """
        Map a GMDN code to country/registry-specific classification codes.

        The GMDN API v1 provides a cross-reference endpoint:
          GET /v1/terms/{gmdn_code}/cross-references?country={country}

        Returns a dict with known cross-references for the target country,
        for example:
          {
            "gmdn_code":        "58266",
            "country":          "US",
            "fda_product_code": "KYF",
            "fda_regulation":   "888.3560",
            "device_class":     "II",
          }

        Supported country keys in the response:
          US  -> fda_product_code, fda_regulation, device_class
          EU  -> eudamed_code, mdr_class, nomenclature_code
          AU  -> tga_gmdn_code, artg_device_type
          CA  -> mdall_reference
          IN  -> cdsco_category
          BR  -> anvisa_category
          JP  -> pmda_class

        Args:
            gmdn_code: 5-digit GMDN code.
            country: ISO 3166-1 alpha-2 code.

        Returns:
            Dict of cross-reference fields. Empty dict on failure or if no
            mapping exists for the requested country.
        """
        if not gmdn_code or not country:
            return {}

        code = gmdn_code.strip()
        cc = country.strip().upper()

        try:
            response = await self._get(
                f"/v1/terms/{code}/cross-references",
                params={"country": cc},
            )
            data = response.json()

            if isinstance(data, dict):
                # Normalise the result to have gmdn_code and country keys
                result = {
                    "gmdn_code": code,
                    "country": cc,
                }
                result.update({
                    k: self._safe_str(v)
                    for k, v in data.items()
                    if v is not None
                })
                logger.info(
                    "[GMDN] map_to_country_code(%s, %s): %d keys",
                    code, cc, len(result),
                )
                return result

            logger.debug("[GMDN] No cross-reference data for %s / %s", code, cc)
            return {"gmdn_code": code, "country": cc}

        except Exception as exc:
            logger.warning(
                "[GMDN] map_to_country_code(%s, %s) failed: %s", code, cc, exc
            )
            return {"gmdn_code": code, "country": cc}

    # ------------------------------------------------------------------
    # BaseRegistryAdapter interface (wraps GMDN-specific methods)
    # ------------------------------------------------------------------

    async def fetch_device(self, udi_di: str) -> Optional[_GMDNDeviceRecord]:
        """
        Satisfy BaseRegistryAdapter interface: treat udi_di as a GMDN code.

        Returns a _GMDNDeviceRecord wrapping the GMDNTerm data, or None.
        For richer data, use fetch_term() directly.
        """
        term = await self.fetch_term(udi_di)
        if not term:
            return None
        return self._term_to_record(term)

    async def search_devices(
        self, query: str, limit: int = 20
    ) -> list[_GMDNDeviceRecord]:
        """
        Satisfy BaseRegistryAdapter interface: wraps search_terms().

        Returns _GMDNDeviceRecord list. For richer data use search_terms().
        """
        terms = await self.search_terms(query, limit=limit)
        return [self._term_to_record(t) for t in terms if t]

    async def fetch_registrations(self, country: str) -> list[RegistrationRecord]:
        """
        GMDN is a global nomenclature registry, not a national registration DB.
        Returns empty list; not applicable to GMDN.
        """
        logger.debug(
            "[GMDN] fetch_registrations: GMDN is nomenclature-only; returning []"
        )
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_term(self, raw: dict[str, Any]) -> Optional[GMDNTerm]:
        """Parse a single GMDN API response entry into a GMDNTerm."""
        try:
            gmdn_code = self._safe_str(
                raw.get("code",
                raw.get("gmdnCode",
                raw.get("termCode", "")))
            )
            if not gmdn_code:
                return None

            term_name = self._safe_str(
                raw.get("name",
                raw.get("termName",
                raw.get("preferredTerm", "")))
            )
            definition = self._safe_str(
                raw.get("definition",
                raw.get("termDefinition",
                raw.get("description", "")))
            )

            # Collective term
            ct_raw = raw.get("collectiveTerm", raw.get("collectiveTermDetails", {}))
            if isinstance(ct_raw, dict):
                coll_code = self._safe_str(
                    ct_raw.get("code", ct_raw.get("collectiveCode", ""))
                )
                coll_name = self._safe_str(
                    ct_raw.get("name", ct_raw.get("collectiveName", ""))
                )
            else:
                coll_code = self._safe_str(
                    raw.get("collectiveTermCode", "")
                )
                coll_name = self._safe_str(
                    raw.get("collectiveTermName", "")
                )

            # Template
            tmpl_raw = raw.get("template", raw.get("templateDetails", {}))
            if isinstance(tmpl_raw, dict):
                tmpl_number = self._safe_str(
                    tmpl_raw.get("number", tmpl_raw.get("templateNumber", ""))
                )
                tmpl_name = self._safe_str(
                    tmpl_raw.get("name", tmpl_raw.get("templateName", ""))
                )
            else:
                tmpl_number = self._safe_str(raw.get("templateNumber", ""))
                tmpl_name = self._safe_str(raw.get("templateName", ""))

            # Status
            status_raw = self._safe_str(
                raw.get("status", raw.get("termStatus", ""))
            )
            is_obsolete = bool(
                raw.get("isObsolete",
                raw.get("obsolete",
                status_raw.upper() == "OBSOLETE"))
            )
            successor_code = self._safe_str(
                raw.get("successorCode",
                raw.get("replacedBy", ""))
            )

            return GMDNTerm(
                gmdn_code=gmdn_code,
                term_name=term_name,
                definition=definition,
                collective_term_code=coll_code,
                collective_term_name=coll_name,
                template_number=tmpl_number,
                template_name=tmpl_name,
                is_obsolete=is_obsolete,
                successor_code=successor_code,
                status=status_raw or ("Obsolete" if is_obsolete else "Active"),
                created_date=self._safe_str(
                    raw.get("createdDate",
                    raw.get("dateCreated", ""))
                ),
                modified_date=self._safe_str(
                    raw.get("modifiedDate",
                    raw.get("lastModified", ""))
                ),
                raw=raw,
                fetched_at=datetime.now(timezone.utc),
            )

        except Exception as exc:
            logger.warning("[GMDN] Failed to parse term entry: %s", exc)
            return None

    def _term_to_record(self, term: GMDNTerm) -> _GMDNDeviceRecord:
        """Convert a GMDNTerm into a _GMDNDeviceRecord for adapter interface."""
        return _GMDNDeviceRecord(
            device_id=term.gmdn_code,
            device_name=term.term_name,
            manufacturer="",
            description=term.definition,
            risk_class="",
            gmdn_code=term.gmdn_code,
            term_name=term.term_name,
            definition=term.definition,
            is_obsolete=term.is_obsolete,
            raw=term.raw,
            fetched_at=term.fetched_at,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

gmdn_adapter = GMDNAdapter()
