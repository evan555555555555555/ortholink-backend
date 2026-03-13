"""
OrthoLink --- Health Canada MDALL Adapter (Canada)

Provides async access to Health Canada's Medical Device Active Licence Listing
(MDALL) and Medical Device Establishment Licence (MDEL) public REST APIs for:
  - Device licence lookup by licence number
  - Device name search with pagination
  - Establishment (MDEL) lookup for manufacturers and importers

API reference: https://health-products.canada.ca/api/v1/mdall
Authentication: None required.
Rate limit: ~60 requests/minute (Health Canada guidance).

Regulatory context:
  - Medical Device Regulations SOR/98-282 governs all Canadian market devices
  - Class II/III/IV devices require a Medical Device Licence (MDL)
  - Class I devices require establishment licence (MDEL) but no MDL
  - Importers and distributors must hold MDEL
  - Canada does not use IVDR; IVDs follow a separate classification table
  - MDR SOR/98-282 aligns broadly with EU MDR for non-IVD devices

Usage:
    from app.adapters.mdall_adapter import mdall_adapter
    device = await mdall_adapter.fetch_device("98765")   # licence number
    estab = await mdall_adapter.get_establishment("6789")  # MDEL number
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MDELEstablishment(BaseModel):
    """Health Canada Medical Device Establishment Licence record."""

    mdel_number: str = ""
    company_name: str = ""
    address_street: str = ""
    address_city: str = ""
    address_province: str = ""       # 2-letter CA province code
    address_country: str = ""
    postal_code: str = ""

    # Establishment activities
    activities: list[str] = Field(default_factory=list)
    # e.g. ["Manufacturer", "Importer", "Distributor", "Specification Developer"]

    licence_status: str = ""         # Active | Cancelled | Suspended
    issue_date: str = ""
    expiry_date: str = ""

    raw: dict[str, Any] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MDALLDevice(DeviceRecord):
    """
    Medical device licence record from Health Canada MDALL.

    Covers SOR/98-282 Class II, III, and IV medical device licences.
    Class I devices are not individually licensed but appear in MDEL records.
    """

    source_registry: str = "MDALL"
    country: str = "CA"

    # Licence details
    licence_number: str = ""         # Medical Device Licence number
    licence_class: str = ""          # II | III | IV (Class I not individually licensed)
    licence_status: str = ""         # Active | Cancelled | Suspended | Expired

    # Company
    company_name: str = ""
    contact_name: str = ""

    # Establishment link
    mdel_licence: str = ""           # Associated MDEL establishment number

    # Scope and conditions
    conditions: list[str] = Field(default_factory=list)
    indications_for_use: str = ""

    # Dates
    issue_date: str = ""
    expiry_date: str = ""
    amendment_date: str = ""

    # Device identifiers
    model_number: str = ""
    catalogue_number: str = ""

    # Standards compliance declared by applicant
    standards: list[str] = Field(default_factory=list)
    # e.g. ["ISO 13485:2016", "IEC 60601-1"]


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


class MDALLAdapter(BaseRegistryAdapter[MDALLDevice]):
    """
    Async adapter for Health Canada MDALL and MDEL REST APIs.

    Provides single-licence lookup, free-text device name search,
    and establishment (MDEL) record retrieval.
    """

    BASE_URL = "https://health-products.canada.ca/api/v1/mdall"
    REGISTRY_NAME = "MDALL"
    DEFAULT_TIMEOUT = 20.0
    RATE_LIMIT_RPS = 1.0    # 60 req/min; conservative
    RATE_LIMIT_BURST = 8

    def get_source_url(self) -> str:
        return "https://health-products.canada.ca/mdall-limh"

    def _health_check_path(self) -> str:
        # Probe with a search for a single result
        return "/devices?pageSize=1"

    # ------------------------------------------------------------------
    # fetch_device
    # ------------------------------------------------------------------

    async def fetch_device(self, licence_number: str) -> Optional[MDALLDevice]:
        """
        Fetch a single medical device licence by its MDALL licence number.

        Args:
            licence_number: MDALL device licence number (e.g. "98765").

        Returns:
            MDALLDevice or None if not found.
        """
        if not licence_number or not licence_number.strip():
            logger.warning("[MDALL] fetch_device called with empty licence_number")
            return None

        lic = licence_number.strip()
        try:
            response = await self._get(f"/devices/{lic}")
            data = response.json()
            device = self._parse_device(data)
            if device:
                logger.info(
                    "[MDALL] Fetched licence %s: %s",
                    lic,
                    device.device_name,
                )
            return device

        except Exception as exc:
            logger.warning("[MDALL] fetch_device(%s) failed: %s", lic, exc)
            return None

    # ------------------------------------------------------------------
    # search_devices
    # ------------------------------------------------------------------

    async def search_devices(
        self, query: str, limit: int = 20
    ) -> list[MDALLDevice]:
        """
        Search device licences by device name.

        Endpoint: GET /devices?deviceName={query}&pageSize={limit}

        Args:
            query: Device name or partial name.
            limit: Maximum results (1–100).

        Returns:
            List of MDALLDevice records. Empty list on failure.
        """
        if not query or not query.strip():
            return []

        clamped = max(1, min(limit, 100))
        try:
            response = await self._get(
                "/devices",
                params={"deviceName": query.strip(), "pageSize": clamped},
            )
            data = response.json()

            raw_list = data if isinstance(data, list) else data.get(
                "data", data.get("items", data.get("results", []))
            )
            if not isinstance(raw_list, list):
                raw_list = []

            results: list[MDALLDevice] = []
            for entry in raw_list:
                device = self._parse_device(entry)
                if device:
                    results.append(device)

            logger.info("[MDALL] search '%s': %d results", query, len(results))
            return results

        except Exception as exc:
            logger.warning("[MDALL] search_devices('%s') failed: %s", query, exc)
            return []

    # ------------------------------------------------------------------
    # fetch_registrations
    # ------------------------------------------------------------------

    async def fetch_registrations(self, country: str) -> list[RegistrationRecord]:
        """
        MDALL is CA-only. Returns empty for non-CA countries.

        For CA, performs a broad active-licence search.
        """
        if country.upper() != "CA":
            logger.debug(
                "[MDALL] fetch_registrations skipped for country=%s (CA only)",
                country,
            )
            return []

        try:
            response = await self._get(
                "/devices",
                params={"licenceStatus": "Active", "pageSize": 100},
            )
            data = response.json()
            raw_list = data if isinstance(data, list) else data.get("data", [])
            if not isinstance(raw_list, list):
                raw_list = []

            records: list[RegistrationRecord] = []
            for entry in raw_list:
                records.append(
                    RegistrationRecord(
                        registry="MDALL",
                        country="CA",
                        device_id=self._safe_str(
                            entry.get("licenceNumber", entry.get("deviceLicenceNumber", ""))
                        ),
                        device_name=self._safe_str(
                            entry.get("deviceName", entry.get("name", ""))
                        ),
                        manufacturer=self._safe_str(
                            entry.get("companyName", entry.get("manufacturerName", ""))
                        ),
                        status=self._safe_str(
                            entry.get("licenceStatus", entry.get("status", ""))
                        ),
                        valid_until=self._safe_str(
                            entry.get("expiryDate", entry.get("expiry_date", ""))
                        ),
                        raw=entry,
                    )
                )

            logger.info("[MDALL] fetch_registrations(CA): %d records", len(records))
            return records

        except Exception as exc:
            logger.warning("[MDALL] fetch_registrations failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # get_establishment
    # ------------------------------------------------------------------

    async def get_establishment(self, mdel_number: str) -> Optional[MDELEstablishment]:
        """
        Fetch an establishment record by its MDEL (Medical Device Establishment
        Licence) number.

        Endpoint: GET /establishments/{mdel_number}

        Args:
            mdel_number: MDEL licence number (e.g. "6789").

        Returns:
            MDELEstablishment or None if not found.
        """
        if not mdel_number or not mdel_number.strip():
            logger.warning("[MDALL] get_establishment called with empty mdel_number")
            return None

        mdel = mdel_number.strip()
        try:
            response = await self._get(f"/establishments/{mdel}")
            data = response.json()
            estab = self._parse_establishment(data)
            if estab:
                logger.info(
                    "[MDALL] Fetched MDEL %s: %s", mdel, estab.company_name
                )
            return estab

        except Exception as exc:
            logger.warning("[MDALL] get_establishment(%s) failed: %s", mdel, exc)
            return None

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_device(self, raw: dict[str, Any]) -> Optional[MDALLDevice]:
        """
        Parse a single MDALL API response into MDALLDevice.

        Health Canada returns both camelCase and snake_case variants
        depending on API version; handle both.
        """
        try:
            # Licence number
            licence_number = self._safe_str(
                raw.get("licenceNumber",
                raw.get("deviceLicenceNumber",
                raw.get("licence_number", "")))
            )
            device_name = self._safe_str(
                raw.get("deviceName",
                raw.get("device_name",
                raw.get("name", "")))
            )

            # Class normalisation — SOR/98-282 uses numeric + roman numerals
            raw_class = self._safe_str(
                raw.get("licenceClass",
                raw.get("deviceClass",
                raw.get("class", "")))
            )
            licence_class = self._normalise_class(raw_class)

            company_name = self._safe_str(
                raw.get("companyName",
                raw.get("company_name",
                raw.get("manufacturerName", "")))
            )

            # Conditions (may be a list or a delimited string)
            conditions_raw = raw.get("conditions", raw.get("licenceConditions", []))
            if isinstance(conditions_raw, str):
                conditions = [c.strip() for c in conditions_raw.split(";") if c.strip()]
            elif isinstance(conditions_raw, list):
                conditions = [self._safe_str(c) for c in conditions_raw if c]
            else:
                conditions = []

            # Standards compliance
            standards_raw = raw.get("standards", raw.get("complianceStandards", []))
            if isinstance(standards_raw, str):
                standards = [s.strip() for s in standards_raw.split(";") if s.strip()]
            elif isinstance(standards_raw, list):
                standards = [self._safe_str(s) for s in standards_raw if s]
            else:
                standards = []

            device_id = licence_number or device_name[:30]
            if not device_id:
                return None

            return MDALLDevice(
                device_id=device_id,
                device_name=device_name,
                manufacturer=company_name,
                description=self._safe_str(
                    raw.get("indicationsForUse",
                    raw.get("intendedUse",
                    raw.get("description", "")))
                ),
                risk_class=licence_class,
                licence_number=licence_number,
                licence_class=licence_class,
                licence_status=self._safe_str(
                    raw.get("licenceStatus",
                    raw.get("status", ""))
                ),
                company_name=company_name,
                contact_name=self._safe_str(
                    raw.get("contactName",
                    raw.get("contact_name", ""))
                ),
                mdel_licence=self._safe_str(
                    raw.get("mdelLicence",
                    raw.get("mdelNumber",
                    raw.get("establishmentLicenceNumber", "")))
                ),
                conditions=conditions,
                indications_for_use=self._safe_str(
                    raw.get("indicationsForUse",
                    raw.get("intendedUse", ""))
                ),
                issue_date=self._safe_str(
                    raw.get("issueDate",
                    raw.get("issue_date", ""))
                ),
                expiry_date=self._safe_str(
                    raw.get("expiryDate",
                    raw.get("expiry_date", ""))
                ),
                amendment_date=self._safe_str(
                    raw.get("amendmentDate",
                    raw.get("amendment_date", ""))
                ),
                model_number=self._safe_str(
                    raw.get("modelNumber",
                    raw.get("model_number", ""))
                ),
                catalogue_number=self._safe_str(
                    raw.get("catalogueNumber",
                    raw.get("catalogNumber",
                    raw.get("catalog_number", "")))
                ),
                standards=standards,
                raw=raw,
                fetched_at=datetime.now(timezone.utc),
            )

        except Exception as exc:
            logger.warning("[MDALL] Failed to parse device entry: %s", exc)
            return None

    def _parse_establishment(
        self, raw: dict[str, Any]
    ) -> Optional[MDELEstablishment]:
        """Parse a single Health Canada MDEL establishment record."""
        try:
            mdel_number = self._safe_str(
                raw.get("mdelNumber",
                raw.get("establishmentLicenceNumber",
                raw.get("mdel_number", "")))
            )
            company_name = self._safe_str(
                raw.get("companyName",
                raw.get("company_name", ""))
            )

            # Address fields
            address = raw.get("address", {})
            if isinstance(address, dict):
                street = self._safe_str(address.get("street", address.get("streetAddress", "")))
                city = self._safe_str(address.get("city", ""))
                province = self._safe_str(address.get("province", address.get("state", "")))
                postal = self._safe_str(address.get("postalCode", address.get("zip", "")))
                addr_country = self._safe_str(address.get("country", "CA"))
            else:
                street = self._safe_str(raw.get("street", ""))
                city = self._safe_str(raw.get("city", ""))
                province = self._safe_str(raw.get("province", ""))
                postal = self._safe_str(raw.get("postalCode", ""))
                addr_country = "CA"

            # Activities (may be list or semicolon-delimited string)
            activities_raw = raw.get("activities", raw.get("licenceActivities", []))
            if isinstance(activities_raw, str):
                activities = [a.strip() for a in activities_raw.split(";") if a.strip()]
            elif isinstance(activities_raw, list):
                activities = [self._safe_str(a) for a in activities_raw if a]
            else:
                activities = []

            return MDELEstablishment(
                mdel_number=mdel_number,
                company_name=company_name,
                address_street=street,
                address_city=city,
                address_province=province,
                address_country=addr_country,
                postal_code=postal,
                activities=activities,
                licence_status=self._safe_str(
                    raw.get("licenceStatus", raw.get("status", ""))
                ),
                issue_date=self._safe_str(raw.get("issueDate", "")),
                expiry_date=self._safe_str(raw.get("expiryDate", "")),
                raw=raw,
                fetched_at=datetime.now(timezone.utc),
            )

        except Exception as exc:
            logger.warning("[MDALL] Failed to parse establishment entry: %s", exc)
            return None

    @staticmethod
    def _normalise_class(raw_class: str) -> str:
        """
        Normalise a raw class string from the MDALL API to canonical form.

        Health Canada may return "2", "II", "Class II", "CLASS 2", etc.
        Returns one of: "I", "II", "III", "IV", or the original string.
        """
        if not raw_class:
            return ""
        cleaned = raw_class.strip().upper().replace("CLASS ", "").replace("CLASSE ", "")
        mapping = {
            "1": "I", "I": "I",
            "2": "II", "II": "II",
            "3": "III", "III": "III",
            "4": "IV", "IV": "IV",
        }
        return mapping.get(cleaned, raw_class.strip())


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

mdall_adapter = MDALLAdapter()
