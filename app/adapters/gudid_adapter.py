"""
OrthoLink --- AccessGUDID Adapter (USA)

Provides async access to the NLM AccessGUDID REST API v3 for:
  - Single device lookup by UDI-DI (primary or secondary)
  - Free-text device search
  - Registration listing for US-market devices

API docs: https://accessgudid.nlm.nih.gov/resources/developers/v3/api_documentation
No authentication required. Rate limit: ~240 requests/minute (undocumented soft limit).

Usage:
    from app.adapters.gudid_adapter import gudid_adapter
    device = await gudid_adapter.fetch_device("08717648200274")
    results = await gudid_adapter.search_devices("knee implant", limit=10)
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
# Pydantic models for AccessGUDID responses
# ---------------------------------------------------------------------------


class GUDIDProductCode(BaseModel):
    """FDA product code associated with a device."""

    product_code: str = ""
    product_code_name: str = ""
    device_class: str = ""
    regulation_number: str = ""


class GUDIDIdentifier(BaseModel):
    """Device identifier (UDI-DI, secondary DI, unit-of-use DI, etc.)."""

    device_id: str = ""
    device_id_type: str = ""
    device_id_issuing_agency: str = ""


class GUDIDGMDNTerm(BaseModel):
    """Global Medical Device Nomenclature term."""

    gmdn_pt_name: str = ""
    gmdn_pt_definition: str = ""
    gmdn_code: str = ""


class GUDIDDevice(DeviceRecord):
    """Full AccessGUDID device record with US-specific fields."""

    source_registry: str = "GUDID"
    country: str = "US"

    brand_name: str = ""
    version_model_number: str = ""
    company_name: str = ""
    catalog_number: str = ""
    device_description: str = ""
    device_count_in_base_package: int = 1

    # Classification
    device_class: str = ""
    premarket_submission_number: str = ""

    # GMDN and product codes
    gmdn_terms: list[GUDIDGMDNTerm] = Field(default_factory=list)
    product_codes: list[GUDIDProductCode] = Field(default_factory=list)
    identifiers: list[GUDIDIdentifier] = Field(default_factory=list)

    # Status
    commercial_distribution_status: str = ""
    device_publish_date: str = ""
    mri_safety_status: str = ""

    # Sterilization & single-use
    is_single_use: Optional[bool] = None
    is_sterile: Optional[bool] = None
    sterilization_methods: list[str] = Field(default_factory=list)

    # Contact
    customer_contact_phone: str = ""


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


class GUDIDAdapter(BaseRegistryAdapter[GUDIDDevice]):
    """
    AsyncGUDID adapter for the NLM AccessGUDID v3 REST API.
    """

    BASE_URL = "https://accessgudid.nlm.nih.gov/api/v3"
    REGISTRY_NAME = "GUDID"
    DEFAULT_TIMEOUT = 15.0
    RATE_LIMIT_RPS = 4.0  # conservative; API soft-limits at ~4/s
    RATE_LIMIT_BURST = 8

    def get_source_url(self) -> str:
        return "https://accessgudid.nlm.nih.gov"

    def _health_check_path(self) -> str:
        # Lightweight call: search with an empty query returns quickly
        return "/devices/search.json?query=test&pageSize=1"

    # ------------------------------------------------------------------
    # fetch_device
    # ------------------------------------------------------------------

    async def fetch_device(self, udi_di: str) -> Optional[GUDIDDevice]:
        """
        Look up a device by its UDI-DI (primary device identifier).
        Returns GUDIDDevice or None.
        """
        if not udi_di or not udi_di.strip():
            logger.warning("[GUDID] fetch_device called with empty udi_di")
            return None

        try:
            response = await self._get(
                "/devices/lookup.json",
                params={"di": udi_di.strip()},
            )
            data = response.json()
            device = self._parse_device(data)
            if device:
                logger.info(
                    "[GUDID] Fetched device %s: %s",
                    udi_di,
                    device.brand_name or device.device_name,
                )
            return device

        except Exception as exc:
            logger.warning("[GUDID] fetch_device(%s) failed: %s", udi_di, exc)
            return None

    # ------------------------------------------------------------------
    # search_devices
    # ------------------------------------------------------------------

    async def search_devices(
        self, query: str, limit: int = 20
    ) -> list[GUDIDDevice]:
        """
        Free-text search on AccessGUDID.
        Returns up to *limit* GUDIDDevice records.
        """
        if not query or not query.strip():
            return []

        clamped_limit = max(1, min(limit, 100))

        try:
            response = await self._get(
                "/devices/search.json",
                params={"query": query.strip(), "pageSize": clamped_limit},
            )
            data = response.json()
            devices_raw = data.get("devices", [])

            results: list[GUDIDDevice] = []
            for entry in devices_raw:
                device = self._parse_device(entry)
                if device:
                    results.append(device)

            logger.info(
                "[GUDID] Search '%s': %d/%d parsed",
                query,
                len(results),
                len(devices_raw),
            )
            return results

        except Exception as exc:
            logger.warning("[GUDID] search_devices('%s') failed: %s", query, exc)
            return []

    # ------------------------------------------------------------------
    # fetch_registrations
    # ------------------------------------------------------------------

    async def fetch_registrations(self, country: str) -> list[RegistrationRecord]:
        """
        AccessGUDID is US-only; returns empty list for non-US country codes.
        For US, performs a broad search and maps to RegistrationRecord.
        """
        if country.upper() != "US":
            logger.debug(
                "[GUDID] fetch_registrations skipped for country=%s (US only)",
                country,
            )
            return []

        try:
            response = await self._get(
                "/devices/search.json",
                params={"query": "*", "pageSize": 100},
            )
            data = response.json()
            devices_raw = data.get("devices", [])

            records: list[RegistrationRecord] = []
            for entry in devices_raw:
                dev_data = entry.get("device", entry)
                records.append(
                    RegistrationRecord(
                        registry="GUDID",
                        country="US",
                        device_id=self._safe_str(
                            dev_data.get("primaryDi", dev_data.get("deviceId", ""))
                        ),
                        device_name=self._safe_str(
                            dev_data.get("brandName", dev_data.get("deviceDescription", ""))
                        ),
                        manufacturer=self._safe_str(dev_data.get("companyName", "")),
                        status=self._safe_str(
                            dev_data.get("commercialDistributionStatus", "")
                        ),
                        raw=dev_data,
                    )
                )

            logger.info(
                "[GUDID] fetch_registrations(US): %d records", len(records)
            )
            return records

        except Exception as exc:
            logger.warning("[GUDID] fetch_registrations failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_device(self, raw_entry: dict[str, Any]) -> Optional[GUDIDDevice]:
        """
        Parse a single AccessGUDID API response into GUDIDDevice.
        The API wraps the device inside {"device": {...}} for lookup,
        and {"devices": [{"device": {...}}, ...]} for search.
        """
        try:
            dev = raw_entry.get("device", raw_entry)

            # GMDN terms
            gmdn_raw = self._safe_list(dev.get("gmdnTerms", []))
            gmdn_terms = [
                GUDIDGMDNTerm(
                    gmdn_pt_name=self._safe_str(g.get("gmdnPTName", "")),
                    gmdn_pt_definition=self._safe_str(g.get("gmdnPTDefinition", "")),
                    gmdn_code=self._safe_str(g.get("gmdnCode", "")),
                )
                for g in gmdn_raw
                if isinstance(g, dict)
            ]

            # Product codes
            pc_raw = self._safe_list(dev.get("productCodes", []))
            product_codes = [
                GUDIDProductCode(
                    product_code=self._safe_str(pc.get("productCode", "")),
                    product_code_name=self._safe_str(pc.get("productCodeName", "")),
                    device_class=self._safe_str(pc.get("deviceClass", "")),
                    regulation_number=self._safe_str(pc.get("regulationNumber", "")),
                )
                for pc in pc_raw
                if isinstance(pc, dict)
            ]

            # Identifiers
            id_raw = self._safe_list(dev.get("identifiers", []))
            identifiers = [
                GUDIDIdentifier(
                    device_id=self._safe_str(i.get("deviceId", "")),
                    device_id_type=self._safe_str(i.get("deviceIdType", "")),
                    device_id_issuing_agency=self._safe_str(
                        i.get("deviceIdIssuingAgency", "")
                    ),
                )
                for i in id_raw
                if isinstance(i, dict)
            ]

            # Sterilization
            sterilization_raw = self._safe_list(
                dev.get("sterilization", {}).get("sterilizationMethods", [])
                if isinstance(dev.get("sterilization"), dict)
                else []
            )
            sterilization_methods = [
                self._safe_str(s) for s in sterilization_raw if s
            ]

            primary_di = self._safe_str(
                dev.get("primaryDi", dev.get("deviceId", ""))
            )
            brand_name = self._safe_str(dev.get("brandName", ""))
            company_name = self._safe_str(dev.get("companyName", ""))
            description = self._safe_str(dev.get("deviceDescription", ""))
            device_class_val = self._safe_str(dev.get("deviceClass", ""))

            # Derive device class from product codes if missing at top level
            if not device_class_val and product_codes:
                device_class_val = product_codes[0].device_class

            return GUDIDDevice(
                device_id=primary_di,
                device_name=brand_name or description[:120],
                manufacturer=company_name,
                description=description,
                risk_class=device_class_val,
                brand_name=brand_name,
                version_model_number=self._safe_str(
                    dev.get("versionModelNumber", "")
                ),
                company_name=company_name,
                catalog_number=self._safe_str(dev.get("catalogNumber", "")),
                device_description=description,
                device_count_in_base_package=int(
                    dev.get("deviceCountInBasePackage", 1) or 1
                ),
                device_class=device_class_val,
                premarket_submission_number=self._safe_str(
                    dev.get("premarketSubmissionNumber", "")
                ),
                gmdn_terms=gmdn_terms,
                product_codes=product_codes,
                identifiers=identifiers,
                commercial_distribution_status=self._safe_str(
                    dev.get("commercialDistributionStatus", "")
                ),
                device_publish_date=self._safe_str(
                    dev.get("devicePublishDate", "")
                ),
                mri_safety_status=self._safe_str(
                    dev.get("MRISafetyStatus", "")
                ),
                is_single_use=dev.get("deviceSterile", {}).get("isSingleUse")
                if isinstance(dev.get("deviceSterile"), dict)
                else None,
                is_sterile=dev.get("deviceSterile", {}).get("isSterile")
                if isinstance(dev.get("deviceSterile"), dict)
                else None,
                sterilization_methods=sterilization_methods,
                customer_contact_phone=self._safe_str(
                    dev.get("customerContactPhone", "")
                ),
                raw=dev,
                fetched_at=datetime.now(timezone.utc),
            )

        except Exception as exc:
            logger.warning("[GUDID] Failed to parse device entry: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

gudid_adapter = GUDIDAdapter()
