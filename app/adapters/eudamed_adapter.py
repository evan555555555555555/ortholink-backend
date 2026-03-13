"""
OrthoLink --- EUDAMED Adapter (European Union)

Provides async access to the European Database on Medical Devices (EUDAMED)
public REST API v1 for:
  - Single device lookup by UDI-DI (Basic UDI-DI)
  - Free-text device search
  - Field Safety Notices (FSNs / DHPCs)

API reference: https://ec.europa.eu/tools/eudamed/api/swagger-ui.html
Authentication: None required for public read endpoints.
Rate limit: ~60 requests/minute (EU Commission soft cap).

Regulatory context:
  - EUDAMED mandatory for MDR devices from May 28 2026
  - Legacy MDD/AIMDD certificates also indexed during transition
  - Device classes: I, IIa, IIb, III (MDR), IVD-A through IVD-D (IVDR)
  - Notified Body (NB) details embedded in certification records

Usage:
    from app.adapters.eudamed_adapter import eudamed_adapter
    device = await eudamed_adapter.fetch_device("04046719004547")
    notices = await eudamed_adapter.fetch_field_safety_notices(days=30)
"""

import logging
from datetime import datetime, timedelta, timezone
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


class EUDAMEDCertificate(BaseModel):
    """Notified Body certificate attached to a EUDAMED device record."""

    certificate_number: str = ""
    certificate_type: str = ""       # e.g. "EC_DESIGN_EXAMINATION", "EU_TECHNICAL_DOC"
    notified_body_id: str = ""       # NB registration number (e.g. "0086")
    notified_body_name: str = ""
    country_of_nb: str = ""          # ISO 3166-1 alpha-2
    valid_from: str = ""
    valid_until: str = ""
    status: str = ""                 # active | suspended | withdrawn | expired


class EUDAMEDDevice(DeviceRecord):
    """Full EUDAMED device record with EU MDR/IVDR-specific fields."""

    source_registry: str = "EUDAMED"
    country: str = "EU"

    # EUDAMED primary identifiers
    basic_udi_di: str = ""           # The Basic UDI-DI (root identifier)
    trade_name: str = ""
    intended_purpose: str = ""

    # Classification
    device_class: str = ""           # MDR: I, IIa, IIb, III | IVDR: A, B, C, D
    regulation: str = ""             # MDR (2017/745) or IVDR (2017/746)

    # Manufacturer details
    manufacturer_address: str = ""
    manufacturer_country: str = ""
    srn: str = ""                    # Single Registration Number of manufacturer

    # Certification
    certificates: list[EUDAMEDCertificate] = Field(default_factory=list)
    certification_number: str = ""   # Primary active certificate number
    nbr_organization: str = ""       # Primary NB name

    # Distribution / market
    country_of_registration: str = ""
    market_status: str = ""          # on_market | withdrawn | etc.

    # Legacy / transition
    legacy_mdd_ref: str = ""         # Old MDD/AIMDD reference if applicable


class EUDAMEDFieldSafetyNotice(BaseModel):
    """A Field Safety Notice (FSN) published in EUDAMED."""

    fsn_id: str = ""
    reference_number: str = ""
    title: str = ""
    description: str = ""
    manufacturer_name: str = ""
    manufacturer_srn: str = ""
    affected_devices: list[str] = Field(default_factory=list)   # UDI-DIs
    countries_affected: list[str] = Field(default_factory=list) # ISO alpha-2
    published_date: str = ""
    last_updated: str = ""
    severity: str = ""   # SERIOUS_PUBLIC_HEALTH_THREAT | HAZARDOUS | MINOR
    fsca_type: str = ""  # FSN type code
    source_url: str = ""


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


class EUDAMEDAdapter(BaseRegistryAdapter[EUDAMEDDevice]):
    """
    Async adapter for the EUDAMED public REST API v1.

    Base URL for device queries: https://ec.europa.eu/tools/eudamed/api
    Mandatory compliance date: May 28 2026 (MDR transition fully active).
    """

    BASE_URL = "https://ec.europa.eu/tools/eudamed/api"
    REGISTRY_NAME = "EUDAMED"
    DEFAULT_TIMEOUT = 20.0
    RATE_LIMIT_RPS = 1.5   # conservative; EU Commission infrastructure
    RATE_LIMIT_BURST = 5

    def get_source_url(self) -> str:
        return "https://ec.europa.eu/tools/eudamed"

    def _health_check_path(self) -> str:
        # Lightweight probe — search for a single result
        return "/devices/basicUdiDis?pageSize=1"

    # ------------------------------------------------------------------
    # fetch_device
    # ------------------------------------------------------------------

    async def fetch_device(self, udi_di: str) -> Optional[EUDAMEDDevice]:
        """
        Look up a single device by its Basic UDI-DI.

        Args:
            udi_di: The Basic UDI-DI string (e.g. "04046719004547").

        Returns:
            EUDAMEDDevice or None if not found / error.
        """
        if not udi_di or not udi_di.strip():
            logger.warning("[EUDAMED] fetch_device called with empty udi_di")
            return None

        basic_udi = udi_di.strip()
        try:
            response = await self._get(f"/devices/basicUdiDis/{basic_udi}")
            data = response.json()
            device = self._parse_device(data)
            if device:
                logger.info(
                    "[EUDAMED] Fetched device %s: %s",
                    basic_udi,
                    device.trade_name or device.device_name,
                )
            return device

        except Exception as exc:
            logger.warning("[EUDAMED] fetch_device(%s) failed: %s", basic_udi, exc)
            return None

    # ------------------------------------------------------------------
    # search_devices
    # ------------------------------------------------------------------

    async def search_devices(
        self, query: str, limit: int = 20
    ) -> list[EUDAMEDDevice]:
        """
        Free-text search across EUDAMED device records.

        Args:
            query: Trade name, manufacturer name, or keyword.
            limit: Maximum results (capped at 50 per EUDAMED API constraints).

        Returns:
            List of EUDAMEDDevice records. Empty list on failure.
        """
        if not query or not query.strip():
            return []

        clamped = max(1, min(limit, 50))
        try:
            response = await self._get(
                "/devices/basicUdiDis",
                params={"searchTerm": query.strip(), "pageSize": clamped},
            )
            data = response.json()
            raw_list = data.get("content", data.get("data", []))
            if not isinstance(raw_list, list):
                raw_list = [data] if data else []

            results: list[EUDAMEDDevice] = []
            for entry in raw_list:
                device = self._parse_device(entry)
                if device:
                    results.append(device)

            logger.info(
                "[EUDAMED] search '%s': %d/%d parsed",
                query,
                len(results),
                len(raw_list),
            )
            return results

        except Exception as exc:
            logger.warning("[EUDAMED] search_devices('%s') failed: %s", query, exc)
            return []

    # ------------------------------------------------------------------
    # fetch_registrations
    # ------------------------------------------------------------------

    async def fetch_registrations(self, country: str) -> list[RegistrationRecord]:
        """
        Fetch device registrations available for a given EU/EEA country.

        EUDAMED is an EU-level registry; country filtering is applied via
        countryOfRegistration search parameter.

        Args:
            country: ISO 3166-1 alpha-2 country code (e.g. "DE", "FR").

        Returns:
            List of RegistrationRecord entries.
        """
        eu_countries = {
            "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI",
            "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
            "NL", "PL", "PT", "RO", "SE", "SI", "SK", "NO", "IS", "LI",
        }
        cc = country.upper()
        if cc not in eu_countries:
            logger.debug(
                "[EUDAMED] fetch_registrations: %s not in EU/EEA — returning empty",
                country,
            )
            return []

        try:
            response = await self._get(
                "/devices/basicUdiDis",
                params={"countryOfRegistration": cc, "pageSize": 50},
            )
            data = response.json()
            raw_list = data.get("content", data.get("data", []))
            if not isinstance(raw_list, list):
                raw_list = []

            records: list[RegistrationRecord] = []
            for entry in raw_list:
                basic_udi = self._safe_str(
                    entry.get("basicUdiDi", entry.get("id", ""))
                )
                records.append(
                    RegistrationRecord(
                        registry="EUDAMED",
                        country=cc,
                        device_id=basic_udi,
                        device_name=self._safe_str(entry.get("tradeName", "")),
                        manufacturer=self._safe_str(
                            entry.get("manufacturer", {}).get("name", "")
                            if isinstance(entry.get("manufacturer"), dict)
                            else entry.get("manufacturerName", "")
                        ),
                        status=self._safe_str(entry.get("marketStatus", "")),
                        raw=entry,
                    )
                )

            logger.info(
                "[EUDAMED] fetch_registrations(%s): %d records", cc, len(records)
            )
            return records

        except Exception as exc:
            logger.warning(
                "[EUDAMED] fetch_registrations(%s) failed: %s", country, exc
            )
            return []

    # ------------------------------------------------------------------
    # fetch_field_safety_notices
    # ------------------------------------------------------------------

    async def fetch_field_safety_notices(
        self, days: int = 30
    ) -> list[EUDAMEDFieldSafetyNotice]:
        """
        Retrieve Field Safety Notices published within the last *days* days.

        Endpoint: GET /vigilance/fsns
        Returns up to 50 most-recent FSNs within the date window.

        Args:
            days: How many calendar days back to query (default 30).

        Returns:
            List of EUDAMEDFieldSafetyNotice. Empty list on failure.
        """
        since = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).strftime("%Y-%m-%d")

        try:
            response = await self._get(
                "/vigilance/fsns",
                params={"publishedFrom": since, "pageSize": 50},
            )
            data = response.json()
            raw_list = data.get("content", data.get("data", []))
            if not isinstance(raw_list, list):
                raw_list = []

            notices: list[EUDAMEDFieldSafetyNotice] = []
            for entry in raw_list:
                notice = self._parse_fsn(entry)
                if notice:
                    notices.append(notice)

            logger.info(
                "[EUDAMED] fetch_field_safety_notices(days=%d): %d notices",
                days,
                len(notices),
            )
            return notices

        except Exception as exc:
            logger.warning(
                "[EUDAMED] fetch_field_safety_notices failed: %s", exc
            )
            return []

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_device(self, raw: dict[str, Any]) -> Optional[EUDAMEDDevice]:
        """
        Parse a single EUDAMED API response entry into EUDAMEDDevice.

        EUDAMED nests manufacturer info under "manufacturer" object and
        certificates under "certificates" array.
        """
        try:
            basic_udi = self._safe_str(
                raw.get("basicUdiDi", raw.get("id", ""))
            )
            if not basic_udi:
                return None

            trade_name = self._safe_str(raw.get("tradeName", ""))
            intended_purpose = self._safe_str(raw.get("intendedPurpose", ""))
            device_class = self._safe_str(
                raw.get("deviceClass", raw.get("riskClass", ""))
            )
            regulation = self._safe_str(
                raw.get("regulation", raw.get("regulationType", ""))
            )
            market_status = self._safe_str(raw.get("marketStatus", ""))
            country_of_reg = self._safe_str(
                raw.get("countryOfRegistration", raw.get("country", ""))
            )

            # Manufacturer
            mfr_raw = raw.get("manufacturer", {})
            if isinstance(mfr_raw, dict):
                manufacturer_name = self._safe_str(
                    mfr_raw.get("name", mfr_raw.get("manufacturerName", ""))
                )
                manufacturer_address = self._safe_str(
                    mfr_raw.get("address", "")
                )
                manufacturer_country = self._safe_str(
                    mfr_raw.get("country", mfr_raw.get("countryCode", ""))
                )
                srn = self._safe_str(mfr_raw.get("srn", ""))
            else:
                manufacturer_name = self._safe_str(raw.get("manufacturerName", ""))
                manufacturer_address = ""
                manufacturer_country = ""
                srn = ""

            # Certificates
            certs_raw = self._safe_list(raw.get("certificates", []))
            certificates: list[EUDAMEDCertificate] = []
            primary_cert_number = ""
            nbr_org = ""

            for cert_entry in certs_raw:
                if not isinstance(cert_entry, dict):
                    continue
                nb_raw = cert_entry.get("notifiedBody", {})
                nb_name = ""
                nb_id = ""
                nb_country = ""
                if isinstance(nb_raw, dict):
                    nb_name = self._safe_str(nb_raw.get("name", ""))
                    nb_id = self._safe_str(
                        nb_raw.get("notifiedBodyNumber", nb_raw.get("id", ""))
                    )
                    nb_country = self._safe_str(
                        nb_raw.get("country", nb_raw.get("countryCode", ""))
                    )

                cert = EUDAMEDCertificate(
                    certificate_number=self._safe_str(
                        cert_entry.get("certificateNumber", "")
                    ),
                    certificate_type=self._safe_str(
                        cert_entry.get("certificateType", "")
                    ),
                    notified_body_id=nb_id,
                    notified_body_name=nb_name,
                    country_of_nb=nb_country,
                    valid_from=self._safe_str(cert_entry.get("validFrom", "")),
                    valid_until=self._safe_str(cert_entry.get("validUntil", "")),
                    status=self._safe_str(cert_entry.get("status", "")),
                )
                certificates.append(cert)
                # Track the first active certificate as primary
                if not primary_cert_number and cert.status.lower() in (
                    "active", "valid", ""
                ):
                    primary_cert_number = cert.certificate_number
                    nbr_org = cert.notified_body_name

            # If nothing set from certs, fall back to top-level fields
            if not primary_cert_number:
                primary_cert_number = self._safe_str(
                    raw.get("certificationNumber", "")
                )
            if not nbr_org:
                nbr_org = self._safe_str(raw.get("nbrOrganization", ""))

            legacy_ref = self._safe_str(
                raw.get("legacyMddRef", raw.get("legacyReference", ""))
            )

            return EUDAMEDDevice(
                device_id=basic_udi,
                device_name=trade_name,
                manufacturer=manufacturer_name,
                description=intended_purpose,
                risk_class=device_class,
                basic_udi_di=basic_udi,
                trade_name=trade_name,
                intended_purpose=intended_purpose,
                device_class=device_class,
                regulation=regulation,
                manufacturer_address=manufacturer_address,
                manufacturer_country=manufacturer_country,
                srn=srn,
                certificates=certificates,
                certification_number=primary_cert_number,
                nbr_organization=nbr_org,
                country_of_registration=country_of_reg,
                market_status=market_status,
                legacy_mdd_ref=legacy_ref,
                raw=raw,
                fetched_at=datetime.now(timezone.utc),
            )

        except Exception as exc:
            logger.warning("[EUDAMED] Failed to parse device entry: %s", exc)
            return None

    def _parse_fsn(
        self, raw: dict[str, Any]
    ) -> Optional[EUDAMEDFieldSafetyNotice]:
        """Parse a single FSN entry from the EUDAMED vigilance API."""
        try:
            fsn_id = self._safe_str(raw.get("id", raw.get("fsnId", "")))
            ref_num = self._safe_str(raw.get("referenceNumber", ""))
            title = self._safe_str(raw.get("title", raw.get("name", "")))
            description = self._safe_str(raw.get("description", ""))

            mfr_raw = raw.get("manufacturer", {})
            mfr_name = ""
            mfr_srn = ""
            if isinstance(mfr_raw, dict):
                mfr_name = self._safe_str(mfr_raw.get("name", ""))
                mfr_srn = self._safe_str(mfr_raw.get("srn", ""))
            else:
                mfr_name = self._safe_str(raw.get("manufacturerName", ""))

            affected_raw = self._safe_list(
                raw.get("affectedDevices", raw.get("devices", []))
            )
            affected_devices = [
                self._safe_str(d.get("basicUdiDi", d) if isinstance(d, dict) else d)
                for d in affected_raw
            ]

            countries_raw = self._safe_list(raw.get("countriesAffected", []))
            countries = [
                self._safe_str(c.get("code", c) if isinstance(c, dict) else c)
                for c in countries_raw
            ]

            return EUDAMEDFieldSafetyNotice(
                fsn_id=fsn_id,
                reference_number=ref_num,
                title=title,
                description=description,
                manufacturer_name=mfr_name,
                manufacturer_srn=mfr_srn,
                affected_devices=[d for d in affected_devices if d],
                countries_affected=[c for c in countries if c],
                published_date=self._safe_str(
                    raw.get("publishedDate", raw.get("datePublished", ""))
                ),
                last_updated=self._safe_str(raw.get("lastUpdated", "")),
                severity=self._safe_str(raw.get("severity", "")),
                fsca_type=self._safe_str(raw.get("fscaType", raw.get("type", ""))),
                source_url=self._safe_str(
                    raw.get("url", f"{self.get_source_url()}/fsn/{fsn_id}")
                ),
            )

        except Exception as exc:
            logger.warning("[EUDAMED] Failed to parse FSN entry: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

eudamed_adapter = EUDAMEDAdapter()
