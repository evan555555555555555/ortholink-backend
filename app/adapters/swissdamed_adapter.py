"""
OrthoLink --- Swissdamed Adapter (Switzerland)

Provides async access to Swissmedic's Swissdamed device registry via the
FHIR R4 M2M API for:
  - Device lookup by UDI-DI identifier
  - DeviceDefinition search by manufacturer name
  - Legacy MDD/AIMDD certificate mapping to MDR equivalents

API reference: https://www.swissmedic.ch/swissdamed/api/fhir/R4/metadata
Authentication: None required for public Device/DeviceDefinition queries.
Rate limit: ~30 requests/minute (Swissmedic guidance).

Regulatory context:
  - Switzerland harmonised with EU MDR 2017/745 via MedDO (SR 812.213)
  - MedDO applies equivalent MDR classes (I, IIa, IIb, III)
  - Legacy MDD/AIMDD devices may remain until May 26 2024 grace period ends
  - Swissmedic number format: XXXXXX (6-digit product authorisation)
  - Notified Body approval required for Classes IIa, IIb, III

Usage:
    from app.adapters.swissdamed_adapter import swissdamed_adapter
    device = await swissdamed_adapter.fetch_device("7612345678901")
    legacy = await swissdamed_adapter.map_legacy_certificate("0483-MDD-2020-12345")
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from app.adapters.base_adapter import (
    BaseRegistryAdapter,
    DeviceRecord,
    RegistrationRecord,
)
from app.services.fhir_connector import FHIRConnector, get_fhir_connector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SwissdamedDevice(DeviceRecord):
    """
    Medical device record from Swissmedic Swissdamed FHIR R4 API.

    Combines FHIR standard fields with Switzerland-specific regulatory data
    from Swissmedic extensions.
    """

    source_registry: str = "SWISSDAMED"
    country: str = "CH"

    swissmedic_number: str = ""      # 6-digit Swissmedic product authorisation number
    device_name: str = ""
    model_number: str = ""
    catalog_number: str = ""

    # Classification (MedDO / MDR-aligned)
    mdr_class: str = ""              # I | IIa | IIb | III
    ivdr_class: str = ""             # A | B | C | D (if IVD)
    regulation_ref: str = ""         # e.g. "MedDO Art. 17" or "MDR 2017/745"

    # Notified Body & certificates
    notified_body: str = ""          # NB name
    certificate_number: str = ""     # Active certificate reference
    certificate_valid_until: str = ""

    # Legacy MDD/AIMDD migration
    legacy_mdd_ref: str = ""         # Original MDD/AIMDD certificate ref
    legacy_directive: str = ""       # MDD | AIMDD | IVDD
    legacy_transition_status: str = "" # compliant | pending | expired

    # FHIR-derived fields
    fhir_resource_id: str = ""
    udi_di: str = ""
    udi_issuer: str = ""
    status: str = ""                 # FHIR Device.status


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


class SwissdamedAdapter(BaseRegistryAdapter[SwissdamedDevice]):
    """
    Async adapter for the Swissmedic Swissdamed FHIR R4 API.

    Uses FHIRConnector for parsing FHIR Device and DeviceDefinition resources.
    Falls back gracefully when resources are not found or the server is
    temporarily unavailable.
    """

    BASE_URL = "https://www.swissmedic.ch/swissdamed/api/fhir/R4"
    REGISTRY_NAME = "SWISSDAMED"
    DEFAULT_TIMEOUT = 20.0
    RATE_LIMIT_RPS = 0.5    # 30 req/min per Swissmedic guidance
    RATE_LIMIT_BURST = 5

    def __init__(self) -> None:
        super().__init__()
        self._fhir: FHIRConnector = get_fhir_connector()
        # FHIR servers require Accept: application/fhir+json
        self._client_headers["Accept"] = "application/fhir+json"

    def get_source_url(self) -> str:
        return "https://www.swissmedic.ch/swissdamed"

    def _health_check_path(self) -> str:
        return "/metadata"

    # ------------------------------------------------------------------
    # fetch_device
    # ------------------------------------------------------------------

    async def fetch_device(self, udi_di: str) -> Optional[SwissdamedDevice]:
        """
        Look up a device by its UDI-DI using FHIR Device?identifier search.

        Args:
            udi_di: UDI Device Identifier string.

        Returns:
            SwissdamedDevice or None if not found.
        """
        if not udi_di or not udi_di.strip():
            logger.warning("[SWISSDAMED] fetch_device called with empty udi_di")
            return None

        identifier = udi_di.strip()
        try:
            response = await self._get(
                "/Device",
                params={"identifier": identifier, "_format": "json"},
            )
            data = response.json()

            # Response is a FHIR Bundle
            bundle = self._fhir.parse_bundle(data, country_hint="CH")
            if not bundle.devices:
                logger.info("[SWISSDAMED] No device found for UDI-DI: %s", identifier)
                return None

            fhir_device = bundle.devices[0]
            device = self._map_fhir_to_swissdamed(fhir_device, raw=data)
            logger.info(
                "[SWISSDAMED] Fetched device %s: %s",
                identifier,
                device.device_name,
            )
            return device

        except Exception as exc:
            logger.warning(
                "[SWISSDAMED] fetch_device(%s) failed: %s", identifier, exc
            )
            return None

    # ------------------------------------------------------------------
    # search_devices
    # ------------------------------------------------------------------

    async def search_devices(
        self, query: str, limit: int = 20
    ) -> list[SwissdamedDevice]:
        """
        Search DeviceDefinition resources by manufacturer name or device name.

        Uses FHIR DeviceDefinition?manufacturer search parameter. Swissdamed
        indexes device definitions by manufacturer name.

        Args:
            query: Manufacturer or device name to search for.
            limit: Maximum results.

        Returns:
            List of SwissdamedDevice records. Empty list on failure.
        """
        if not query or not query.strip():
            return []

        clamped = max(1, min(limit, 50))
        try:
            response = await self._get(
                "/DeviceDefinition",
                params={
                    "manufacturer": query.strip(),
                    "_count": clamped,
                    "_format": "json",
                },
            )
            data = response.json()
            bundle = self._fhir.parse_bundle(data, country_hint="CH")

            results: list[SwissdamedDevice] = []
            for fhir_device in bundle.devices:
                device = self._map_fhir_to_swissdamed(fhir_device, raw={})
                if device:
                    results.append(device)

            logger.info(
                "[SWISSDAMED] search '%s': %d devices", query, len(results)
            )
            return results

        except Exception as exc:
            logger.warning(
                "[SWISSDAMED] search_devices('%s') failed: %s", query, exc
            )
            return []

    # ------------------------------------------------------------------
    # fetch_registrations
    # ------------------------------------------------------------------

    async def fetch_registrations(self, country: str) -> list[RegistrationRecord]:
        """
        Swissdamed is CH-only. Returns empty list for non-CH countries.

        For CH, performs a broad Device search and maps to RegistrationRecord.
        """
        if country.upper() != "CH":
            logger.debug(
                "[SWISSDAMED] fetch_registrations skipped for country=%s (CH only)",
                country,
            )
            return []

        try:
            response = await self._get(
                "/Device",
                params={"status": "active", "_count": 100, "_format": "json"},
            )
            data = response.json()
            bundle = self._fhir.parse_bundle(data, country_hint="CH")

            records: list[RegistrationRecord] = []
            for fhir_device in bundle.devices:
                records.append(
                    RegistrationRecord(
                        registry="SWISSDAMED",
                        country="CH",
                        device_id=fhir_device.udi_carrier or fhir_device.resource_id,
                        device_name=fhir_device.device_name,
                        manufacturer=fhir_device.manufacturer,
                        status=fhir_device.status,
                        raw={
                            "resource_id": fhir_device.resource_id,
                            "swissmedic_listing": fhir_device.swissmedic_listing_number,
                            "swissmedic_auth": fhir_device.swissmedic_authorization,
                        },
                    )
                )

            logger.info(
                "[SWISSDAMED] fetch_registrations(CH): %d records", len(records)
            )
            return records

        except Exception as exc:
            logger.warning("[SWISSDAMED] fetch_registrations failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # map_legacy_certificate
    # ------------------------------------------------------------------

    async def map_legacy_certificate(
        self, mdd_cert: str
    ) -> Optional[SwissdamedDevice]:
        """
        Map an old MDD/AIMDD/IVDD certificate reference to the corresponding
        Swissdamed (MDR-aligned) record.

        FHIR query: GET /Device?identifier={mdd_cert}
        The Swissmedic FHIR server indexes legacy identifiers under the
        identifier element with system "http://swissmedic.ch/ns/legacy-cert".

        Args:
            mdd_cert: Legacy certificate number (e.g. "0483-MDD-2020-12345").

        Returns:
            SwissdamedDevice with legacy_mdd_ref populated, or None if no
            matching record exists.
        """
        if not mdd_cert or not mdd_cert.strip():
            logger.warning("[SWISSDAMED] map_legacy_certificate: empty cert ref")
            return None

        cert_ref = mdd_cert.strip()
        try:
            response = await self._get(
                "/Device",
                params={
                    "identifier": f"http://swissmedic.ch/ns/legacy-cert|{cert_ref}",
                    "_format": "json",
                },
            )
            data = response.json()
            bundle = self._fhir.parse_bundle(data, country_hint="CH")
            if not bundle.devices:
                logger.info(
                    "[SWISSDAMED] No MDR record found for legacy cert: %s", cert_ref
                )
                return None

            fhir_device = bundle.devices[0]
            device = self._map_fhir_to_swissdamed(fhir_device, raw=data)
            if device:
                # Annotate with legacy ref context
                device.legacy_mdd_ref = cert_ref
                # Determine directive type from cert format heuristics
                upper = cert_ref.upper()
                if "MDD" in upper:
                    device.legacy_directive = "MDD"
                elif "AIMDD" in upper:
                    device.legacy_directive = "AIMDD"
                elif "IVDD" in upper:
                    device.legacy_directive = "IVDD"
                device.legacy_transition_status = "compliant"
                logger.info(
                    "[SWISSDAMED] Mapped legacy cert %s → device %s",
                    cert_ref,
                    device.swissmedic_number or device.device_id,
                )
            return device

        except Exception as exc:
            logger.warning(
                "[SWISSDAMED] map_legacy_certificate(%s) failed: %s", cert_ref, exc
            )
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_fhir_to_swissdamed(
        self,
        fhir_device: Any,
        raw: dict[str, Any],
    ) -> Optional[SwissdamedDevice]:
        """
        Convert a parsed FHIRDevice into a SwissdamedDevice.

        Extracts Swissmedic-specific extensions (listing number, authorization,
        MedDO classification) from the FHIRDevice fields.
        """
        try:
            # Primary device ID — prefer UDI-DI over resource ID
            device_id = fhir_device.udi_carrier or fhir_device.resource_id
            device_name = fhir_device.device_name or ""

            # Classification from FHIR classification element
            mdr_class = ""
            ivdr_class = ""
            regulation_ref = ""
            if fhir_device.classification:
                raw_code = fhir_device.classification.code.upper()
                sys = fhir_device.classification.system.lower()
                if any(cls in raw_code for cls in ("IIA", "IIB", "III", "I")):
                    mdr_class = raw_code
                elif any(ltr in raw_code for ltr in ("A", "B", "C", "D")) and len(raw_code) == 1:
                    ivdr_class = raw_code
                if "meddo" in sys or "swissmedic" in sys:
                    regulation_ref = "MedDO SR 812.213"
                elif "mdr" in sys or "eu" in sys:
                    regulation_ref = "MDR 2017/745"

            # Swissmedic-specific extension values from FHIR connector
            swissmedic_number = fhir_device.swissmedic_listing_number
            certificate_number = fhir_device.swissmedic_authorization

            # Extract cert validity from raw_extensions if present
            cert_valid_until = ""
            for ext_url, ext_val in fhir_device.raw_extensions.items():
                if "valid" in ext_url.lower() and isinstance(ext_val, str):
                    cert_valid_until = ext_val
                    break

            # Notified body — look for Swissmedic-specific extension or identifier
            notified_body = ""
            for ident in fhir_device.identifiers:
                if "notified" in ident.system.lower() or "nb" in ident.system.lower():
                    notified_body = ident.value
                    break

            return SwissdamedDevice(
                device_id=device_id,
                device_name=device_name,
                manufacturer=fhir_device.manufacturer,
                description=fhir_device.description,
                risk_class=mdr_class or ivdr_class,
                swissmedic_number=swissmedic_number,
                model_number=fhir_device.model_number,
                catalog_number=fhir_device.catalog_number,
                mdr_class=mdr_class,
                ivdr_class=ivdr_class,
                regulation_ref=regulation_ref,
                notified_body=notified_body,
                certificate_number=certificate_number,
                certificate_valid_until=cert_valid_until,
                fhir_resource_id=fhir_device.resource_id,
                udi_di=fhir_device.udi_carrier,
                udi_issuer=fhir_device.udi_issuer,
                status=fhir_device.status,
                raw=raw,
                fetched_at=datetime.now(timezone.utc),
            )

        except Exception as exc:
            logger.warning("[SWISSDAMED] _map_fhir_to_swissdamed failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

swissdamed_adapter = SwissdamedAdapter()
