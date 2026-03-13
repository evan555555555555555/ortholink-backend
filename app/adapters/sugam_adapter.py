"""
OrthoLink --- CDSCO SUGAM Adapter (India)

Provides async access to India's Central Drugs Standard Control Organisation
(CDSCO) SUGAM online portal for medical device registration data via:
  - Device registration search (POST /deviceSearch)
  - Device detail lookup by registration number (GET /getDeviceDetails)

API reference: https://cdscomdonline.gov.in/NewMedDev/API
Authentication: None required for public search endpoints.
Rate limit: ~20 requests/minute (CDSCO portal constraints).

Regulatory context:
  - Medical Devices Rules 2017 (MDR 2017) governs all medical devices in India
  - Four risk classes: A (lowest), B, C, D (highest) — defined in Schedule III
  - All devices must be registered with CDSCO before import/manufacture
  - Import Registration (FORM MD-15) and Domestic Registration (FORM MD-13)
  - AI/ML devices: CDSCO AI/ML Guidelines 2026 add new classification pathway
    for Software as a Medical Device (SaMD) using AI/ML components
  - Notified Bodies (Testing Labs) accredited by NABH/NABL handle testing

Usage:
    from app.adapters.sugam_adapter import sugam_adapter
    device = await sugam_adapter.fetch_device("MD-15/2023/45678")
    results = await sugam_adapter.search_devices("pacemaker", limit=10)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from app.adapters.base_adapter import (
    BaseRegistryAdapter,
    DeviceRecord,
    RegistrationRecord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SUGAMDevice(DeviceRecord):
    """
    Medical device registration record from CDSCO SUGAM.

    Covers devices registered under India's Medical Devices Rules 2017,
    including import licences (MD-15) and domestic manufacturing licences (MD-9).
    Includes 2026 AI/ML classification fields per CDSCO AI Guidelines.
    """

    source_registry: str = "SUGAM"
    country: str = "IN"

    # Registration
    registration_number: str = ""   # e.g. "MD-15/2023/45678" or "MD-9/2024/12345"
    registration_type: str = ""     # Import | Domestic Manufacturing | Export
    registration_form: str = ""     # MD-13 (domestic) | MD-15 (import) | MD-14/16 (IVD)

    # Risk classification per MDR 2017 Schedule III
    risk_class: str = ""            # A | B | C | D
    risk_class_basis: str = ""      # Rule reference e.g. "Rule 13, Schedule III, Entry 1"

    # Applicant / registrant
    applicant_name: str = ""        # Company holding the registration
    applicant_address: str = ""
    applicant_state: str = ""       # Indian state

    # Manufacturer details (may differ from applicant for imports)
    manufacturer_country: str = ""
    manufacturer_address: str = ""

    # Import-specific
    import_licence: str = ""        # Import licence number (if applicable)
    port_of_entry: str = ""         # Indian port listed on licence

    # Notified body / testing lab in India
    notified_body_india: str = ""   # NABH/NABL accredited testing laboratory name

    # Validity
    valid_from: str = ""
    valid_until: str = ""
    registration_status: str = ""   # Active | Cancelled | Suspended | Expired

    # AI/ML fields (CDSCO AI/ML Guidelines 2026)
    is_ai_ml_device: bool = False
    ai_ml_classification: str = ""
    # SaMD classification per IEC 62304 + CDSCO 2026 guidelines:
    # Class I (non-serious), Class II (serious), Class III (critical)
    samd_class: str = ""
    ai_algorithm_type: str = ""     # Supervised | Unsupervised | Reinforcement | LLM
    ai_intended_use: str = ""       # Clinical Decision Support | Diagnostic | Therapeutic
    ai_risk_level: str = ""         # Low | Medium | High (per CDSCO 2026 §4.2)

    # Device identifiers
    model_number: str = ""
    catalogue_number: str = ""


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


class SUGAMAdapter(BaseRegistryAdapter[SUGAMDevice]):
    """
    Async adapter for CDSCO SUGAM medical device registration portal.

    Endpoints:
      Search: POST /NewMedDev/API/deviceSearch
      Detail: GET  /NewMedDev/API/getDeviceDetails?registrationNo={reg_no}
    """

    BASE_URL = "https://cdscomdonline.gov.in/NewMedDev/API"
    REGISTRY_NAME = "SUGAM"
    DEFAULT_TIMEOUT = 30.0   # CDSCO portal can be slow
    RATE_LIMIT_RPS = 0.4     # ~20 req/min; portal is resource-constrained
    RATE_LIMIT_BURST = 3

    def get_source_url(self) -> str:
        return "https://cdscomdonline.gov.in"

    def _health_check_path(self) -> str:
        # No lightweight GET endpoint; use a minimal POST search
        # For health check we probe the base URL
        return "/"

    # Override health_check to use a POST probe
    async def health_check(self):  # type: ignore[override]
        """Health check via a minimal device search POST."""
        import time
        from app.adapters.base_adapter import AdapterHealthStatus
        start = time.monotonic()
        try:
            response = await self._post(
                "/deviceSearch",
                json_body={"deviceName": "test", "pageSize": 1},
                timeout=10.0,
            )
            latency_ms = (time.monotonic() - start) * 1000.0
            healthy = 200 <= response.status_code < 400
            logger.info(
                "[SUGAM] Health check %s (%.0fms)",
                "OK" if healthy else f"FAIL({response.status_code})",
                latency_ms,
            )
            return AdapterHealthStatus(
                adapter=self.REGISTRY_NAME,
                healthy=healthy,
                latency_ms=round(latency_ms, 1),
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000.0
            logger.warning("[SUGAM] Health check FAILED: %s", exc)
            return AdapterHealthStatus(
                adapter=self.REGISTRY_NAME,
                healthy=False,
                latency_ms=round(latency_ms, 1),
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # fetch_device
    # ------------------------------------------------------------------

    async def fetch_device(self, registration_no: str) -> Optional[SUGAMDevice]:
        """
        Fetch a device registration detail by its CDSCO registration number.

        Endpoint: GET /getDeviceDetails?registrationNo={reg_no}

        Args:
            registration_no: CDSCO registration number
                             (e.g. "MD-15/2023/45678").

        Returns:
            SUGAMDevice or None if not found.
        """
        if not registration_no or not registration_no.strip():
            logger.warning("[SUGAM] fetch_device called with empty registration_no")
            return None

        reg_no = registration_no.strip()
        try:
            response = await self._get(
                "/getDeviceDetails",
                params={"registrationNo": reg_no},
            )
            data = response.json()

            # API may return single object or list
            if isinstance(data, list):
                raw = data[0] if data else {}
            elif isinstance(data, dict):
                raw = data.get("data", data.get("result", data))
            else:
                return None

            if not raw:
                return None

            device = self._parse_device(raw)
            if device:
                logger.info(
                    "[SUGAM] Fetched device %s: %s",
                    reg_no,
                    device.device_name,
                )
            return device

        except Exception as exc:
            logger.warning("[SUGAM] fetch_device(%s) failed: %s", reg_no, exc)
            return None

    # ------------------------------------------------------------------
    # search_devices
    # ------------------------------------------------------------------

    async def search_devices(
        self, query: str, limit: int = 20
    ) -> list[SUGAMDevice]:
        """
        Search CDSCO device registrations by device name.

        Endpoint: POST /deviceSearch
        Body: {"deviceName": query, "pageSize": limit}

        Args:
            query: Device name or partial name.
            limit: Maximum results (1–50).

        Returns:
            List of SUGAMDevice records. Empty list on failure.
        """
        if not query or not query.strip():
            return []

        clamped = max(1, min(limit, 50))
        try:
            response = await self._post(
                "/deviceSearch",
                json_body={"deviceName": query.strip(), "pageSize": clamped},
            )
            data = response.json()

            raw_list = data if isinstance(data, list) else data.get(
                "data", data.get("results", data.get("items", []))
            )
            if not isinstance(raw_list, list):
                raw_list = []

            results: list[SUGAMDevice] = []
            for entry in raw_list:
                device = self._parse_device(entry)
                if device:
                    results.append(device)

            logger.info("[SUGAM] search '%s': %d results", query, len(results))
            return results

        except Exception as exc:
            logger.warning("[SUGAM] search_devices('%s') failed: %s", query, exc)
            return []

    # ------------------------------------------------------------------
    # fetch_registrations
    # ------------------------------------------------------------------

    async def fetch_registrations(self, country: str) -> list[RegistrationRecord]:
        """
        SUGAM is IN-only. Returns empty for non-IN countries.

        For IN, performs a broad search of active registrations.
        """
        if country.upper() != "IN":
            logger.debug(
                "[SUGAM] fetch_registrations skipped for country=%s (IN only)",
                country,
            )
            return []

        try:
            response = await self._post(
                "/deviceSearch",
                json_body={"status": "Active", "pageSize": 100},
            )
            data = response.json()
            raw_list = data if isinstance(data, list) else data.get("data", [])
            if not isinstance(raw_list, list):
                raw_list = []

            records: list[RegistrationRecord] = []
            for entry in raw_list:
                reg_no = self._safe_str(
                    entry.get("registrationNumber",
                    entry.get("regNo", ""))
                )
                records.append(
                    RegistrationRecord(
                        registry="SUGAM",
                        country="IN",
                        device_id=reg_no,
                        device_name=self._safe_str(
                            entry.get("deviceName",
                            entry.get("productName", ""))
                        ),
                        manufacturer=self._safe_str(
                            entry.get("manufacturerName",
                            entry.get("manufacturer", ""))
                        ),
                        status=self._safe_str(
                            entry.get("status",
                            entry.get("registrationStatus", ""))
                        ),
                        valid_until=self._safe_str(
                            entry.get("validUpto",
                            entry.get("expiryDate", ""))
                        ),
                        raw=entry,
                    )
                )

            logger.info(
                "[SUGAM] fetch_registrations(IN): %d records", len(records)
            )
            return records

        except Exception as exc:
            logger.warning("[SUGAM] fetch_registrations failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_device(self, raw: dict[str, Any]) -> Optional[SUGAMDevice]:
        """
        Parse a single SUGAM API response into SUGAMDevice.

        CDSCO portal returns mixed camelCase and snake_case fields.
        """
        try:
            # Registration number
            registration_number = self._safe_str(
                raw.get("registrationNumber",
                raw.get("regNo",
                raw.get("licenceNumber", "")))
            )
            device_name = self._safe_str(
                raw.get("deviceName",
                raw.get("productName",
                raw.get("name", "")))
            )

            # Risk class — MDR 2017 uses A/B/C/D
            raw_class = self._safe_str(
                raw.get("riskClass",
                raw.get("deviceClass",
                raw.get("class", "")))
            )
            risk_class = self._normalise_risk_class(raw_class)

            manufacturer = self._safe_str(
                raw.get("manufacturerName",
                raw.get("manufacturer", ""))
            )
            manufacturer_country = self._safe_str(
                raw.get("countryOfManufacture",
                raw.get("manufacturerCountry", ""))
            )
            manufacturer_address = self._safe_str(
                raw.get("manufacturerAddress",
                raw.get("mfgAddress", ""))
            )

            applicant_name = self._safe_str(
                raw.get("applicantName",
                raw.get("companyName",
                raw.get("registrantName", "")))
            )
            applicant_address = self._safe_str(
                raw.get("applicantAddress",
                raw.get("address", ""))
            )
            applicant_state = self._safe_str(
                raw.get("state",
                raw.get("applicantState", ""))
            )

            import_licence = self._safe_str(
                raw.get("importLicenceNumber",
                raw.get("importLicence",
                raw.get("importLicNo", "")))
            )

            # AI/ML fields per CDSCO 2026 guidelines
            is_ai_ml = bool(raw.get("isAiMlDevice", raw.get("aiMlFlag", False)))
            ai_classification = self._safe_str(
                raw.get("aiMlClassification",
                raw.get("aiClassification", ""))
            )
            samd_class = self._safe_str(
                raw.get("samdClass",
                raw.get("softwareClass", ""))
            )
            ai_algorithm = self._safe_str(
                raw.get("aiAlgorithmType",
                raw.get("algorithmType", ""))
            )
            ai_intended_use = self._safe_str(
                raw.get("aiIntendedUse",
                raw.get("aiUseCase", ""))
            )
            ai_risk_level = self._safe_str(
                raw.get("aiRiskLevel",
                raw.get("riskLevel", ""))
            )

            # Notified body
            notified_body = self._safe_str(
                raw.get("testingLaboratory",
                raw.get("notifiedBody",
                raw.get("testLab", "")))
            )

            device_id = registration_number or device_name[:30]
            if not device_id:
                return None

            return SUGAMDevice(
                device_id=device_id,
                device_name=device_name,
                manufacturer=manufacturer,
                description=self._safe_str(
                    raw.get("intendedUse",
                    raw.get("indication",
                    raw.get("description", "")))
                ),
                risk_class=risk_class,
                registration_number=registration_number,
                registration_type=self._safe_str(
                    raw.get("registrationType",
                    raw.get("licenceType", ""))
                ),
                registration_form=self._safe_str(
                    raw.get("formNumber",
                    raw.get("form", ""))
                ),
                risk_class_basis=self._safe_str(
                    raw.get("riskClassBasis",
                    raw.get("classificationRule", ""))
                ),
                applicant_name=applicant_name,
                applicant_address=applicant_address,
                applicant_state=applicant_state,
                manufacturer_country=manufacturer_country,
                manufacturer_address=manufacturer_address,
                import_licence=import_licence,
                port_of_entry=self._safe_str(
                    raw.get("portOfEntry",
                    raw.get("port", ""))
                ),
                notified_body_india=notified_body,
                valid_from=self._safe_str(raw.get("validFrom", "")),
                valid_until=self._safe_str(
                    raw.get("validUpto",
                    raw.get("expiryDate", ""))
                ),
                registration_status=self._safe_str(
                    raw.get("status",
                    raw.get("registrationStatus", ""))
                ),
                is_ai_ml_device=is_ai_ml,
                ai_ml_classification=ai_classification,
                samd_class=samd_class,
                ai_algorithm_type=ai_algorithm,
                ai_intended_use=ai_intended_use,
                ai_risk_level=ai_risk_level,
                model_number=self._safe_str(
                    raw.get("modelNumber",
                    raw.get("model", ""))
                ),
                catalogue_number=self._safe_str(
                    raw.get("catalogueNumber",
                    raw.get("catalogNumber", ""))
                ),
                raw=raw,
                fetched_at=datetime.now(timezone.utc),
            )

        except Exception as exc:
            logger.warning("[SUGAM] Failed to parse device entry: %s", exc)
            return None

    @staticmethod
    def _normalise_risk_class(raw_class: str) -> str:
        """
        Normalise a raw class string to MDR 2017 canonical form: A, B, C, or D.

        Handles common variations including numeric (1-4), full names, and
        mixed capitalisation.
        """
        if not raw_class:
            return ""
        cleaned = raw_class.strip().upper().replace("CLASS ", "").replace("CLASSE ", "")
        mapping = {
            "A": "A", "1": "A",
            "B": "B", "2": "B",
            "C": "C", "3": "C",
            "D": "D", "4": "D",
        }
        return mapping.get(cleaned, raw_class.strip())


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

sugam_adapter = SUGAMAdapter()
