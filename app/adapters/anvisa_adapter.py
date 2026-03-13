"""
OrthoLink --- ANVISA SIUD Adapter (Brazil)

Provides async access to Brazil's ANVISA Sistema de Identificacao Unica de
Dispositivos (SIUD) via:
  - FHIR R4 M2M API (Normative Instruction 426/2026)
  - Legacy ANVISA consultas REST API for registration search

API references:
  - FHIR: https://consultas.anvisa.gov.br/api/fhir/R4/metadata
  - Consultas: https://consultas.anvisa.gov.br/api/consulta/dispositivos
  - IN 426/2026: https://www.in.gov.br/en/web/dou/-/instrucao-normativa-anvisa-n-426-de-...

Authentication: None required for public read endpoints.
Rate limit: ~20 requests/minute (ANVISA guidance for non-authenticated access).

Regulatory context:
  - RDC 751/2022 defines 4 risk classes: I (lowest) through IV (highest)
  - UDI mandatory for Class III and IV devices from 2024; Class II from 2025
  - All devices must have ANVISA registration (Registro/Cadastro) before sale
  - Importers must hold an Autorização de Funcionamento de Empresa (AFE)

Usage:
    from app.adapters.anvisa_adapter import anvisa_adapter
    device = await anvisa_adapter.fetch_device("7898971030149")
    results = await anvisa_adapter.search_devices("cateter", limit=10)
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


class ANVISADevice(DeviceRecord):
    """
    Medical device registration record from ANVISA SIUD.

    Combines FHIR R4 Device resource fields with Brazil-specific ANVISA
    regulatory attributes from IN 426/2026 extensions.
    """

    source_registry: str = "ANVISA"
    country: str = "BR"

    # ANVISA registration
    registration_number: str = ""    # Registro (e.g. "80255870001") or Cadastro number
    registration_type: str = ""      # Registro | Cadastro | Notificacao
    anvisa_enquadramento: str = ""   # Product category / enquadramento

    # Risk class per RDC 751/2022
    risk_class: str = ""             # I | II | III | IV
    risk_class_rationale: str = ""

    # Manufacturer and importer
    manufacturer_country: str = ""
    importer: str = ""               # Brazilian importer company (if applicable)
    importer_cnpj: str = ""          # Brazilian company registration number
    importer_afe: str = ""           # Autorização de Funcionamento de Empresa

    # UDI fields (IN 426/2026)
    udi_di: str = ""
    udi_issuer: str = ""             # GS1 | HIBCC | ICCBBA | IFA
    udi_mandatory_date: str = ""     # When UDI became/becomes mandatory for this class

    # Registration validity
    valid_from: str = ""
    valid_until: str = ""
    registration_status: str = ""    # Vigente | Cancelado | Suspenso | Vencido

    # FHIR provenance
    fhir_resource_id: str = ""


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


class ANVISAAdapter(BaseRegistryAdapter[ANVISADevice]):
    """
    Async adapter for ANVISA SIUD — FHIR R4 + legacy consultas API.

    Routes:
      FHIR: GET /api/fhir/R4/Device?identifier={udi_di}
      Legacy: GET /api/consulta/dispositivos?nome={query}&tamanhoPagina={limit}
    """

    # FHIR endpoint (IN 426/2026)
    BASE_URL = "https://consultas.anvisa.gov.br/api/fhir/R4"

    # Legacy consultas endpoint (registration number / name search)
    _CONSULTAS_BASE = "https://consultas.anvisa.gov.br/api"

    REGISTRY_NAME = "ANVISA"
    DEFAULT_TIMEOUT = 25.0
    RATE_LIMIT_RPS = 0.4     # ~20 req/min per ANVISA guidance
    RATE_LIMIT_BURST = 4

    def __init__(self) -> None:
        super().__init__()
        self._fhir: FHIRConnector = get_fhir_connector()
        # FHIR servers accept application/fhir+json
        self._client_headers["Accept"] = "application/fhir+json"

    def get_source_url(self) -> str:
        return "https://consultas.anvisa.gov.br"

    def _health_check_path(self) -> str:
        return "/metadata"

    # ------------------------------------------------------------------
    # fetch_device
    # ------------------------------------------------------------------

    async def fetch_device(self, udi_di: str) -> Optional[ANVISADevice]:
        """
        Look up a device by its UDI-DI via ANVISA SIUD FHIR R4 API.

        Falls back to the legacy consultas API if FHIR returns no results.

        Args:
            udi_di: UDI Device Identifier (GS1/HIBCC format).

        Returns:
            ANVISADevice or None.
        """
        if not udi_di or not udi_di.strip():
            logger.warning("[ANVISA] fetch_device called with empty udi_di")
            return None

        identifier = udi_di.strip()
        try:
            response = await self._get(
                "/Device",
                params={"identifier": identifier, "_format": "json"},
            )
            data = response.json()
            bundle = self._fhir.parse_bundle(data, country_hint="BR")

            if bundle.devices:
                fhir_device = bundle.devices[0]
                device = self._map_fhir_to_anvisa(fhir_device, raw=data)
                if device:
                    logger.info(
                        "[ANVISA] Fetched device %s: %s",
                        identifier,
                        device.device_name,
                    )
                    return device

        except Exception as exc:
            logger.warning(
                "[ANVISA] FHIR fetch_device(%s) failed: %s — trying legacy",
                identifier,
                exc,
            )

        # Fallback: try legacy registration search
        return await self._fetch_via_legacy(identifier)

    # ------------------------------------------------------------------
    # search_devices
    # ------------------------------------------------------------------

    async def search_devices(
        self, query: str, limit: int = 20
    ) -> list[ANVISADevice]:
        """
        Search ANVISA device registrations by name via legacy consultas API.

        Endpoint: GET /api/consulta/dispositivos?nome={query}&tamanhoPagina={limit}

        Args:
            query: Device name or partial name (Portuguese accepted).
            limit: Maximum results (1–50).

        Returns:
            List of ANVISADevice records. Empty list on failure.
        """
        if not query or not query.strip():
            return []

        clamped = max(1, min(limit, 50))
        try:
            response = await self._get(
                f"{self._CONSULTAS_BASE}/consulta/dispositivos",
                params={"nome": query.strip(), "tamanhoPagina": clamped},
            )
            data = response.json()

            # Consultas API returns list directly or under "content"/"data"
            raw_list = data if isinstance(data, list) else data.get(
                "content", data.get("data", data.get("itens", []))
            )
            if not isinstance(raw_list, list):
                raw_list = []

            results: list[ANVISADevice] = []
            for entry in raw_list:
                device = self._parse_legacy_entry(entry)
                if device:
                    results.append(device)

            logger.info(
                "[ANVISA] search '%s': %d results", query, len(results)
            )
            return results

        except Exception as exc:
            logger.warning("[ANVISA] search_devices('%s') failed: %s", query, exc)
            return []

    # ------------------------------------------------------------------
    # fetch_registrations
    # ------------------------------------------------------------------

    async def fetch_registrations(self, country: str) -> list[RegistrationRecord]:
        """
        ANVISA is BR-only. Returns empty list for non-BR countries.

        For BR, performs a general registration search and maps results.
        """
        if country.upper() != "BR":
            logger.debug(
                "[ANVISA] fetch_registrations skipped for country=%s (BR only)",
                country,
            )
            return []

        try:
            response = await self._get(
                "/Device",
                params={"status": "active", "_count": 100, "_format": "json"},
            )
            data = response.json()
            bundle = self._fhir.parse_bundle(data, country_hint="BR")

            records: list[RegistrationRecord] = []
            for fhir_device in bundle.devices:
                records.append(
                    RegistrationRecord(
                        registry="ANVISA",
                        country="BR",
                        device_id=fhir_device.anvisa_registro or fhir_device.resource_id,
                        device_name=fhir_device.device_name,
                        manufacturer=fhir_device.manufacturer,
                        status=fhir_device.status,
                        raw={
                            "resource_id": fhir_device.resource_id,
                            "anvisa_registro": fhir_device.anvisa_registro,
                            "classe_risco": fhir_device.anvisa_classe_risco,
                        },
                    )
                )

            logger.info(
                "[ANVISA] fetch_registrations(BR): %d records", len(records)
            )
            return records

        except Exception as exc:
            logger.warning("[ANVISA] fetch_registrations failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_via_legacy(self, identifier: str) -> Optional[ANVISADevice]:
        """
        Try fetching a device via the legacy ANVISA consultas API.
        Used as fallback when FHIR lookup returns no results.
        """
        try:
            response = await self._get(
                f"{self._CONSULTAS_BASE}/consulta/dispositivos",
                params={"nome": identifier, "tamanhoPagina": 1},
            )
            data = response.json()
            raw_list = data if isinstance(data, list) else data.get(
                "content", data.get("itens", [])
            )
            if isinstance(raw_list, list) and raw_list:
                return self._parse_legacy_entry(raw_list[0])
        except Exception as exc:
            logger.debug("[ANVISA] Legacy fallback for %s failed: %s", identifier, exc)
        return None

    def _map_fhir_to_anvisa(
        self,
        fhir_device: Any,
        raw: dict[str, Any],
    ) -> Optional[ANVISADevice]:
        """Map a parsed FHIRDevice into an ANVISADevice."""
        try:
            device_id = fhir_device.anvisa_registro or fhir_device.udi_carrier or fhir_device.resource_id
            risk_class = fhir_device.anvisa_classe_risco or ""
            # Normalise to I/II/III/IV per RDC 751/2022
            if risk_class.upper() in ("CLASS I", "CLASSE I", "1"):
                risk_class = "I"
            elif risk_class.upper() in ("CLASS II", "CLASSE II", "2"):
                risk_class = "II"
            elif risk_class.upper() in ("CLASS III", "CLASSE III", "3"):
                risk_class = "III"
            elif risk_class.upper() in ("CLASS IV", "CLASSE IV", "4"):
                risk_class = "IV"

            return ANVISADevice(
                device_id=device_id,
                device_name=fhir_device.device_name,
                manufacturer=fhir_device.manufacturer,
                description=fhir_device.description,
                risk_class=risk_class,
                registration_number=fhir_device.anvisa_registro,
                anvisa_enquadramento=fhir_device.anvisa_enquadramento,
                udi_di=fhir_device.udi_carrier,
                udi_issuer=fhir_device.udi_issuer,
                registration_status=fhir_device.status,
                fhir_resource_id=fhir_device.resource_id,
                raw=raw,
                fetched_at=datetime.now(timezone.utc),
            )

        except Exception as exc:
            logger.warning("[ANVISA] _map_fhir_to_anvisa failed: %s", exc)
            return None

    def _parse_legacy_entry(self, entry: dict[str, Any]) -> Optional[ANVISADevice]:
        """
        Parse a single record from the ANVISA legacy consultas API.

        The consultas API returns a flat JSON dict per device with
        Portuguese-language field names.
        """
        try:
            # Multiple field name variants across ANVISA API versions
            registration_number = self._safe_str(
                entry.get("numeroRegistro",
                entry.get("numRegistro",
                entry.get("registro", "")))
            )
            device_name = self._safe_str(
                entry.get("nomeProduto",
                entry.get("nome",
                entry.get("nomeDispositivo", "")))
            )
            risk_class_raw = self._safe_str(
                entry.get("classeRisco",
                entry.get("classe",
                entry.get("riskClass", "")))
            )
            # Normalise risk class
            rc_map = {
                "I": "I", "CLASSE I": "I", "1": "I",
                "II": "II", "CLASSE II": "II", "2": "II",
                "III": "III", "CLASSE III": "III", "3": "III",
                "IV": "IV", "CLASSE IV": "IV", "4": "IV",
            }
            risk_class = rc_map.get(risk_class_raw.upper(), risk_class_raw)

            manufacturer = self._safe_str(
                entry.get("fabricante",
                entry.get("nomeEmpresa",
                entry.get("manufacturer", "")))
            )
            importer = self._safe_str(
                entry.get("importador",
                entry.get("nomeImportador", ""))
            )
            valid_until = self._safe_str(
                entry.get("vencimento",
                entry.get("dataVencimento",
                entry.get("validUntil", "")))
            )
            registration_status = self._safe_str(
                entry.get("situacao",
                entry.get("status", ""))
            )
            enquadramento = self._safe_str(
                entry.get("enquadramento",
                entry.get("categoriaProduto", ""))
            )
            manufacturer_country = self._safe_str(
                entry.get("paisFabricante",
                entry.get("countryOfManufacture", ""))
            )

            device_id = registration_number or device_name[:30]
            if not device_id:
                return None

            return ANVISADevice(
                device_id=device_id,
                device_name=device_name,
                manufacturer=manufacturer,
                description=enquadramento,
                risk_class=risk_class,
                registration_number=registration_number,
                registration_type=self._safe_str(
                    entry.get("tipoRegistro", entry.get("type", ""))
                ),
                anvisa_enquadramento=enquadramento,
                manufacturer_country=manufacturer_country,
                importer=importer,
                importer_cnpj=self._safe_str(entry.get("cnpjImportador", "")),
                valid_until=valid_until,
                registration_status=registration_status,
                raw=entry,
                fetched_at=datetime.now(timezone.utc),
            )

        except Exception as exc:
            logger.warning("[ANVISA] _parse_legacy_entry failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

anvisa_adapter = ANVISAAdapter()
