"""
OrthoLink Global Registry Endpoints
Router prefix: /registries

Unified REST API for querying international medical device registries:
  GUDID (US), EUDAMED (EU), Swissdamed (CH), ANVISA (BR), ARTG (AU),
  MDALL (CA), SUGAM (IN), GMDN (nomenclature).

Endpoints:
  GET  /registries/search                   — unified cross-registry search
  GET  /registries/{country}/devices        — single country device search
  GET  /registries/{country}/device/{id}    — device lookup by UDI-DI or reg number
  GET  /registries/gmdn/{code}              — GMDN nomenclature lookup
  GET  /registries/udi/lookup               — UDI-DI lookup across all databases
  POST /registries/legacy/map               — map MDD/AIMDD cert to MDR
  GET  /registries/health                   — adapter connectivity health check
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.middleware.auth import AuthenticatedUser, get_current_user
from app.services.crypto_signer import sign_payload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/registries", tags=["Registries"])

# ─────────────────────────────────────────────────────────────────────────────
# Registry adapter imports — graceful degradation if modules not yet built
# ─────────────────────────────────────────────────────────────────────────────

try:
    from app.adapters.gudid_adapter import GUDIDAdapter
except ImportError:
    GUDIDAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.eudamed_adapter import EUDAMEDAdapter
except ImportError:
    EUDAMEDAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.swissdamed_adapter import SwissdamedAdapter
except ImportError:
    SwissdamedAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.anvisa_adapter import ANVISAAdapter
except ImportError:
    ANVISAAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.artg_adapter import ARTGAdapter
except ImportError:
    ARTGAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.mdall_adapter import MDALLAdapter
except ImportError:
    MDALLAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.sugam_adapter import SUGAMAdapter
except ImportError:
    SUGAMAdapter = None  # type: ignore[assignment, misc]

try:
    from app.adapters.gmdn_adapter import GMDNAdapter
except ImportError:
    GMDNAdapter = None  # type: ignore[assignment, misc]

try:
    from app.services.legacy_mapper import LegacyMapper
except ImportError:
    LegacyMapper = None  # type: ignore[assignment, misc]


# ─────────────────────────────────────────────────────────────────────────────
# Country → adapter mapping
# ─────────────────────────────────────────────────────────────────────────────

_COUNTRY_ADAPTER_MAP: dict[str, tuple[str, Any]] = {}


def _build_adapter_map() -> dict[str, tuple[str, Any]]:
    """Build country-code to (label, adapter_class) mapping at import time."""
    mapping: dict[str, tuple[str, Any]] = {}
    if GUDIDAdapter is not None:
        mapping["US"] = ("GUDID", GUDIDAdapter)
    if EUDAMEDAdapter is not None:
        mapping["EU"] = ("EUDAMED", EUDAMEDAdapter)
    if SwissdamedAdapter is not None:
        mapping["CH"] = ("Swissdamed", SwissdamedAdapter)
    if ANVISAAdapter is not None:
        mapping["BR"] = ("ANVISA", ANVISAAdapter)
    if ARTGAdapter is not None:
        mapping["AU"] = ("ARTG", ARTGAdapter)
    if MDALLAdapter is not None:
        mapping["CA"] = ("MDALL", MDALLAdapter)
    if SUGAMAdapter is not None:
        mapping["IN"] = ("SUGAM", SUGAMAdapter)
    return mapping


_COUNTRY_ADAPTER_MAP = _build_adapter_map()


def _get_adapter(country: str) -> Any:
    """Instantiate and return the adapter for a given country code."""
    entry = _COUNTRY_ADAPTER_MAP.get(country.upper())
    if entry is None:
        return None
    _, adapter_cls = entry
    return adapter_cls()


# ─────────────────────────────────────────────────────────────────────────────
# Request / response schemas
# ─────────────────────────────────────────────────────────────────────────────

class DeviceResult(BaseModel):
    """A single device record returned from a registry search."""
    device_id: str = Field(..., description="UDI-DI, registration number, or internal ID")
    device_name: str = ""
    manufacturer: str = ""
    country: str = ""
    registry: str = ""
    device_class: str = ""
    status: str = ""
    udi_di: Optional[str] = None
    gmdn_code: Optional[str] = None
    gmdn_term: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class RegistrySearchResponse(BaseModel):
    """Response wrapper for multi-registry search."""
    query: str
    total_results: int = 0
    countries_searched: list[str] = Field(default_factory=list)
    results_by_country: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)


class GMDNResult(BaseModel):
    """GMDN nomenclature lookup result."""
    code: str
    term: str = ""
    definition: str = ""
    country_mappings: dict[str, list[str]] = Field(default_factory=dict)


class LegacyMapRequest(BaseModel):
    """Request body for legacy MDD/AIMDD certificate mapping."""
    cert_number: str = Field(..., description="Legacy MDD or AIMDD certificate number")
    notified_body: str = Field(..., description="Notified Body that issued the certificate")


class LegacyMapResult(BaseModel):
    """Result of legacy certificate-to-MDR mapping."""
    cert_number: str
    notified_body: str
    mdr_status: str = ""
    mdr_reference: Optional[str] = None
    transition_deadline: Optional[str] = None
    device_classes: list[str] = Field(default_factory=list)
    notes: str = ""


class UDILookupResponse(BaseModel):
    """Response for UDI-DI cross-database lookup."""
    udi_di: str
    found_in: list[str] = Field(default_factory=list)
    results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    total_matches: int = 0


class AdapterHealthStatus(BaseModel):
    """Health status for a single registry adapter."""
    registry: str
    country: str
    available: bool
    status: str = ""
    latency_ms: Optional[float] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run adapter query in thread (adapters may use blocking I/O)
# ─────────────────────────────────────────────────────────────────────────────

async def _query_adapter(
    country: str,
    registry_name: str,
    adapter: Any,
    query: str,
    limit: int,
) -> tuple[str, list[dict[str, Any]], Optional[str]]:
    """
    Run a single adapter search in a thread and return (country, results, error).
    Never raises; captures exceptions as error strings.
    """
    try:
        if asyncio.iscoroutinefunction(getattr(adapter, "search", None)):
            results = await adapter.search(query=query, limit=limit)
        else:
            results = await asyncio.to_thread(adapter.search, query=query, limit=limit)

        # Normalize results to list of dicts
        normalized: list[dict[str, Any]] = []
        for item in results or []:
            if hasattr(item, "model_dump"):
                normalized.append(item.model_dump())
            elif isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({"raw": str(item)})

        # Tag each result with country and registry
        for entry in normalized:
            entry.setdefault("country", country)
            entry.setdefault("registry", registry_name)

        return country, normalized, None
    except Exception as e:
        logger.warning("Registry adapter %s (%s) query failed: %s", registry_name, country, e)
        return country, [], str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/search", response_model=RegistrySearchResponse)
async def unified_search(
    query: str = Query(..., min_length=1, max_length=500, description="Device name, keyword, or UDI-DI"),
    countries: Optional[str] = Query(
        None,
        description="Comma-separated country codes to search (e.g. 'US,EU,AU'). Omit for all.",
    ),
    limit: int = Query(20, ge=1, le=100, description="Max results per registry"),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Unified search across all configured device registries.

    Queries GUDID, EUDAMED, Swissdamed, ANVISA, ARTG, MDALL, SUGAM
    concurrently via asyncio.gather(). Results grouped by country code.
    """
    if not _COUNTRY_ADAPTER_MAP:
        raise HTTPException(
            status_code=503,
            detail="No registry adapters are configured. Install adapter modules to enable registry search.",
        )

    # Determine which countries to search
    if countries:
        target_countries = [c.strip().upper() for c in countries.split(",") if c.strip()]
        unknown = [c for c in target_countries if c not in _COUNTRY_ADAPTER_MAP]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported country codes: {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(_COUNTRY_ADAPTER_MAP.keys()))}",
            )
    else:
        target_countries = list(_COUNTRY_ADAPTER_MAP.keys())

    # Fire parallel queries
    tasks = []
    for cc in target_countries:
        registry_name, adapter_cls = _COUNTRY_ADAPTER_MAP[cc]
        adapter = adapter_cls()
        tasks.append(_query_adapter(cc, registry_name, adapter, query, limit))

    gathered = await asyncio.gather(*tasks)

    results_by_country: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    total = 0

    for country_code, results, error in gathered:
        if error:
            errors[country_code] = error
        if results:
            results_by_country[country_code] = results
            total += len(results)

    response = RegistrySearchResponse(
        query=query,
        total_results=total,
        countries_searched=target_countries,
        results_by_country=results_by_country,
        errors=errors,
    )

    try:
        signed = sign_payload(response.model_dump())
        return JSONResponse(content=signed)
    except Exception:
        return response


@router.get("/{country}/devices")
async def search_country_devices(
    country: str,
    query: str = Query(..., min_length=1, max_length=500, description="Search term"),
    limit: int = Query(20, ge=1, le=100),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Search devices in a specific country's registry.

    Supported countries: US (GUDID), EU (EUDAMED), CH (Swissdamed),
    BR (ANVISA), AU (ARTG), CA (MDALL), IN (SUGAM).
    """
    country_code = country.strip().upper()
    adapter = _get_adapter(country_code)
    if adapter is None:
        available = sorted(_COUNTRY_ADAPTER_MAP.keys())
        raise HTTPException(
            status_code=404,
            detail=f"No registry adapter for country '{country_code}'. "
            f"Available: {', '.join(available)}",
        )

    registry_name = _COUNTRY_ADAPTER_MAP[country_code][0]

    try:
        if asyncio.iscoroutinefunction(getattr(adapter, "search", None)):
            results = await adapter.search(query=query, limit=limit)
        else:
            results = await asyncio.to_thread(adapter.search, query=query, limit=limit)
    except Exception as e:
        logger.error("Registry search failed for %s: %s", country_code, e, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Registry query failed for {registry_name} ({country_code}): {e}",
        )

    normalized = []
    for item in results or []:
        if hasattr(item, "model_dump"):
            normalized.append(item.model_dump())
        elif isinstance(item, dict):
            normalized.append(item)
        else:
            normalized.append({"raw": str(item)})

    payload = {
        "country": country_code,
        "registry": registry_name,
        "query": query,
        "count": len(normalized),
        "devices": normalized,
    }

    try:
        return JSONResponse(content=sign_payload(payload))
    except Exception:
        return payload


@router.get("/{country}/device/{device_id}")
async def get_device_by_id(
    country: str,
    device_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Lookup a specific device by UDI-DI or registration number in a country's registry.
    """
    country_code = country.strip().upper()
    adapter = _get_adapter(country_code)
    if adapter is None:
        raise HTTPException(
            status_code=404,
            detail=f"No registry adapter for country '{country_code}'.",
        )

    registry_name = _COUNTRY_ADAPTER_MAP[country_code][0]

    try:
        if asyncio.iscoroutinefunction(getattr(adapter, "get_device", None)):
            device = await adapter.get_device(device_id=device_id)
        elif hasattr(adapter, "get_device"):
            device = await asyncio.to_thread(adapter.get_device, device_id=device_id)
        else:
            # Fallback: search with device_id as query, take first exact match
            if asyncio.iscoroutinefunction(getattr(adapter, "search", None)):
                results = await adapter.search(query=device_id, limit=5)
            else:
                results = await asyncio.to_thread(adapter.search, query=device_id, limit=5)
            device = results[0] if results else None
    except Exception as e:
        logger.error("Device lookup failed: %s/%s: %s", country_code, device_id, e, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Device lookup failed for {registry_name}: {e}",
        )

    if device is None:
        raise HTTPException(
            status_code=404,
            detail=f"Device '{device_id}' not found in {registry_name} ({country_code}).",
        )

    if hasattr(device, "model_dump"):
        device_data = device.model_dump()
    elif isinstance(device, dict):
        device_data = device
    else:
        device_data = {"raw": str(device)}

    device_data.setdefault("country", country_code)
    device_data.setdefault("registry", registry_name)

    payload = {
        "country": country_code,
        "registry": registry_name,
        "device_id": device_id,
        "device": device_data,
    }

    try:
        return JSONResponse(content=sign_payload(payload))
    except Exception:
        return payload


@router.get("/gmdn/{code}")
async def gmdn_lookup(
    code: str,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    GMDN nomenclature lookup by code.

    Returns the GMDN term definition and cross-country mapping showing
    which national registries reference this GMDN code.
    """
    if GMDNAdapter is None:
        raise HTTPException(
            status_code=503,
            detail="GMDN adapter not available. Install app.adapters.gmdn_adapter.",
        )

    adapter = GMDNAdapter()

    try:
        if asyncio.iscoroutinefunction(getattr(adapter, "lookup", None)):
            result = await adapter.lookup(code=code)
        elif hasattr(adapter, "lookup"):
            result = await asyncio.to_thread(adapter.lookup, code=code)
        else:
            raise HTTPException(
                status_code=501,
                detail="GMDN adapter does not implement lookup().",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("GMDN lookup failed for code %s: %s", code, e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"GMDN lookup failed: {e}")

    if result is None:
        raise HTTPException(status_code=404, detail=f"GMDN code '{code}' not found.")

    if hasattr(result, "model_dump"):
        result_data = result.model_dump()
    elif isinstance(result, dict):
        result_data = result
    else:
        result_data = {"code": code, "raw": str(result)}

    payload = {"gmdn_code": code, "result": result_data}

    try:
        return JSONResponse(content=sign_payload(payload))
    except Exception:
        return payload


@router.get("/udi/lookup")
async def udi_lookup(
    udi_di: str = Query(..., min_length=1, max_length=200, description="UDI-DI string to search"),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Cross-database UDI-DI lookup.

    Searches GUDID (US), EUDAMED (EU), Swissdamed (CH), AusUDID (AU),
    and SIUD (BR) concurrently for a given UDI-DI.
    """
    # UDI databases: a subset of registries that support UDI-based lookup
    udi_registries: dict[str, str] = {}
    for cc, (name, _) in _COUNTRY_ADAPTER_MAP.items():
        if cc in ("US", "EU", "CH", "AU", "BR"):
            udi_registries[cc] = name

    if not udi_registries:
        raise HTTPException(
            status_code=503,
            detail="No UDI-capable registry adapters are available.",
        )

    async def _lookup_udi_in_registry(cc: str) -> tuple[str, Optional[dict[str, Any]], Optional[str]]:
        adapter = _get_adapter(cc)
        if adapter is None:
            return cc, None, "Adapter not available"
        try:
            if hasattr(adapter, "lookup_udi"):
                if asyncio.iscoroutinefunction(adapter.lookup_udi):
                    result = await adapter.lookup_udi(udi_di=udi_di)
                else:
                    result = await asyncio.to_thread(adapter.lookup_udi, udi_di=udi_di)
            else:
                # Fall back to search
                if asyncio.iscoroutinefunction(getattr(adapter, "search", None)):
                    results = await adapter.search(query=udi_di, limit=3)
                else:
                    results = await asyncio.to_thread(adapter.search, query=udi_di, limit=3)
                result = results[0] if results else None

            if result is None:
                return cc, None, None

            if hasattr(result, "model_dump"):
                return cc, result.model_dump(), None
            elif isinstance(result, dict):
                return cc, result, None
            else:
                return cc, {"raw": str(result)}, None
        except Exception as e:
            return cc, None, str(e)

    tasks = [_lookup_udi_in_registry(cc) for cc in udi_registries]
    gathered = await asyncio.gather(*tasks)

    found_in: list[str] = []
    results_map: dict[str, dict[str, Any]] = {}

    for cc, result, error in gathered:
        if result is not None:
            found_in.append(cc)
            results_map[cc] = result
        if error:
            logger.debug("UDI lookup %s error for %s: %s", udi_di, cc, error)

    payload = UDILookupResponse(
        udi_di=udi_di,
        found_in=found_in,
        results=results_map,
        total_matches=len(found_in),
    ).model_dump()

    try:
        return JSONResponse(content=sign_payload(payload))
    except Exception:
        return payload


@router.post("/legacy/map")
async def map_legacy_certificate(
    body: LegacyMapRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Map a legacy MDD/AIMDD certificate to its MDR equivalent.

    Uses the LegacyMapper service to check transition status, deadlines,
    and MDR device classification for certificates issued under the
    Medical Devices Directive (93/42/EEC) or Active Implantable Medical
    Devices Directive (90/385/EEC).
    """
    if LegacyMapper is None:
        raise HTTPException(
            status_code=503,
            detail="LegacyMapper service not available. Install app.services.legacy_mapper.",
        )

    mapper = LegacyMapper()

    try:
        if asyncio.iscoroutinefunction(getattr(mapper, "map_certificate", None)):
            result = await mapper.map_certificate(
                cert_number=body.cert_number,
                notified_body=body.notified_body,
            )
        elif hasattr(mapper, "map_certificate"):
            result = await asyncio.to_thread(
                mapper.map_certificate,
                cert_number=body.cert_number,
                notified_body=body.notified_body,
            )
        else:
            raise HTTPException(
                status_code=501,
                detail="LegacyMapper does not implement map_certificate().",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Legacy mapping failed for cert %s: %s", body.cert_number, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Legacy mapping failed: {e}")

    if hasattr(result, "model_dump"):
        result_data = result.model_dump()
    elif isinstance(result, dict):
        result_data = result
    else:
        result_data = {"raw": str(result)}

    payload = {
        "cert_number": body.cert_number,
        "notified_body": body.notified_body,
        "mapping": result_data,
    }

    try:
        return JSONResponse(content=sign_payload(payload))
    except Exception:
        return payload


@router.get("/health")
async def registry_health(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Health check for all configured registry adapters.

    Returns connectivity status, availability, and latency for each
    registry adapter. Useful for monitoring and debugging data source issues.
    """
    statuses: list[dict[str, Any]] = []

    async def _check_adapter(cc: str, registry_name: str, adapter_cls: Any) -> AdapterHealthStatus:
        import time as _time

        try:
            adapter = adapter_cls()
            start = _time.monotonic()

            if hasattr(adapter, "health_check"):
                if asyncio.iscoroutinefunction(adapter.health_check):
                    ok = await adapter.health_check()
                else:
                    ok = await asyncio.to_thread(adapter.health_check)
            elif hasattr(adapter, "search"):
                # Probe with a minimal search
                if asyncio.iscoroutinefunction(adapter.search):
                    await adapter.search(query="test", limit=1)
                else:
                    await asyncio.to_thread(adapter.search, query="test", limit=1)
                ok = True
            else:
                ok = True  # Adapter exists but has no search — assume OK

            elapsed_ms = round((_time.monotonic() - start) * 1000, 1)

            return AdapterHealthStatus(
                registry=registry_name,
                country=cc,
                available=bool(ok),
                status="healthy" if ok else "degraded",
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            return AdapterHealthStatus(
                registry=registry_name,
                country=cc,
                available=False,
                status="unreachable",
                error=str(e),
            )

    tasks = [
        _check_adapter(cc, name, cls)
        for cc, (name, cls) in _COUNTRY_ADAPTER_MAP.items()
    ]

    # Also check GMDN
    if GMDNAdapter is not None:
        tasks.append(_check_adapter("GMDN", "GMDN", GMDNAdapter))

    results = await asyncio.gather(*tasks)

    health_list = [r.model_dump() for r in results]
    healthy_count = sum(1 for r in results if r.available)
    total_count = len(results)

    payload = {
        "adapters": health_list,
        "healthy": healthy_count,
        "total": total_count,
        "overall_status": "healthy" if healthy_count == total_count else (
            "degraded" if healthy_count > 0 else "down"
        ),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    return payload
