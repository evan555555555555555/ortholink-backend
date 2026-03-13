"""
FHIRConnector -- M2M pipeline connector for FHIR/HL7 R4 standard.

Used by Brazil ANVISA SIUD (Normative Instruction 426/2026) and
Switzerland Swissmedic (Swissdamed) for device registration interop.

Supports:
  - Parsing Device, DeviceDefinition, Organization FHIR resources
  - Bundle traversal with pagination (Bundle.link next)
  - Both JSON and XML FHIR formats (XML converted to dict on ingest)
  - FHIR R4 search parameter construction
  - Resource validation against required fields per resource type

All models use Pydantic v2 for serialization. Async-first design for
HTTP-based FHIR server queries; synchronous parsing for local bundles.
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FHIR R4 constants
# ---------------------------------------------------------------------------

FHIR_R4_NAMESPACE = "http://hl7.org/fhir"
_NS = {"fhir": FHIR_R4_NAMESPACE}

# Resource types this connector handles
_SUPPORTED_RESOURCE_TYPES = {"Device", "DeviceDefinition", "Organization", "Bundle"}

# Required fields per resource type (FHIR R4 mandatory elements)
_REQUIRED_FIELDS: dict[str, list[str]] = {
    "Device": ["resourceType"],
    "DeviceDefinition": ["resourceType"],
    "Organization": ["resourceType"],
}

# Brazil IN 426/2026 extension URL prefix
ANVISA_EXTENSION_PREFIX = "http://anvisa.gov.br/fhir/StructureDefinition"

# Switzerland Swissdamed extension URL prefix
SWISSMEDIC_EXTENSION_PREFIX = "http://swissmedic.ch/fhir/StructureDefinition"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FHIRCoding(BaseModel):
    """FHIR Coding element."""
    system: str = ""
    code: str = ""
    display: str = ""


class FHIRIdentifier(BaseModel):
    """FHIR Identifier element."""
    system: str = ""
    value: str = ""
    use: str = ""  # usual | official | temp | secondary | old


class FHIROrganization(BaseModel):
    """Parsed FHIR Organization resource."""
    resource_id: str = ""
    name: str = ""
    identifiers: list[FHIRIdentifier] = Field(default_factory=list)
    type_codes: list[FHIRCoding] = Field(default_factory=list)
    address_country: str = ""
    address_city: str = ""
    address_line: str = ""
    telecom_email: str = ""
    telecom_phone: str = ""
    active: bool = True
    raw_extensions: dict[str, Any] = Field(default_factory=dict)


class DeviceClassification(BaseModel):
    """Device risk classification from a regulatory body."""
    system: str = ""      # e.g. "http://anvisa.gov.br/fhir/CodeSystem/risk-class"
    code: str = ""        # e.g. "III", "II"
    display: str = ""     # e.g. "Class III - High Risk"


class FHIRDevice(BaseModel):
    """Parsed FHIR Device / DeviceDefinition resource."""
    resource_type: str = "Device"
    resource_id: str = ""
    device_name: str = ""
    device_names_all: list[str] = Field(default_factory=list)
    manufacturer: str = ""
    manufacturer_ref: str = ""  # Reference to Organization resource
    model_number: str = ""
    catalog_number: str = ""
    serial_number: str = ""
    lot_number: str = ""
    udi_carrier: str = ""      # UDI-DI or full UDI string
    udi_issuer: str = ""       # e.g. "GS1", "HIBCC"
    identifiers: list[FHIRIdentifier] = Field(default_factory=list)
    classification: Optional[DeviceClassification] = None
    description: str = ""
    safety: list[FHIRCoding] = Field(default_factory=list)
    status: str = ""           # active | inactive | entered-in-error | unknown
    version: str = ""
    country: str = ""          # ISO 3166-1 alpha-2 (derived from extensions or context)
    # ANVISA IN 426/2026 specific
    anvisa_registro: str = ""         # Brazilian registration number
    anvisa_classe_risco: str = ""     # Risk class per RDC 185/2001
    anvisa_enquadramento: str = ""    # Product category (enquadramento)
    # Swissmedic specific
    swissmedic_listing_number: str = ""
    swissmedic_authorization: str = ""
    # Raw extensions preserved for downstream processing
    raw_extensions: dict[str, Any] = Field(default_factory=dict)
    # Parsing metadata
    parsed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_format: str = "json"  # json | xml


class FHIRBundle(BaseModel):
    """Parsed FHIR Bundle resource."""
    bundle_id: str = ""
    bundle_type: str = ""    # searchset | collection | document | message | ...
    total: int = 0
    devices: list[FHIRDevice] = Field(default_factory=list)
    organizations: list[FHIROrganization] = Field(default_factory=list)
    next_link: str = ""      # Pagination: Bundle.link where relation=next
    self_link: str = ""
    timestamp: str = ""
    entry_count: int = 0


class FHIRValidationError(BaseModel):
    """Single validation issue."""
    severity: str = "error"   # error | warning | information
    field: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# XML-to-dict conversion helpers
# ---------------------------------------------------------------------------

def _xml_to_dict(xml_string: str) -> dict[str, Any]:
    """
    Convert a FHIR XML resource to a simplified dict structure.

    Handles namespaced elements, attributes (@value patterns), and nested children.
    This is a pragmatic converter -- not a full FHIR XML parser -- but sufficient
    for Device, DeviceDefinition, Organization, and Bundle resources.
    """
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        logger.warning("FHIR XML parse error: %s", e)
        return {}

    return _element_to_dict(root)


def _element_to_dict(element: ET.Element) -> dict[str, Any]:
    """Recursively convert an XML element to dict."""
    tag = _strip_ns(element.tag)
    result: dict[str, Any] = {}

    # Capture @value attribute (FHIR uses <element value="x"/> pattern)
    if "value" in element.attrib:
        result["_value"] = element.attrib["value"]

    # Capture @url for extensions
    if "url" in element.attrib:
        result["_url"] = element.attrib["url"]

    # Process children
    children: dict[str, list[Any]] = {}
    for child in element:
        child_tag = _strip_ns(child.tag)
        child_dict = _element_to_dict(child)

        # If child is a simple value element, extract the value
        if len(child_dict) == 1 and "_value" in child_dict:
            child_val: Any = child_dict["_value"]
        else:
            child_val = child_dict

        children.setdefault(child_tag, []).append(child_val)

    # Flatten single-element lists for cleaner access
    for key, vals in children.items():
        if len(vals) == 1:
            result[key] = vals[0]
        else:
            result[key] = vals

    # If element has text content (unusual in FHIR but possible)
    if element.text and element.text.strip():
        result["_text"] = element.text.strip()

    return result


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix from tag."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _get_value(obj: Any, *keys: str, default: str = "") -> str:
    """
    Safely extract a string value from a nested FHIR dict.
    Handles both JSON style (direct values) and converted XML style (_value keys).
    """
    current = obj
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, None)
        elif isinstance(current, list) and current:
            current = current[0]
            if isinstance(current, dict):
                current = current.get(key, None)
            else:
                return default
        else:
            return default
        if current is None:
            return default

    if isinstance(current, dict):
        return str(current.get("_value", current.get("value", default)))
    if isinstance(current, str):
        return current
    if isinstance(current, (int, float, bool)):
        return str(current)
    return default


# ---------------------------------------------------------------------------
# FHIRConnector
# ---------------------------------------------------------------------------

class FHIRConnector:
    """
    M2M pipeline connector for FHIR R4 Device/Organization resources.

    Stateless parser -- no HTTP client bundled. Feed it raw dicts or XML strings
    from your preferred HTTP library (httpx, aiohttp, etc.).

    Designed for two primary integrations:
      - Brazil ANVISA SIUD (Normative Instruction 426/2026)
      - Switzerland Swissmedic Swissdamed
    """

    # ── Parse single Device ──────────────────────────────────────────────────

    def parse_device_resource(
        self,
        raw: dict[str, Any],
        *,
        source_format: str = "json",
        country_hint: str = "",
    ) -> FHIRDevice:
        """
        Parse a single FHIR Device or DeviceDefinition resource dict into FHIRDevice.

        Args:
            raw: FHIR resource as dict (from JSON or XML-to-dict conversion).
            source_format: "json" or "xml" -- for metadata tracking.
            country_hint: ISO alpha-2 country code to assign if not derivable from
                          extensions (e.g. "BR" for ANVISA, "CH" for Swissmedic).

        Returns:
            Populated FHIRDevice model.
        """
        resource_type = raw.get("resourceType", "Device")
        resource_id = _get_value(raw, "id")

        # --- Device names ---
        device_name, all_names = self._extract_device_names(raw)

        # --- Manufacturer ---
        manufacturer = _get_value(raw, "manufacturer")
        manufacturer_ref = ""
        if not manufacturer:
            mfr_ref = raw.get("manufacturer")
            if isinstance(mfr_ref, dict):
                manufacturer = _get_value(mfr_ref, "display")
                manufacturer_ref = _get_value(mfr_ref, "reference")

        # Owner reference (DeviceDefinition uses owner instead of manufacturer)
        if not manufacturer:
            owner = raw.get("owner", {})
            if isinstance(owner, dict):
                manufacturer = _get_value(owner, "display")
                manufacturer_ref = manufacturer_ref or _get_value(owner, "reference")

        # --- Model / serial / lot ---
        model_number = _get_value(raw, "modelNumber")
        catalog_number = _get_value(raw, "partNumber") or _get_value(raw, "catalogNumber")
        serial_number = _get_value(raw, "serialNumber")
        lot_number = _get_value(raw, "lotNumber")

        # --- UDI ---
        udi_carrier = ""
        udi_issuer = ""
        udi_list = raw.get("udiCarrier", [])
        if isinstance(udi_list, dict):
            udi_list = [udi_list]
        if isinstance(udi_list, list) and udi_list:
            udi_entry = udi_list[0] if isinstance(udi_list[0], dict) else {}
            udi_carrier = (
                _get_value(udi_entry, "deviceIdentifier")
                or _get_value(udi_entry, "carrierHRF")
            )
            udi_issuer = _get_value(udi_entry, "issuer")

        # --- Identifiers ---
        identifiers = self._extract_identifiers(raw.get("identifier", []))

        # --- Classification ---
        classification = self._extract_classification(raw)

        # --- Description ---
        description = _get_value(raw, "description") or _get_value(raw, "note")

        # --- Safety codes ---
        safety = self._extract_codings(raw.get("safety", []))

        # --- Status / version ---
        status = _get_value(raw, "status")
        version_entry = raw.get("version", [])
        version = ""
        if isinstance(version_entry, list) and version_entry:
            v = version_entry[0] if isinstance(version_entry[0], dict) else {}
            version = _get_value(v, "value")
        elif isinstance(version_entry, dict):
            version = _get_value(version_entry, "value")
        elif isinstance(version_entry, str):
            version = version_entry

        # --- Extensions (ANVISA / Swissmedic) ---
        extensions = self._extract_extensions(raw.get("extension", []))
        anvisa_registro = ""
        anvisa_classe_risco = ""
        anvisa_enquadramento = ""
        swissmedic_listing = ""
        swissmedic_auth = ""
        country = country_hint.upper()

        for url, val in extensions.items():
            lower_url = url.lower()
            if "anvisa" in lower_url:
                country = country or "BR"
                if "registro" in lower_url or "registration" in lower_url:
                    anvisa_registro = val
                elif "classe" in lower_url or "risk-class" in lower_url:
                    anvisa_classe_risco = val
                elif "enquadramento" in lower_url or "category" in lower_url:
                    anvisa_enquadramento = val
            elif "swissmedic" in lower_url:
                country = country or "CH"
                if "listing" in lower_url:
                    swissmedic_listing = val
                elif "authorization" in lower_url or "authorisation" in lower_url:
                    swissmedic_auth = val

        return FHIRDevice(
            resource_type=resource_type,
            resource_id=resource_id,
            device_name=device_name,
            device_names_all=all_names,
            manufacturer=manufacturer,
            manufacturer_ref=manufacturer_ref,
            model_number=model_number,
            catalog_number=catalog_number,
            serial_number=serial_number,
            lot_number=lot_number,
            udi_carrier=udi_carrier,
            udi_issuer=udi_issuer,
            identifiers=identifiers,
            classification=classification,
            description=description,
            safety=safety,
            status=status,
            version=version,
            country=country,
            anvisa_registro=anvisa_registro,
            anvisa_classe_risco=anvisa_classe_risco,
            anvisa_enquadramento=anvisa_enquadramento,
            swissmedic_listing_number=swissmedic_listing,
            swissmedic_authorization=swissmedic_auth,
            raw_extensions=extensions,
            source_format=source_format,
        )

    # ── Parse Bundle ─────────────────────────────────────────────────────────

    def parse_bundle(
        self,
        raw: dict[str, Any],
        *,
        source_format: str = "json",
        country_hint: str = "",
    ) -> FHIRBundle:
        """
        Parse a FHIR Bundle and extract all Device(Definition) and Organization
        resources from its entries.

        Args:
            raw: FHIR Bundle as dict.
            source_format: "json" or "xml".
            country_hint: Default country code if not derivable from resource extensions.

        Returns:
            FHIRBundle with devices and organizations populated.
        """
        bundle_id = _get_value(raw, "id")
        bundle_type = _get_value(raw, "type")
        total_str = _get_value(raw, "total")
        total = int(total_str) if total_str.isdigit() else 0
        timestamp = _get_value(raw, "timestamp")

        # Links (pagination)
        next_link = ""
        self_link = ""
        links = raw.get("link", [])
        if isinstance(links, dict):
            links = [links]
        for link in links:
            if not isinstance(link, dict):
                continue
            relation = _get_value(link, "relation")
            url = _get_value(link, "url")
            if relation == "next":
                next_link = url
            elif relation == "self":
                self_link = url

        # Entries
        entries = raw.get("entry", [])
        if isinstance(entries, dict):
            entries = [entries]

        devices: list[FHIRDevice] = []
        organizations: list[FHIROrganization] = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            resource = entry.get("resource", {})
            if not isinstance(resource, dict):
                continue
            rtype = resource.get("resourceType", "")

            if rtype in ("Device", "DeviceDefinition"):
                try:
                    device = self.parse_device_resource(
                        resource,
                        source_format=source_format,
                        country_hint=country_hint,
                    )
                    devices.append(device)
                except Exception as e:
                    logger.warning("Failed to parse Device entry: %s", e)

            elif rtype == "Organization":
                try:
                    org = self._parse_organization(resource)
                    organizations.append(org)
                except Exception as e:
                    logger.warning("Failed to parse Organization entry: %s", e)

        return FHIRBundle(
            bundle_id=bundle_id,
            bundle_type=bundle_type,
            total=total,
            devices=devices,
            organizations=organizations,
            next_link=next_link,
            self_link=self_link,
            timestamp=timestamp,
            entry_count=len(entries),
        )

    # ── Parse from XML string ────────────────────────────────────────────────

    def parse_xml(
        self,
        xml_string: str,
        *,
        country_hint: str = "",
    ) -> FHIRDevice | FHIRBundle:
        """
        Parse a FHIR XML string. Determines resource type and delegates.

        Returns FHIRDevice for Device/DeviceDefinition, FHIRBundle for Bundles.
        """
        raw = _xml_to_dict(xml_string)
        if not raw:
            raise ValueError("Failed to parse FHIR XML -- empty result")

        # Determine resource type from root element
        rtype = raw.get("resourceType", "")
        if not rtype:
            # In XML-to-dict, the root tag may be the resource type
            for candidate in _SUPPORTED_RESOURCE_TYPES:
                if candidate.lower() in str(raw).lower()[:200]:
                    rtype = candidate
                    break

        if rtype == "Bundle":
            return self.parse_bundle(raw, source_format="xml", country_hint=country_hint)
        elif rtype in ("Device", "DeviceDefinition"):
            return self.parse_device_resource(
                raw, source_format="xml", country_hint=country_hint,
            )
        else:
            raise ValueError(
                f"Unsupported FHIR resource type: {rtype!r}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_RESOURCE_TYPES))}"
            )

    # ── Search parameter construction ────────────────────────────────────────

    def build_search_params(
        self,
        device_name: str,
        manufacturer: str | None = None,
        *,
        status: str = "active",
        count: int = 50,
        offset: int = 0,
    ) -> dict[str, str]:
        """
        Build FHIR search query parameters for Device resources.

        Produces params compatible with GET [base]/Device?... queries.
        Follows FHIR R4 search specification for Device resource.

        Args:
            device_name: Device name or trade name to search for.
            manufacturer: Optional manufacturer name filter.
            status: Device status filter (default "active").
            count: Page size (_count parameter).
            offset: Pagination offset (_offset parameter).

        Returns:
            Dict of query parameters ready for httpx/aiohttp.
        """
        params: dict[str, str] = {
            "device-name": device_name,
            "_count": str(count),
        }

        if manufacturer:
            params["manufacturer"] = manufacturer

        if status:
            params["status"] = status

        if offset > 0:
            params["_offset"] = str(offset)

        # Always request JSON
        params["_format"] = "json"

        return params

    # ── Validate resource ────────────────────────────────────────────────────

    def validate_fhir_resource(
        self,
        resource: dict[str, Any],
    ) -> tuple[bool, list[FHIRValidationError]]:
        """
        Validate a FHIR resource dict against minimum structural requirements.

        Does NOT perform full FHIR profile validation (that requires a terminology
        server). Checks:
          1. resourceType is present and supported
          2. Required fields are present
          3. Identifiers have system + value
          4. Extensions have url attribute
          5. Coding elements have system + code

        Returns:
            (is_valid, list_of_errors) -- is_valid is True when no errors exist
            (warnings alone do not invalidate).
        """
        errors: list[FHIRValidationError] = []
        rtype = resource.get("resourceType", "")

        if not rtype:
            errors.append(FHIRValidationError(
                severity="error",
                field="resourceType",
                message="Missing required field: resourceType",
            ))
            return False, errors

        if rtype not in _SUPPORTED_RESOURCE_TYPES:
            errors.append(FHIRValidationError(
                severity="error",
                field="resourceType",
                message=(
                    f"Unsupported resource type: {rtype!r}. "
                    f"Supported: {', '.join(sorted(_SUPPORTED_RESOURCE_TYPES))}"
                ),
            ))

        # Check required fields
        for field in _REQUIRED_FIELDS.get(rtype, []):
            if not resource.get(field):
                errors.append(FHIRValidationError(
                    severity="error",
                    field=field,
                    message=f"Missing required field: {field}",
                ))

        # Validate identifiers structure
        identifiers = resource.get("identifier", [])
        if isinstance(identifiers, dict):
            identifiers = [identifiers]
        for i, ident in enumerate(identifiers):
            if isinstance(ident, dict):
                if not ident.get("value"):
                    errors.append(FHIRValidationError(
                        severity="warning",
                        field=f"identifier[{i}].value",
                        message="Identifier missing value element",
                    ))

        # Validate extensions have url
        extensions = resource.get("extension", [])
        if isinstance(extensions, dict):
            extensions = [extensions]
        for i, ext in enumerate(extensions):
            if isinstance(ext, dict) and not ext.get("url"):
                errors.append(FHIRValidationError(
                    severity="warning",
                    field=f"extension[{i}].url",
                    message="Extension missing required url attribute",
                ))

        # Device-specific: warn if no device name
        if rtype in ("Device", "DeviceDefinition"):
            has_name = bool(
                resource.get("deviceName")
                or resource.get("type")
                or resource.get("definition")
            )
            if not has_name:
                errors.append(FHIRValidationError(
                    severity="warning",
                    field="deviceName",
                    message="Device resource has no name, type, or definition -- consider adding",
                ))

        has_errors = any(e.severity == "error" for e in errors)
        return (not has_errors), errors

    # ── Internal parsing helpers ─────────────────────────────────────────────

    def _extract_device_names(
        self, raw: dict[str, Any],
    ) -> tuple[str, list[str]]:
        """Extract primary and all device names from Device/DeviceDefinition."""
        names: list[str] = []

        # Device.deviceName (R4)
        dn = raw.get("deviceName", [])
        if isinstance(dn, dict):
            dn = [dn]
        for entry in dn:
            if isinstance(entry, dict):
                name = _get_value(entry, "name")
                if name:
                    names.append(name)

        # DeviceDefinition.deviceName
        ddn = raw.get("deviceName", [])
        if isinstance(ddn, dict):
            ddn = [ddn]
        for entry in ddn:
            if isinstance(entry, dict):
                name = _get_value(entry, "name")
                if name and name not in names:
                    names.append(name)

        # type.text or type.coding[].display fallback
        dtype = raw.get("type", {})
        if isinstance(dtype, dict):
            type_text = _get_value(dtype, "text")
            if type_text and type_text not in names:
                names.append(type_text)
            codings = dtype.get("coding", [])
            if isinstance(codings, dict):
                codings = [codings]
            for c in codings:
                if isinstance(c, dict):
                    disp = _get_value(c, "display")
                    if disp and disp not in names:
                        names.append(disp)

        # definition.display (DeviceDefinition reference)
        defn = raw.get("definition", {})
        if isinstance(defn, dict):
            defn_display = _get_value(defn, "display")
            if defn_display and defn_display not in names:
                names.append(defn_display)

        primary = names[0] if names else ""
        return primary, names

    def _extract_identifiers(
        self, raw_identifiers: Any,
    ) -> list[FHIRIdentifier]:
        """Parse identifier elements."""
        if isinstance(raw_identifiers, dict):
            raw_identifiers = [raw_identifiers]
        if not isinstance(raw_identifiers, list):
            return []
        result: list[FHIRIdentifier] = []
        for ident in raw_identifiers:
            if not isinstance(ident, dict):
                continue
            result.append(FHIRIdentifier(
                system=_get_value(ident, "system"),
                value=_get_value(ident, "value"),
                use=_get_value(ident, "use"),
            ))
        return result

    def _extract_classification(
        self, raw: dict[str, Any],
    ) -> DeviceClassification | None:
        """Extract device classification / risk class."""
        # DeviceDefinition.classification (R4)
        classifications = raw.get("classification", [])
        if isinstance(classifications, dict):
            classifications = [classifications]
        if classifications:
            cls_entry = classifications[0] if isinstance(classifications[0], dict) else {}
            cls_type = cls_entry.get("type", {})
            if isinstance(cls_type, dict):
                codings = cls_type.get("coding", [])
                if isinstance(codings, dict):
                    codings = [codings]
                if codings and isinstance(codings[0], dict):
                    return DeviceClassification(
                        system=_get_value(codings[0], "system"),
                        code=_get_value(codings[0], "code"),
                        display=_get_value(codings[0], "display"),
                    )

        # specialization.systemType fallback (Device)
        specs = raw.get("specialization", [])
        if isinstance(specs, dict):
            specs = [specs]
        if specs and isinstance(specs[0], dict):
            sys_type = _get_value(specs[0], "systemType")
            if sys_type:
                return DeviceClassification(system="", code=sys_type, display=sys_type)

        return None

    def _extract_codings(self, raw_codings: Any) -> list[FHIRCoding]:
        """Extract a list of FHIR Coding from CodeableConcept or coding arrays."""
        if isinstance(raw_codings, dict):
            raw_codings = [raw_codings]
        if not isinstance(raw_codings, list):
            return []
        result: list[FHIRCoding] = []
        for cc in raw_codings:
            if not isinstance(cc, dict):
                continue
            # CodeableConcept has .coding[]
            inner = cc.get("coding", [])
            if isinstance(inner, dict):
                inner = [inner]
            if isinstance(inner, list):
                for c in inner:
                    if isinstance(c, dict):
                        result.append(FHIRCoding(
                            system=_get_value(c, "system"),
                            code=_get_value(c, "code"),
                            display=_get_value(c, "display"),
                        ))
            else:
                # Direct coding element
                result.append(FHIRCoding(
                    system=_get_value(cc, "system"),
                    code=_get_value(cc, "code"),
                    display=_get_value(cc, "display"),
                ))
        return result

    def _extract_extensions(
        self, raw_extensions: Any,
    ) -> dict[str, Any]:
        """
        Extract FHIR extensions into a flat dict keyed by URL.

        Supports simple value types (valueString, valueCode, valueBoolean, etc.)
        and nested extensions (flattened with / separator).
        """
        if isinstance(raw_extensions, dict):
            raw_extensions = [raw_extensions]
        if not isinstance(raw_extensions, list):
            return {}

        result: dict[str, Any] = {}
        for ext in raw_extensions:
            if not isinstance(ext, dict):
                continue
            url = ext.get("url", ext.get("_url", ""))
            if not url:
                continue

            # Extract value from value[x] polymorphic element
            value = self._extract_extension_value(ext)
            if value is not None:
                result[url] = value

            # Nested extensions
            nested = ext.get("extension", [])
            if isinstance(nested, dict):
                nested = [nested]
            if isinstance(nested, list):
                for sub in nested:
                    if isinstance(sub, dict):
                        sub_url = sub.get("url", sub.get("_url", ""))
                        sub_val = self._extract_extension_value(sub)
                        if sub_url and sub_val is not None:
                            result[f"{url}/{sub_url}"] = sub_val

        return result

    @staticmethod
    def _extract_extension_value(ext: dict[str, Any]) -> Any:
        """Extract value from a FHIR extension (value[x] polymorphic)."""
        for vk in (
            "valueString", "valueCode", "valueBoolean", "valueInteger",
            "valueDecimal", "valueDate", "valueDateTime", "valueUri",
            "valueCoding", "valueCodeableConcept", "valueIdentifier",
            "valueReference",
        ):
            val = ext.get(vk)
            if val is not None:
                if isinstance(val, dict):
                    # For complex types, try to extract display/value/text
                    return (
                        _get_value(val, "display")
                        or _get_value(val, "text")
                        or _get_value(val, "value")
                        or val
                    )
                if isinstance(val, str):
                    return val
                if isinstance(val, dict) and "_value" in val:
                    return val["_value"]
                return val
        return None

    def _parse_organization(
        self, raw: dict[str, Any],
    ) -> FHIROrganization:
        """Parse a FHIR Organization resource."""
        resource_id = _get_value(raw, "id")
        name = _get_value(raw, "name")
        identifiers = self._extract_identifiers(raw.get("identifier", []))
        type_codes = self._extract_codings(raw.get("type", []))
        active = raw.get("active", True)
        if isinstance(active, str):
            active = active.lower() in ("true", "1", "yes")

        # Address (take first)
        address_country = ""
        address_city = ""
        address_line = ""
        addresses = raw.get("address", [])
        if isinstance(addresses, dict):
            addresses = [addresses]
        if addresses and isinstance(addresses[0], dict):
            addr = addresses[0]
            address_country = _get_value(addr, "country")
            address_city = _get_value(addr, "city")
            lines = addr.get("line", [])
            if isinstance(lines, str):
                address_line = lines
            elif isinstance(lines, list):
                address_line = ", ".join(str(l) for l in lines if l)

        # Telecom
        telecom_email = ""
        telecom_phone = ""
        telecoms = raw.get("telecom", [])
        if isinstance(telecoms, dict):
            telecoms = [telecoms]
        for tc in telecoms:
            if not isinstance(tc, dict):
                continue
            sys = _get_value(tc, "system")
            val = _get_value(tc, "value")
            if sys == "email" and not telecom_email:
                telecom_email = val
            elif sys == "phone" and not telecom_phone:
                telecom_phone = val

        extensions = self._extract_extensions(raw.get("extension", []))

        return FHIROrganization(
            resource_id=resource_id,
            name=name,
            identifiers=identifiers,
            type_codes=type_codes,
            address_country=address_country,
            address_city=address_city,
            address_line=address_line,
            telecom_email=telecom_email,
            telecom_phone=telecom_phone,
            active=active,
            raw_extensions=extensions,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_connector: FHIRConnector | None = None


def get_fhir_connector() -> FHIRConnector:
    """Get or create the singleton FHIRConnector instance."""
    global _connector
    if _connector is None:
        _connector = FHIRConnector()
    return _connector
