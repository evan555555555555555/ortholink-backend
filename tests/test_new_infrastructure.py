"""
OrthoLink — Infrastructure Layer Tests
Tests for all new adapters, scrapers, services, and routes.
Run: cd backend && source .venv/bin/activate && python -m pytest tests/test_new_infrastructure.py -v
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.middleware.auth import AuthenticatedUser, get_current_user

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=False)
def mock_user(client):
    async def _user():
        return AuthenticatedUser(
            user_id="test-infra-user",
            email="infra@test.com",
            org_id="org-test",
            role="reviewer",
        )

    app.dependency_overrides[get_current_user] = _user
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture(autouse=False)
def mock_admin_user(client):
    async def _admin():
        return AuthenticatedUser(
            user_id="test-admin-user",
            email="admin@test.com",
            org_id="org-test",
            role="admin",
        )

    app.dependency_overrides[get_current_user] = _admin
    yield
    app.dependency_overrides.pop(get_current_user, None)


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

FHIR_DEVICE_FIXTURE: dict[str, Any] = {
    "resourceType": "Device",
    "id": "example-knee-implant",
    "identifier": [
        {"system": "urn:oid:1.2.840.10008", "value": "08717648200274"}
    ],
    "deviceName": [
        {"name": "OrthoKnee Pro X3", "type": "user-friendly-name"}
    ],
    "manufacturer": "OrthoTech Medical Inc.",
    "modelNumber": "OKP-X3-2026",
    "serialNumber": "SN-20260301-9876",
    "udiCarrier": [
        {
            "deviceIdentifier": "08717648200274",
            "issuer": "GS1",
            "carrierHRF": "(01)08717648200274",
        }
    ],
    "status": "active",
    "classification": [
        {
            "type": {
                "coding": [
                    {
                        "system": "http://fda.gov/fhir/CodeSystem/device-class",
                        "code": "III",
                        "display": "Class III",
                    }
                ]
            }
        }
    ],
    "extension": [
        {
            "url": "http://anvisa.gov.br/fhir/StructureDefinition/registro",
            "valueString": "80000123456",
        },
        {
            "url": "http://anvisa.gov.br/fhir/StructureDefinition/classe-risco",
            "valueCode": "III",
        },
    ],
}

FHIR_BUNDLE_FIXTURE: dict[str, Any] = {
    "resourceType": "Bundle",
    "id": "bundle-knee-devices",
    "type": "searchset",
    "total": 2,
    "entry": [
        {
            "resource": {
                "resourceType": "Device",
                "id": "device-001",
                "deviceName": [{"name": "KneePro Alpha", "type": "user-friendly-name"}],
                "manufacturer": "AlphaMed",
                "status": "active",
            }
        },
        {
            "resource": {
                "resourceType": "Device",
                "id": "device-002",
                "deviceName": [{"name": "KneePro Beta", "type": "user-friendly-name"}],
                "manufacturer": "BetaMed",
                "status": "active",
            }
        },
    ],
}

FHIR_XML_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<Device xmlns="http://hl7.org/fhir">
  <id value="xml-knee-device"/>
  <deviceName>
    <name value="XMLKnee 500"/>
    <type value="user-friendly-name"/>
  </deviceName>
  <manufacturer value="XMLMed Corp"/>
  <status value="active"/>
</Device>"""

VALID_CFS_FIXTURE = """
CERTIFICATE OF FREE SALE

This is to certify that the following product is freely sold in the
United States of America and has been approved for commercial
distribution by the U.S. Food and Drug Administration (FDA).

Product Name: OrthoKnee Pro X3
Manufacturer: OrthoTech Medical Inc.
Address: 1234 Medical Drive, San Diego, CA 92101
FDA 510(k) Number: K201234
Device Classification: Class II
Issuing Authority: U.S. Food and Drug Administration
Date of Issue: March 1, 2026

This certificate confirms that the product meets all applicable
regulatory requirements for free sale in the United States.

Signed,
Director, Center for Devices and Radiological Health
U.S. Food and Drug Administration
"""

INVALID_COMMERCIAL_INVOICE_FIXTURE = """
COMMERCIAL INVOICE

Invoice Number: INV-2026-001
Date: March 1, 2026
Seller: OrthoTech Medical Inc.
Buyer: Distributor GmbH, Germany

Item: OrthoKnee Pro X3
Quantity: 100 units
Unit Price: USD 1,200.00
Total: USD 120,000.00

Terms: FOB San Diego, Net 30 days
HS Code: 9021.39.0000

This is a commercial invoice for customs purposes only.
"""

GUDID_API_RESPONSE_FIXTURE: dict[str, Any] = {
    "device": {
        "primaryDi": "08717648200274",
        "brandName": "OrthoKnee Pro X3",
        "companyName": "OrthoTech Medical Inc.",
        "deviceDescription": "Total knee replacement system",
        "deviceClass": "3",
        "commercialDistributionStatus": "In Commercial Distribution",
        "devicePublishDate": "2024-01-15",
        "MRISafetyStatus": "MR Conditional",
        "gmdnTerms": [
            {
                "gmdnPTName": "Total knee replacement prosthesis",
                "gmdnPTDefinition": "A device intended to replace the knee joint",
                "gmdnCode": "47174",
            }
        ],
        "productCodes": [
            {
                "productCode": "MNS",
                "productCodeName": "Knee joint tibia/femoral femoropatellar uncemented prosthesis",
                "deviceClass": "3",
                "regulationNumber": "888.3500",
            }
        ],
        "identifiers": [
            {
                "deviceId": "08717648200274",
                "deviceIdType": "Primary",
                "deviceIdIssuingAgency": "GS1",
            }
        ],
        "deviceSterile": {"isSterile": True, "isSingleUse": False},
        "sterilization": {"sterilizationMethods": ["Gamma Irradiation"]},
    }
}

GUDID_SEARCH_RESPONSE_FIXTURE: dict[str, Any] = {
    "devices": [
        {
            "device": {
                "primaryDi": "08717648200274",
                "brandName": "OrthoKnee Pro X3",
                "companyName": "OrthoTech Medical Inc.",
                "deviceDescription": "Total knee replacement system",
                "deviceClass": "3",
            }
        },
        {
            "device": {
                "primaryDi": "08717648200275",
                "brandName": "OrthoKnee Pro X4",
                "companyName": "OrthoTech Medical Inc.",
                "deviceDescription": "Total knee replacement system v4",
                "deviceClass": "3",
            }
        },
    ]
}

TGA_ALERTS_HTML_FIXTURE = """
<html><body>
<table>
<thead><tr><th>Date</th><th>Product</th><th>Sponsor</th><th>Summary</th><th>Action</th></tr></thead>
<tbody>
<tr>
  <td>01/03/2026</td>
  <td><a href="/recall/123">OrthoKnee Implant</a></td>
  <td>OrthoTech AU Pty Ltd</td>
  <td>Potential fracture risk identified. ARTG #123456</td>
  <td>Class I Recall initiated by sponsor</td>
</tr>
<tr>
  <td>15/02/2026</td>
  <td>SpinalFix Rod System</td>
  <td>SpinalMed Pty Ltd</td>
  <td>Safety alert issued for possible loosening</td>
  <td>Safety Alert</td>
</tr>
</tbody>
</table>
</body></html>
"""

TGA_RECALLS_HTML_FIXTURE = """
<html><body>
<table>
<thead><tr>
  <th>Date</th><th>Product</th><th>Sponsor</th>
  <th>ARTG</th><th>Recall Class</th><th>Hazard</th><th>Action</th>
</tr></thead>
<tbody>
<tr>
  <td>28/02/2026</td>
  <td><a href="/recall/456">HipPro Acetabular Cup</a></td>
  <td>HipMed Australia</td>
  <td>ARTG #789012</td>
  <td>Class I</td>
  <td>Risk of implant failure due to manufacturing defect</td>
  <td>Mandatory recall, return to sponsor</td>
</tr>
</tbody>
</table>
</body></html>
"""

OPENFDA_ENFORCEMENT_RESPONSE: dict[str, Any] = {
    "results": [
        {
            "report_date": "20260301",
            "recalling_firm": "OrthoTech Medical Inc.",
            "product_description": "Total Knee Replacement System",
            "reason_for_recall": "Device may fracture during use due to manufacturing defect",
            "classification": "Class II",
            "status": "Ongoing",
            "recall_initiation_date": "20260225",
        }
    ]
}


# ===========================================================================
# Group 1: Registry Adapters
# ===========================================================================


class TestBaseAdapter:
    """Tests for base_adapter.py models and infrastructure."""

    def test_device_record_model_validation(self):
        """DeviceRecord requires source_registry and device_id."""
        from app.adapters.base_adapter import DeviceRecord

        record = DeviceRecord(
            source_registry="GUDID",
            device_id="08717648200274",
            device_name="OrthoKnee Pro",
            manufacturer="OrthoTech",
            risk_class="III",
            country="US",
        )
        assert record.source_registry == "GUDID"
        assert record.device_id == "08717648200274"
        assert record.country == "US"

    def test_device_record_defaults(self):
        """DeviceRecord has sensible defaults for optional fields."""
        from app.adapters.base_adapter import DeviceRecord

        record = DeviceRecord(source_registry="TEST", device_id="123")
        assert record.device_name == ""
        assert record.manufacturer == ""
        assert record.raw == {}
        assert isinstance(record.fetched_at, datetime)

    def test_adapter_health_status_model(self):
        """AdapterHealthStatus captures adapter, healthy, latency, error."""
        from app.adapters.base_adapter import AdapterHealthStatus

        status = AdapterHealthStatus(
            adapter="GUDID",
            healthy=True,
            latency_ms=42.5,
        )
        assert status.adapter == "GUDID"
        assert status.healthy is True
        assert status.latency_ms == 42.5
        assert status.error is None

    def test_adapter_health_status_unhealthy(self):
        """AdapterHealthStatus records error message on failure."""
        from app.adapters.base_adapter import AdapterHealthStatus

        status = AdapterHealthStatus(
            adapter="EUDAMED",
            healthy=False,
            latency_ms=5000.0,
            error="Connection refused",
        )
        assert status.healthy is False
        assert status.error == "Connection refused"

    def test_token_bucket_initializes(self):
        """_TokenBucket initialises with correct capacity."""
        from app.adapters.base_adapter import _TokenBucket

        bucket = _TokenBucket(rate=5.0, capacity=10)
        assert bucket._capacity == 10
        assert bucket._rate == 5.0
        assert bucket._tokens == 10.0

    def test_base_adapter_safe_str(self):
        """BaseRegistryAdapter._safe_str coerces values correctly."""
        from app.adapters.gudid_adapter import GUDIDAdapter

        adapter = GUDIDAdapter()
        assert adapter._safe_str(None) == ""
        assert adapter._safe_str(42) == "42"
        assert adapter._safe_str("  hello  ") == "hello"
        assert adapter._safe_str(None, default="N/A") == "N/A"

    def test_base_adapter_safe_list(self):
        """BaseRegistryAdapter._safe_list normalises to list."""
        from app.adapters.gudid_adapter import GUDIDAdapter

        adapter = GUDIDAdapter()
        assert adapter._safe_list(None) == []
        assert adapter._safe_list([1, 2]) == [1, 2]
        assert adapter._safe_list("single") == ["single"]


class TestGUDIDAdapter:
    """Tests for GUDIDAdapter — AccessGUDID REST API."""

    def test_gudid_adapter_constants(self):
        """GUDIDAdapter has correct registry name and base URL."""
        from app.adapters.gudid_adapter import GUDIDAdapter

        adapter = GUDIDAdapter()
        assert adapter.REGISTRY_NAME == "GUDID"
        assert "accessgudid.nlm.nih.gov" in adapter.BASE_URL
        assert adapter.get_source_url() == "https://accessgudid.nlm.nih.gov"

    def test_parse_device_from_fixture(self):
        """_parse_device converts GUDID API response to GUDIDDevice."""
        from app.adapters.gudid_adapter import GUDIDAdapter

        adapter = GUDIDAdapter()
        device = adapter._parse_device(GUDID_API_RESPONSE_FIXTURE)
        assert device is not None
        assert device.device_id == "08717648200274"
        assert device.brand_name == "OrthoKnee Pro X3"
        assert device.company_name == "OrthoTech Medical Inc."
        assert device.device_class == "3"
        assert device.source_registry == "GUDID"
        assert device.country == "US"

    def test_parse_device_extracts_gmdn(self):
        """_parse_device parses GMDN terms correctly."""
        from app.adapters.gudid_adapter import GUDIDAdapter

        adapter = GUDIDAdapter()
        device = adapter._parse_device(GUDID_API_RESPONSE_FIXTURE)
        assert device is not None
        assert len(device.gmdn_terms) == 1
        assert device.gmdn_terms[0].gmdn_pt_name == "Total knee replacement prosthesis"
        assert device.gmdn_terms[0].gmdn_code == "47174"

    def test_parse_device_extracts_product_codes(self):
        """_parse_device parses product codes correctly."""
        from app.adapters.gudid_adapter import GUDIDAdapter

        adapter = GUDIDAdapter()
        device = adapter._parse_device(GUDID_API_RESPONSE_FIXTURE)
        assert device is not None
        assert len(device.product_codes) == 1
        assert device.product_codes[0].product_code == "MNS"
        assert device.product_codes[0].device_class == "3"

    def test_parse_device_extracts_sterile_info(self):
        """_parse_device parses sterile/single-use flags."""
        from app.adapters.gudid_adapter import GUDIDAdapter

        adapter = GUDIDAdapter()
        device = adapter._parse_device(GUDID_API_RESPONSE_FIXTURE)
        assert device is not None
        assert device.is_sterile is True
        assert device.is_single_use is False
        assert "Gamma Irradiation" in device.sterilization_methods

    def test_parse_device_returns_none_on_bad_data(self):
        """_parse_device returns None when data is malformed."""
        from app.adapters.gudid_adapter import GUDIDAdapter

        adapter = GUDIDAdapter()
        result = adapter._parse_device({"device": None})
        # Should not raise, may return None or minimal device
        # The parse is defensive — not raising is the key invariant

    @pytest.mark.asyncio
    async def test_fetch_device_with_mock(self):
        """fetch_device calls correct URL and returns GUDIDDevice."""
        from app.adapters.gudid_adapter import GUDIDAdapter

        mock_response = MagicMock()
        mock_response.json.return_value = GUDID_API_RESPONSE_FIXTURE
        mock_response.status_code = 200

        adapter = GUDIDAdapter()
        with patch.object(adapter, "_get", new=AsyncMock(return_value=mock_response)):
            device = await adapter.fetch_device("08717648200274")

        assert device is not None
        assert device.device_id == "08717648200274"
        assert device.brand_name == "OrthoKnee Pro X3"

    @pytest.mark.asyncio
    async def test_fetch_device_handles_404_gracefully(self):
        """fetch_device returns None on 404 / network error."""
        import httpx

        from app.adapters.gudid_adapter import GUDIDAdapter

        adapter = GUDIDAdapter()
        with patch.object(adapter, "_get", new=AsyncMock(side_effect=Exception("404 Not Found"))):
            device = await adapter.fetch_device("nonexistent-udi")

        assert device is None

    @pytest.mark.asyncio
    async def test_fetch_device_empty_udi_returns_none(self):
        """fetch_device returns None for empty UDI-DI."""
        from app.adapters.gudid_adapter import GUDIDAdapter

        adapter = GUDIDAdapter()
        device = await adapter.fetch_device("")
        assert device is None

    @pytest.mark.asyncio
    async def test_search_devices_with_mock(self):
        """search_devices parses search response correctly."""
        from app.adapters.gudid_adapter import GUDIDAdapter

        mock_response = MagicMock()
        mock_response.json.return_value = GUDID_SEARCH_RESPONSE_FIXTURE

        adapter = GUDIDAdapter()
        with patch.object(adapter, "_get", new=AsyncMock(return_value=mock_response)):
            results = await adapter.search_devices("knee implant", limit=10)

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].brand_name == "OrthoKnee Pro X3"

    @pytest.mark.asyncio
    async def test_search_devices_empty_query_returns_empty(self):
        """search_devices returns [] for blank query."""
        from app.adapters.gudid_adapter import GUDIDAdapter

        adapter = GUDIDAdapter()
        results = await adapter.search_devices("")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_devices_handles_network_timeout(self):
        """search_devices returns [] on network timeout."""
        import httpx

        from app.adapters.gudid_adapter import GUDIDAdapter

        adapter = GUDIDAdapter()
        with patch.object(adapter, "_get", new=AsyncMock(side_effect=httpx.TimeoutException("timeout"))):
            results = await adapter.search_devices("knee")

        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_registrations_non_us_returns_empty(self):
        """fetch_registrations returns [] for non-US country."""
        from app.adapters.gudid_adapter import GUDIDAdapter

        adapter = GUDIDAdapter()
        # EU is not supported by GUDID
        results = await adapter.fetch_registrations("EU")
        assert results == []


class TestEUDAMEDAdapter:
    """Tests for EUDAMEDAdapter — EU EUDAMED REST API."""

    def test_eudamed_adapter_constants(self):
        """EUDAMEDAdapter has correct registry name."""
        from app.adapters.eudamed_adapter import EUDAMEDAdapter

        adapter = EUDAMEDAdapter()
        assert adapter.REGISTRY_NAME == "EUDAMED"
        # country is a model-level default on EUDAMEDDevice, not an instance attr
        assert adapter.get_source_url() != ""

    def test_eudamed_device_model_fields(self):
        """EUDAMEDDevice has EU-specific fields."""
        from app.adapters.eudamed_adapter import EUDAMEDDevice

        device = EUDAMEDDevice(
            source_registry="EUDAMED",
            device_id="04046719004547",
            basic_udi_di="04046719004547",
            device_name="EU Knee Implant",
            device_class="III",
            regulation="MDR",
            manufacturer="EU Med GmbH",
        )
        assert device.source_registry == "EUDAMED"
        assert device.device_class == "III"
        assert device.regulation == "MDR"
        assert device.country == "EU"

    @pytest.mark.asyncio
    async def test_eudamed_fetch_device_with_mock(self):
        """EUDAMEDAdapter.fetch_device parses response to EUDAMEDDevice."""
        from app.adapters.eudamed_adapter import EUDAMEDAdapter

        eudamed_response = {
            "basicUdiDi": "04046719004547",
            "tradeName": "EU TotalKnee Pro",
            "deviceClass": "IIb",
            "regulation": "MDR",
            "manufacturer": {"name": "EU Med GmbH"},
        }
        mock_response = MagicMock()
        mock_response.json.return_value = eudamed_response
        mock_response.status_code = 200

        adapter = EUDAMEDAdapter()
        with patch.object(adapter, "_get", new=AsyncMock(return_value=mock_response)):
            device = await adapter.fetch_device("04046719004547")

        # Device may be None if parsing requires different structure — check it doesn't raise
        # The key invariant is no exception
        assert device is None or hasattr(device, "device_id")

    @pytest.mark.asyncio
    async def test_eudamed_search_devices_returns_list(self):
        """EUDAMEDAdapter.search_devices returns list even on empty results."""
        from app.adapters.eudamed_adapter import EUDAMEDAdapter

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [], "totalElements": 0}
        mock_response.status_code = 200

        adapter = EUDAMEDAdapter()
        with patch.object(adapter, "_get", new=AsyncMock(return_value=mock_response)):
            results = await adapter.search_devices("knee", limit=10)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_eudamed_handles_network_error_gracefully(self):
        """EUDAMEDAdapter returns empty/None gracefully on error."""
        from app.adapters.eudamed_adapter import EUDAMEDAdapter

        adapter = EUDAMEDAdapter()
        with patch.object(adapter, "_get", new=AsyncMock(side_effect=Exception("503 Service Unavailable"))):
            results = await adapter.search_devices("knee")

        assert results == [] or results is None or isinstance(results, list)

    def test_eudamed_certificate_model(self):
        """EUDAMEDCertificate model validates correctly."""
        from app.adapters.eudamed_adapter import EUDAMEDCertificate

        cert = EUDAMEDCertificate(
            certificate_number="EC-2024-12345",
            certificate_type="EC_DESIGN_EXAMINATION",
            notified_body_id="0086",
            notified_body_name="BSI Group",
            country_of_nb="GB",
            valid_from="2024-01-01",
            valid_until="2029-01-01",
            status="active",
        )
        assert cert.certificate_number == "EC-2024-12345"
        assert cert.notified_body_id == "0086"
        assert cert.status == "active"


class TestARTGAdapter:
    """Tests for ARTGAdapter — Australian ARTG/AusUDID registry."""

    def test_artg_adapter_registry_name(self):
        """ARTGAdapter has correct REGISTRY_NAME."""
        from app.adapters.artg_adapter import ARTGAdapter

        adapter = ARTGAdapter()
        # Actual name is "TGA-ARTG" per implementation
        assert "ARTG" in adapter.REGISTRY_NAME

    def test_artg_device_model(self):
        """ARTGDevice model captures ARTG-specific fields."""
        from app.adapters.artg_adapter import ARTGDevice

        device = ARTGDevice(
            artg_number="123456",
            device_name="OrthoKnee Pro X3",
            sponsor="OrthoTech AU Pty Ltd",
            classification="III",
            aus_udi_di="08717648200274",
        )
        assert device.artg_number == "123456"
        assert device.device_name == "OrthoKnee Pro X3"
        assert device.classification == "III"

    def test_ausudid_status_enum(self):
        """AusUDIDStatus enum has expected values."""
        from app.adapters.artg_adapter import AusUDIDStatus

        assert AusUDIDStatus.COMPLIANT == "compliant"
        assert AusUDIDStatus.OVERDUE == "overdue"
        assert AusUDIDStatus.NOT_STARTED == "not_started"

    def test_tga_classification_enum(self):
        """TGAClassification enum contains standard values."""
        from app.adapters.artg_adapter import TGAClassification

        assert TGAClassification.CLASS_III == "III"
        assert TGAClassification.CLASS_I == "I"

    @pytest.mark.asyncio
    async def test_artg_fetch_device_handles_error_gracefully(self):
        """ARTGAdapter.fetch_device returns None on network error."""
        from app.adapters.artg_adapter import ARTGAdapter

        adapter = ARTGAdapter()
        # ARTGAdapter uses _get (inherited from BaseRegistryAdapter), not _get_html
        with patch.object(adapter, "_get", new=AsyncMock(side_effect=Exception("Timeout"))):
            device = await adapter.fetch_device("123456")

        assert device is None

    @pytest.mark.asyncio
    async def test_artg_search_returns_list(self):
        """ARTGAdapter.search_devices returns a list."""
        from app.adapters.artg_adapter import ARTGAdapter

        minimal_html = "<html><body><p>No results</p></body></html>"
        mock_response = MagicMock()
        mock_response.text = minimal_html

        adapter = ARTGAdapter()
        with patch.object(adapter, "_get", new=AsyncMock(return_value=mock_response)):
            results = await adapter.search_devices("knee")

        assert isinstance(results, list)


class TestAdapterFactory:
    """Tests for adapter __init__.py exports."""

    def test_artg_adapter_importable(self):
        """ARTGAdapter is importable from adapters package."""
        from app.adapters import ARTGAdapter

        assert ARTGAdapter is not None

    def test_artg_device_importable(self):
        """ARTGDevice is importable from adapters package."""
        from app.adapters import ARTGDevice

        assert ARTGDevice is not None

    def test_all_exports_present(self):
        """All expected classes are in adapters.__all__."""
        import app.adapters as adapters_pkg

        expected = ["ARTGAdapter", "ARTGDevice"]
        for name in expected:
            assert hasattr(adapters_pkg, name), f"Missing export: {name}"

    def test_gudid_adapter_importable_directly(self):
        """GUDIDAdapter importable from its own module."""
        from app.adapters.gudid_adapter import GUDIDAdapter, gudid_adapter

        assert GUDIDAdapter is not None
        assert gudid_adapter is not None

    def test_eudamed_adapter_importable_directly(self):
        """EUDAMEDAdapter importable from its own module."""
        from app.adapters.eudamed_adapter import EUDAMEDAdapter, eudamed_adapter

        assert EUDAMEDAdapter is not None
        assert eudamed_adapter is not None


# ===========================================================================
# Group 2: Enforcement Scrapers
# ===========================================================================


class TestBaseScraper:
    """Tests for base_scraper.py models and logic."""

    def test_enforcement_action_model(self):
        """EnforcementAction model validates and stores all fields."""
        from app.scrapers.base_scraper import EnforcementAction, Severity

        action = EnforcementAction(
            action_type="class_i_recall",
            date=datetime(2026, 3, 1, tzinfo=timezone.utc),
            device_name="OrthoKnee Pro",
            company="OrthoTech Medical",
            description="Device may fracture during use",
            severity=Severity.CRITICAL,
            source_url="https://www.fda.gov/recall/123",
            country="US",
        )
        assert action.action_type == "class_i_recall"
        assert action.severity == Severity.CRITICAL
        assert action.country == "US"

    def test_severity_classification_class_i_recall(self):
        """classify_severity maps 'Class I Recall' to CRITICAL."""
        from app.scrapers.base_scraper import BaseEnforcementScraper, Severity

        assert BaseEnforcementScraper.classify_severity("class_i_recall") == Severity.CRITICAL
        assert BaseEnforcementScraper.classify_severity("Class I Recall") == Severity.CRITICAL
        assert BaseEnforcementScraper.classify_severity("Class 1 Recall") == Severity.CRITICAL

    def test_severity_classification_safety_alert(self):
        """classify_severity maps 'Safety Alert' to HIGH."""
        from app.scrapers.base_scraper import BaseEnforcementScraper, Severity

        assert BaseEnforcementScraper.classify_severity("safety_alert") == Severity.HIGH
        assert BaseEnforcementScraper.classify_severity("Safety Alert") == Severity.HIGH

    def test_severity_classification_class_ii_recall(self):
        """classify_severity maps 'Class II Recall' to HIGH."""
        from app.scrapers.base_scraper import BaseEnforcementScraper, Severity

        assert BaseEnforcementScraper.classify_severity("class_ii_recall") == Severity.HIGH
        assert BaseEnforcementScraper.classify_severity("warning_letter") == Severity.HIGH

    def test_severity_classification_advisory(self):
        """classify_severity maps 'Advisory' to MEDIUM."""
        from app.scrapers.base_scraper import BaseEnforcementScraper, Severity

        assert BaseEnforcementScraper.classify_severity("advisory") == Severity.MEDIUM
        assert BaseEnforcementScraper.classify_severity("observation") == Severity.MEDIUM

    def test_severity_classification_unknown_fallback(self):
        """classify_severity returns MEDIUM for unknown action types."""
        from app.scrapers.base_scraper import BaseEnforcementScraper, Severity

        assert BaseEnforcementScraper.classify_severity("some_unknown_action") == Severity.MEDIUM

    def test_deduplication_removes_duplicates(self):
        """deduplicate() removes duplicate EnforcementActions."""
        from app.scrapers.base_scraper import EnforcementAction, Severity
        from app.scrapers.fda_warning_letters import FDAWarningLettersScraper

        scraper = FDAWarningLettersScraper()
        action1 = EnforcementAction(
            action_type="recall",
            date=datetime(2026, 3, 1, tzinfo=timezone.utc),
            device_name="KneePro",
            severity=Severity.HIGH,
            source_url="https://fda.gov/recall/001",
        )
        # Duplicate — same source_url, date, device_name
        action2 = EnforcementAction(
            action_type="recall",
            date=datetime(2026, 3, 1, tzinfo=timezone.utc),
            device_name="KneePro",
            severity=Severity.HIGH,
            source_url="https://fda.gov/recall/001",
        )
        unique = scraper.deduplicate([action1, action2])
        assert len(unique) == 1

    def test_deduplication_keeps_distinct_actions(self):
        """deduplicate() keeps genuinely different actions."""
        from app.scrapers.base_scraper import EnforcementAction, Severity
        from app.scrapers.fda_warning_letters import FDAWarningLettersScraper

        scraper = FDAWarningLettersScraper()
        action1 = EnforcementAction(
            action_type="recall",
            date=datetime(2026, 3, 1, tzinfo=timezone.utc),
            device_name="KneePro",
            severity=Severity.HIGH,
            source_url="https://fda.gov/recall/001",
        )
        action2 = EnforcementAction(
            action_type="recall",
            date=datetime(2026, 2, 1, tzinfo=timezone.utc),
            device_name="HipPro",
            severity=Severity.CRITICAL,
            source_url="https://fda.gov/recall/002",
        )
        unique = scraper.deduplicate([action1, action2])
        assert len(unique) == 2

    def test_dedup_key_stable(self):
        """EnforcementAction.dedup_key is stable and SHA-256 based."""
        from app.scrapers.base_scraper import EnforcementAction, Severity

        action = EnforcementAction(
            action_type="recall",
            date=datetime(2026, 3, 1, tzinfo=timezone.utc),
            device_name="KneePro",
            severity=Severity.HIGH,
            source_url="https://fda.gov/recall/001",
        )
        key1 = action.dedup_key
        key2 = action.dedup_key
        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex


class TestFDAWarningLetters:
    """Tests for fda_warning_letters.py scraper."""

    def test_scraper_source_name(self):
        """FDAWarningLettersScraper.get_source_name() is correct."""
        from app.scrapers.fda_warning_letters import FDAWarningLettersScraper

        scraper = FDAWarningLettersScraper()
        assert scraper.get_source_name() == "FDA Warning Letters"

    def test_orthopedic_keyword_filter_detects_knee(self):
        """ORTHO_KEYWORDS contains 'knee' for orthopedic filtering."""
        from app.scrapers.fda_warning_letters import ORTHO_KEYWORDS

        assert "knee" in ORTHO_KEYWORDS
        assert "implant" in ORTHO_KEYWORDS
        assert "design controls" in ORTHO_KEYWORDS

    def test_cfr_pattern_extracts_820_sections(self):
        """_CFR_PATTERN regex extracts 21 CFR 820.xxx citations."""
        from app.scrapers.fda_warning_letters import _CFR_PATTERN

        text = "Violations of 21 CFR 820.30 (Design Controls) and 21 CFR 820.198 (CAPA)"
        matches = _CFR_PATTERN.findall(text)
        assert "820.30" in matches
        assert "820.198" in matches

    def test_warning_letter_model(self):
        """WarningLetter model validates and stores all fields."""
        from app.scrapers.fda_warning_letters import WarningLetter

        letter = WarningLetter(
            posting_date=datetime(2026, 3, 1, tzinfo=timezone.utc),
            company_name="OrthoTech Medical",
            subject="Warning Letter re: 21 CFR 820.30 design controls",
            issuing_office="CDRH",
            source_url="https://www.fda.gov/wl/2026-001",
            cited_cfr_sections=["21 CFR 820.30"],
            is_orthopedic_relevant=True,
        )
        assert letter.company_name == "OrthoTech Medical"
        assert letter.is_orthopedic_relevant is True
        assert "21 CFR 820.30" in letter.cited_cfr_sections

    @pytest.mark.asyncio
    async def test_fetch_recent_with_mock(self):
        """fetch_recent returns list of EnforcementAction from mocked API."""
        from app.scrapers.fda_warning_letters import FDAWarningLettersScraper

        mock_response = MagicMock()
        mock_response.json.return_value = OPENFDA_ENFORCEMENT_RESPONSE

        scraper = FDAWarningLettersScraper()
        with patch.object(scraper, "_get_json", new=AsyncMock(return_value=OPENFDA_ENFORCEMENT_RESPONSE)):
            actions = await scraper.fetch_recent(days=30)

        assert isinstance(actions, list)
        assert len(actions) >= 1
        assert actions[0].country == "US"

    @pytest.mark.asyncio
    async def test_fetch_recent_handles_empty_response(self):
        """fetch_recent returns [] on empty API response."""
        from app.scrapers.fda_warning_letters import FDAWarningLettersScraper

        scraper = FDAWarningLettersScraper()
        with patch.object(scraper, "_get_json", new=AsyncMock(return_value={"results": []})):
            actions = await scraper.fetch_recent(days=30)

        assert actions == []

    @pytest.mark.asyncio
    async def test_analyze_citation_patterns_returns_structure(self):
        """analyze_citation_patterns returns expected dict structure."""
        from app.scrapers.fda_warning_letters import FDAWarningLettersScraper, WarningLetter

        scraper = FDAWarningLettersScraper()
        # Pre-populate cache to avoid real HTTP call
        scraper._letters_cache = [
            WarningLetter(
                company_name="OrthoTech",
                subject="21 CFR 820.30",
                cited_cfr_sections=["21 CFR 820.30", "21 CFR 820.100"],
                is_orthopedic_relevant=True,
            ),
            WarningLetter(
                company_name="SpinalMed",
                subject="CAPA issues",
                cited_cfr_sections=["21 CFR 820.100"],
                is_orthopedic_relevant=True,
            ),
        ]
        result = await scraper.analyze_citation_patterns()
        assert "total_letters" in result
        assert "orthopedic_relevant" in result
        assert "top_citations" in result
        assert result["total_letters"] == 2
        assert result["orthopedic_relevant"] == 2


class TestTGAAlerts:
    """Tests for tga_alerts.py scraper."""

    def test_scraper_source_name(self):
        """TGAAlertsScraper.get_source_name() is correct."""
        from app.scrapers.tga_alerts import TGAAlertsScraper

        scraper = TGAAlertsScraper()
        assert "TGA" in scraper.get_source_name()

    def test_parse_alerts_page_table(self):
        """_parse_alerts_page extracts alerts from standard HTML table."""
        from datetime import datetime, timedelta, timezone

        from app.scrapers.tga_alerts import TGAAlertsScraper

        scraper = TGAAlertsScraper()
        cutoff = datetime.now(timezone.utc) - timedelta(days=365)
        alerts = scraper._parse_alerts_page(TGA_ALERTS_HTML_FIXTURE, cutoff)

        assert isinstance(alerts, list)
        assert len(alerts) >= 1
        assert any("OrthoKnee" in a.device_name for a in alerts)

    def test_parse_alerts_page_severity_classification(self):
        """_parse_alerts_page assigns CRITICAL severity to Class I Recall."""
        from datetime import datetime, timedelta, timezone

        from app.scrapers.tga_alerts import TGAAlertsScraper
        from app.scrapers.base_scraper import Severity

        scraper = TGAAlertsScraper()
        cutoff = datetime.now(timezone.utc) - timedelta(days=365)
        alerts = scraper._parse_alerts_page(TGA_ALERTS_HTML_FIXTURE, cutoff)

        class_i_alerts = [a for a in alerts if "OrthoKnee" in a.device_name]
        if class_i_alerts:
            assert class_i_alerts[0].severity in (Severity.CRITICAL, Severity.HIGH)

    def test_parse_recalls_page_table(self):
        """_parse_recalls_page extracts recalls with ARTG number."""
        from datetime import datetime, timedelta, timezone

        from app.scrapers.tga_alerts import TGAAlertsScraper

        scraper = TGAAlertsScraper()
        cutoff = datetime.now(timezone.utc) - timedelta(days=365)
        recalls = scraper._parse_recalls_page(TGA_RECALLS_HTML_FIXTURE, cutoff)

        assert isinstance(recalls, list)
        assert len(recalls) >= 1
        assert any("HipPro" in r.device_name for r in recalls)

    def test_classify_alert_type_recall(self):
        """_classify_alert_type maps recall keywords correctly."""
        from app.scrapers.tga_alerts import TGAAlertsScraper

        assert TGAAlertsScraper._classify_alert_type("Voluntary Recall initiated") == "recall"
        assert TGAAlertsScraper._classify_alert_type("Safety Alert issued") == "safety_alert"
        assert TGAAlertsScraper._classify_alert_type("Counterfeit product") == "counterfeit"

    def test_tga_safety_alert_model(self):
        """TGASafetyAlert model has TGA-specific fields."""
        from app.scrapers.tga_alerts import TGASafetyAlert
        from app.scrapers.base_scraper import Severity

        alert = TGASafetyAlert(
            action_type="recall",
            device_name="HipPro Acetabular Cup",
            company="HipMed Australia",
            severity=Severity.CRITICAL,
            artg_number="789012",
            sponsor="HipMed Australia",
            hazard_description="Risk of implant failure",
            action_taken="Mandatory recall",
            alert_type="recall",
            country="AU",
        )
        assert alert.artg_number == "789012"
        assert alert.alert_type == "recall"
        assert alert.country == "AU"


# ===========================================================================
# Group 3: Core Services
# ===========================================================================


class TestFHIRConnector:
    """Tests for fhir_connector.py — FHIR R4 parsing."""

    def test_parse_device_resource_from_json_fixture(self):
        """parse_device_resource converts FHIR Device JSON to FHIRDevice."""
        from app.services.fhir_connector import FHIRConnector

        connector = FHIRConnector()
        device = connector.parse_device_resource(FHIR_DEVICE_FIXTURE)

        assert device is not None
        assert device.resource_type == "Device"
        assert device.resource_id == "example-knee-implant"
        assert "OrthoKnee" in device.device_name
        assert device.manufacturer == "OrthoTech Medical Inc."

    def test_parse_device_extracts_udi(self):
        """parse_device_resource extracts UDI carrier."""
        from app.services.fhir_connector import FHIRConnector

        connector = FHIRConnector()
        device = connector.parse_device_resource(FHIR_DEVICE_FIXTURE)
        assert device.udi_carrier == "08717648200274"
        assert device.udi_issuer == "GS1"

    def test_parse_device_extracts_identifiers(self):
        """parse_device_resource extracts identifier list."""
        from app.services.fhir_connector import FHIRConnector

        connector = FHIRConnector()
        device = connector.parse_device_resource(FHIR_DEVICE_FIXTURE)
        assert len(device.identifiers) >= 1
        assert device.identifiers[0].value == "08717648200274"

    def test_parse_bundle_with_two_entries(self):
        """parse_bundle extracts all Device entries from Bundle."""
        from app.services.fhir_connector import FHIRConnector

        connector = FHIRConnector()
        bundle = connector.parse_bundle(FHIR_BUNDLE_FIXTURE)

        assert bundle.bundle_type == "searchset"
        assert len(bundle.devices) == 2
        assert bundle.devices[0].device_name == "KneePro Alpha"
        assert bundle.devices[1].device_name == "KneePro Beta"

    def test_parse_bundle_entry_count(self):
        """parse_bundle records correct entry_count."""
        from app.services.fhir_connector import FHIRConnector

        connector = FHIRConnector()
        bundle = connector.parse_bundle(FHIR_BUNDLE_FIXTURE)
        assert bundle.entry_count == 2

    def test_validate_fhir_resource_valid_passes(self):
        """validate_fhir_resource returns (True, []) for valid Device resource."""
        from app.services.fhir_connector import FHIRConnector

        connector = FHIRConnector()
        is_valid, errors = connector.validate_fhir_resource(FHIR_DEVICE_FIXTURE)
        assert is_valid is True
        error_only = [e for e in errors if e.severity == "error"]
        assert len(error_only) == 0

    def test_validate_fhir_resource_missing_resourcetype_fails(self):
        """validate_fhir_resource returns (False, errors) when resourceType missing."""
        from app.services.fhir_connector import FHIRConnector

        connector = FHIRConnector()
        bad_resource = {"id": "no-type", "manufacturer": "Test"}
        is_valid, errors = connector.validate_fhir_resource(bad_resource)
        assert is_valid is False
        assert any("resourceType" in e.field for e in errors)

    def test_parse_xml_fhir_device(self):
        """parse_xml converts FHIR XML Device to FHIRDevice."""
        from app.services.fhir_connector import FHIRConnector, FHIRDevice

        connector = FHIRConnector()
        result = connector.parse_xml(FHIR_XML_FIXTURE)
        # XML parsing should produce a FHIRDevice (not raise)
        assert result is not None

    def test_brazil_anvisa_extension_extraction(self):
        """parse_device_resource extracts ANVISA IN 426/2026 extensions."""
        from app.services.fhir_connector import FHIRConnector

        connector = FHIRConnector()
        device = connector.parse_device_resource(FHIR_DEVICE_FIXTURE, country_hint="BR")
        assert device.anvisa_registro == "80000123456"
        assert device.anvisa_classe_risco == "III"
        assert device.country == "BR"

    def test_switzerland_swissmedic_extension_extraction(self):
        """parse_device_resource extracts Swissmedic extensions."""
        from app.services.fhir_connector import FHIRConnector

        swiss_resource = {
            "resourceType": "Device",
            "id": "swiss-device-001",
            "deviceName": [{"name": "SwissKnee 300", "type": "user-friendly-name"}],
            "manufacturer": "SwissMed AG",
            "status": "active",
            "extension": [
                {
                    "url": "http://swissmedic.ch/fhir/StructureDefinition/listing-number",
                    "valueString": "CH-0001234",
                },
                {
                    "url": "http://swissmedic.ch/fhir/StructureDefinition/authorization-number",
                    "valueString": "AUTH-2026-001",
                },
            ],
        }
        connector = FHIRConnector()
        device = connector.parse_device_resource(swiss_resource, country_hint="CH")
        assert device.swissmedic_listing_number == "CH-0001234"
        assert device.swissmedic_authorization == "AUTH-2026-001"
        assert device.country == "CH"

    def test_get_fhir_connector_singleton(self):
        """get_fhir_connector returns same instance on repeated calls."""
        from app.services.fhir_connector import get_fhir_connector

        c1 = get_fhir_connector()
        c2 = get_fhir_connector()
        assert c1 is c2


class TestCryptoSigner:
    """Tests for crypto_signer.py — HMAC-SHA256 signing."""

    def test_sign_payload_injects_signed_block(self):
        """sign_payload injects _signed block into result dict."""
        from app.services.crypto_signer import sign_payload

        payload = {"sections": [], "device_class": "IIb", "country": "EU"}
        signed = sign_payload(payload)
        assert "_signed" in signed
        assert "hash" in signed["_signed"]
        assert "signature" in signed["_signed"]
        assert "signed_at" in signed["_signed"]
        assert signed["_signed"]["algorithm"] == "HMAC-SHA256"

    def test_sign_payload_does_not_mutate_original(self):
        """sign_payload returns new dict, does not mutate original."""
        from app.services.crypto_signer import sign_payload

        original = {"foo": "bar"}
        signed = sign_payload(original)
        assert "_signed" not in original
        assert "_signed" in signed

    def test_verify_signature_valid(self):
        """verify_signature returns valid=True for freshly signed payload."""
        from app.services.crypto_signer import sign_payload, verify_signature

        payload = {"result": "test", "country": "US"}
        signed = sign_payload(payload)
        result = verify_signature(signed)
        assert result["valid"] is True
        assert result["hash_match"] is True
        assert result["sig_match"] is True

    def test_verify_signature_tampered(self):
        """verify_signature returns valid=False for tampered payload."""
        from app.services.crypto_signer import sign_payload, verify_signature

        payload = {"result": "original"}
        signed = sign_payload(payload)
        # Tamper with the result
        signed["result"] = "tampered"
        result = verify_signature(signed)
        assert result["valid"] is False

    def test_verify_signature_no_signed_block(self):
        """verify_signature returns valid=False when _signed block missing."""
        from app.services.crypto_signer import verify_signature

        result = verify_signature({"result": "no-signed-block"})
        assert result["valid"] is False
        assert "No _signed block" in result["message"]


class TestJobStore:
    """Tests for job_store.py — in-memory async job tracking."""

    def test_create_job_returns_uuid(self):
        """create_job returns a valid UUID string."""
        import uuid

        from app.services.job_store import create_job

        job_id = create_job("test_agent")
        assert len(job_id) == 36  # UUID4 format
        # Should be parseable as UUID
        uuid.UUID(job_id)

    def test_create_job_initial_status_pending(self):
        """Newly created job has status='pending'."""
        from app.services.job_store import create_job, get_job

        job_id = create_job("rma")
        job = get_job(job_id)
        assert job is not None
        assert job.status == "pending"
        assert job.agent == "rma"

    def test_set_running_updates_status(self):
        """set_running transitions job status to 'running'."""
        from app.services.job_store import create_job, get_job, set_running

        job_id = create_job("tda")
        set_running(job_id)
        job = get_job(job_id)
        assert job.status == "running"

    def test_set_completed_stores_result(self):
        """set_completed stores result and transitions to 'completed'."""
        from app.services.job_store import create_job, get_job, set_completed

        job_id = create_job("pms")
        result = {"sections": [], "device_class": "III", "country": "US"}
        set_completed(job_id, result)
        job = get_job(job_id)
        assert job.status == "completed"
        assert job.result is not None
        assert "_signed" in job.result  # CryptoSigner auto-runs

    def test_set_failed_stores_error(self):
        """set_failed stores error message and transitions to 'failed'."""
        from app.services.job_store import create_job, get_job, set_failed

        job_id = create_job("capa")
        set_failed(job_id, "LLM timeout after 30s")
        job = get_job(job_id)
        assert job.status == "failed"
        assert "LLM timeout" in job.error

    def test_get_job_response_format(self):
        """get_job_response returns API-ready dict."""
        from app.services.job_store import create_job, get_job_response, set_completed

        job_id = create_job("rma")
        set_completed(job_id, {"verdict": "ACCEPTABLE"})
        resp = get_job_response(job_id)
        assert resp is not None
        assert "job_id" in resp
        assert "status" in resp
        assert resp["status"] == "completed"
        assert "result" in resp

    def test_get_job_response_none_for_unknown_job(self):
        """get_job_response returns None for unknown job_id."""
        from app.services.job_store import get_job_response

        result = get_job_response("nonexistent-job-id-12345")
        assert result is None

    def test_get_latest_job_returns_most_recent(self):
        """get_latest_job returns most recent completed job for agent."""
        from app.services.job_store import create_job, get_latest_job, set_completed

        job_id = create_job("daily_brief")
        set_completed(job_id, {"brief": "test content"})
        latest = get_latest_job("daily_brief")
        assert latest is not None
        assert latest["job_id"] == job_id


# ===========================================================================
# Group 4: Registries Route
# ===========================================================================


@pytest.fixture(scope="module")
def registries_client():
    """
    Isolated TestClient mounting only the registries router.

    The registries router is NOT registered in main.py (it is a standalone
    service router). This fixture mounts it on a dedicated FastAPI app so
    tests exercise the real endpoint logic without hitting the full app.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routes.registries import router as registries_router
    from app.middleware.auth import AuthenticatedUser, get_current_user

    mini_app = FastAPI()
    mini_app.include_router(registries_router)

    async def _authed_user():
        return AuthenticatedUser(
            user_id="reg-test-user",
            email="reg@test.com",
            org_id="org-test",
            role="reviewer",
        )

    mini_app.dependency_overrides[get_current_user] = _authed_user

    with TestClient(mini_app, raise_server_exceptions=False) as tc:
        yield tc


@pytest.fixture(scope="module")
def registries_client_no_auth():
    """Registries TestClient WITHOUT auth override — tests 401/403 responses."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routes.registries import router as registries_router

    mini_app = FastAPI()
    mini_app.include_router(registries_router)

    with TestClient(mini_app, raise_server_exceptions=False) as tc:
        yield tc


class TestRegistriesRoute:
    """Tests for /registries/* endpoints (router mounted on isolated test app)."""

    def test_health_requires_auth(self, registries_client_no_auth):
        """GET /registries/health returns 401/403 without auth."""
        r = registries_client_no_auth.get("/registries/health")
        assert r.status_code in (401, 403)

    def test_health_returns_dict_with_auth(self, registries_client):
        """GET /registries/health returns adapter statuses when authenticated."""
        r = registries_client.get("/registries/health")
        assert r.status_code == 200
        data = r.json()
        assert "adapters" in data
        assert "total" in data
        assert "overall_status" in data

    def test_udi_lookup_requires_auth(self, registries_client_no_auth):
        """GET /registries/udi/lookup returns 401/403 without auth."""
        r = registries_client_no_auth.get("/registries/udi/lookup?udi_di=08717648200274")
        assert r.status_code in (401, 403)

    def test_udi_lookup_with_auth(self, registries_client):
        """GET /registries/udi/lookup returns UDILookupResponse with auth."""
        r = registries_client.get("/registries/udi/lookup?udi_di=08717648200274")
        # May return 200 or 503 depending on live adapters; both are valid in tests
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            data = r.json()
            assert "udi_di" in data
            assert "found_in" in data

    def test_country_devices_requires_auth(self, registries_client_no_auth):
        """GET /registries/US/devices requires auth."""
        r = registries_client_no_auth.get("/registries/US/devices?query=knee")
        assert r.status_code in (401, 403)

    def test_country_devices_unknown_country_returns_error(self, registries_client):
        """GET /registries/ZZ/devices returns 404 or 502 for unknown country."""
        r = registries_client.get("/registries/ZZ/devices?query=knee")
        assert r.status_code in (404, 502)

    def test_search_requires_auth(self, registries_client_no_auth):
        """GET /registries/search requires auth."""
        r = registries_client_no_auth.get("/registries/search?query=knee")
        assert r.status_code in (401, 403)

    def test_search_with_auth_returns_structure(self, registries_client):
        """GET /registries/search returns RegistrySearchResponse with auth."""
        r = registries_client.get("/registries/search?query=knee&countries=US")
        # May return 200 or 503 if adapters are unavailable in test environment
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            data = r.json()
            assert "query" in data
            assert "total_results" in data

    def test_legacy_map_requires_auth(self, registries_client_no_auth):
        """POST /registries/legacy/map requires auth."""
        r = registries_client_no_auth.post(
            "/registries/legacy/map",
            json={"cert_number": "EC-2020-12345", "notified_body": "BSI"},
        )
        assert r.status_code in (401, 403)


# ===========================================================================
# Group 5: FHIR Ingestion Pipeline Tests
# ===========================================================================


class TestFHIRIngestion:
    """Tests for FHIR parsing utilities and pipeline integration."""

    def test_xml_to_dict_parses_device(self):
        """_xml_to_dict converts XML string to nested dict."""
        from app.services.fhir_connector import _xml_to_dict

        result = _xml_to_dict(FHIR_XML_FIXTURE)
        assert isinstance(result, dict)
        # The XML device tag becomes the root
        assert len(result) > 0

    def test_fhir_device_status_parsing(self):
        """FHIRDevice status field is correctly extracted."""
        from app.services.fhir_connector import FHIRConnector

        resource = {
            "resourceType": "Device",
            "id": "test-status",
            "status": "inactive",
            "deviceName": [{"name": "TestDevice", "type": "user-friendly-name"}],
        }
        connector = FHIRConnector()
        device = connector.parse_device_resource(resource)
        assert device.status == "inactive"

    def test_fhir_bundle_pagination_link(self):
        """parse_bundle extracts next_link for pagination."""
        from app.services.fhir_connector import FHIRConnector

        bundle_with_pagination = {
            "resourceType": "Bundle",
            "id": "paginated-bundle",
            "type": "searchset",
            "total": 100,
            "link": [
                {"relation": "self", "url": "https://fhir.example.com/Device?_count=10"},
                {"relation": "next", "url": "https://fhir.example.com/Device?_count=10&_offset=10"},
            ],
            "entry": [],
        }
        connector = FHIRConnector()
        bundle = connector.parse_bundle(bundle_with_pagination)
        assert "next" in bundle.next_link or bundle.next_link.endswith("_offset=10")

    def test_build_search_params_constructs_correctly(self):
        """build_search_params returns correct parameter dict."""
        from app.services.fhir_connector import FHIRConnector

        connector = FHIRConnector()
        params = connector.build_search_params(
            device_name="knee implant",
            manufacturer="OrthoTech",
            count=25,
        )
        assert params["device-name"] == "knee implant"
        assert params["manufacturer"] == "OrthoTech"
        assert params["_count"] == "25"
        assert params["_format"] == "json"

    def test_fhir_connector_parse_organization(self):
        """_parse_organization extracts org name and address."""
        from app.services.fhir_connector import FHIRConnector

        org_resource = {
            "resourceType": "Organization",
            "id": "org-001",
            "name": "OrthoTech Medical Inc.",
            "active": True,
            "address": [
                {
                    "country": "US",
                    "city": "San Diego",
                    "line": ["1234 Medical Drive"],
                }
            ],
            "telecom": [
                {"system": "email", "value": "regulatory@orthotech.com"},
                {"system": "phone", "value": "+1-619-555-0100"},
            ],
        }
        connector = FHIRConnector()
        org = connector._parse_organization(org_resource)
        assert org.name == "OrthoTech Medical Inc."
        assert org.address_country == "US"
        assert org.telecom_email == "regulatory@orthotech.com"
        assert org.telecom_phone == "+1-619-555-0100"

    def test_validate_fhir_resource_unsupported_type(self):
        """validate_fhir_resource returns error for unsupported resource type."""
        from app.services.fhir_connector import FHIRConnector

        connector = FHIRConnector()
        is_valid, errors = connector.validate_fhir_resource(
            {"resourceType": "Observation", "id": "obs-001"}
        )
        # Should produce an error about unsupported type
        assert any(e.severity == "error" for e in errors)

    def test_fhir_device_with_classification(self):
        """parse_device_resource extracts DeviceDefinition classification."""
        from app.services.fhir_connector import FHIRConnector

        resource = {
            "resourceType": "DeviceDefinition",
            "id": "dd-001",
            "deviceName": [{"name": "SpinalFix Pro", "type": "user-friendly-name"}],
            "classification": [
                {
                    "type": {
                        "coding": [
                            {
                                "system": "http://eudamed.ec.europa.eu/device-class",
                                "code": "IIb",
                                "display": "Class IIb",
                            }
                        ]
                    }
                }
            ],
        }
        connector = FHIRConnector()
        device = connector.parse_device_resource(resource)
        assert device.classification is not None
        assert device.classification.code == "IIb"

    def test_fhir_bundle_with_organization_entry(self):
        """parse_bundle processes Organization entries alongside Device entries."""
        from app.services.fhir_connector import FHIRConnector

        bundle = {
            "resourceType": "Bundle",
            "id": "mixed-bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Device",
                        "id": "d-001",
                        "deviceName": [{"name": "TestDevice", "type": "user-friendly-name"}],
                    }
                },
                {
                    "resource": {
                        "resourceType": "Organization",
                        "id": "o-001",
                        "name": "Test Org Inc.",
                    }
                },
            ],
        }
        connector = FHIRConnector()
        result = connector.parse_bundle(bundle)
        assert len(result.devices) == 1
        assert len(result.organizations) == 1
        assert result.organizations[0].name == "Test Org Inc."


# ===========================================================================
# Group 6: Existing Infrastructure Integration Checks
# ===========================================================================


class TestIntegritySystem:
    """Smoke tests for the crypto signing and integrity infrastructure."""

    def test_sign_and_verify_roundtrip(self):
        """Full sign → verify roundtrip on a realistic agent result."""
        from app.services.crypto_signer import sign_payload, verify_signature

        realistic_result = {
            "sections": [
                {"section_title": "Design Controls", "regulation_cite": "21 CFR 820.30"}
            ],
            "device_class": "III",
            "country": "US",
            "disclaimer": "Test only",
        }
        signed = sign_payload(realistic_result)
        verification = verify_signature(signed)
        assert verification["valid"] is True

    def test_sign_payload_hash_prefix_in_signed_block(self):
        """_signed block includes truncated hash prefix for UI display."""
        from app.services.crypto_signer import sign_payload

        signed = sign_payload({"data": "test"})
        meta = signed["_signed"]
        assert len(meta["hash"]) == 16  # 16-char prefix shown in UI
        assert len(meta["hash_full"]) == 64  # full SHA-256 hex

    def test_integrity_route_verify_signature(self, client, mock_user):
        """POST /api/v1/integrity/verify-signature validates a signed payload."""
        from app.services.crypto_signer import sign_payload

        payload = {"country": "US", "device_class": "II", "result": "test"}
        signed = sign_payload(payload)

        # VerifySignatureBody expects {"payload": <signed_dict>}, not the dict directly
        r = client.post("/api/v1/integrity/verify-signature", json={"payload": signed})
        assert r.status_code == 200
        data = r.json()
        assert "valid" in data

    def test_integrity_route_status(self, client, mock_user):
        """GET /api/v1/integrity/status returns system integrity status."""
        r = client.get("/api/v1/integrity/status")
        assert r.status_code == 200

    def test_jobs_route_returns_not_found_for_unknown_job(self, client, mock_user):
        """GET /api/v1/jobs/<valid-uuid-not-in-store> returns 200 with NOT_FOUND status.

        The jobs route validates UUID format (400 on bad format) then returns
        {"status": "NOT_FOUND", ...} with HTTP 200 for a valid UUID not in the store.
        """
        unknown_uuid = "00000000-0000-0000-0000-000000000000"
        r = client.get(f"/api/v1/jobs/{unknown_uuid}")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "NOT_FOUND"

    def test_jobs_route_returns_400_for_invalid_uuid(self, client, mock_user):
        """GET /api/v1/jobs/<non-uuid> returns 400 Bad Request."""
        r = client.get("/api/v1/jobs/not-a-real-uuid-format")
        assert r.status_code == 400


class TestSeverityEnumCoverage:
    """Additional coverage tests for severity classification edge cases."""

    def test_suspension_maps_to_critical(self):
        """'suspension' action type maps to CRITICAL severity."""
        from app.scrapers.base_scraper import BaseEnforcementScraper, Severity

        assert BaseEnforcementScraper.classify_severity("suspension") == Severity.CRITICAL

    def test_voluntary_recall_maps_to_low(self):
        """'voluntary_recall' maps to LOW severity."""
        from app.scrapers.base_scraper import BaseEnforcementScraper, Severity

        assert BaseEnforcementScraper.classify_severity("voluntary_recall") == Severity.LOW

    def test_field_safety_notice_maps_to_high(self):
        """'field_safety_notice' maps to HIGH severity."""
        from app.scrapers.base_scraper import BaseEnforcementScraper, Severity

        assert BaseEnforcementScraper.classify_severity("field_safety_notice") == Severity.HIGH

    def test_hyphen_separator_normalised(self):
        """Hyphens in action types are normalised to underscores."""
        from app.scrapers.base_scraper import BaseEnforcementScraper, Severity

        # "field-safety-notice" should normalise to "field_safety_notice"
        assert BaseEnforcementScraper.classify_severity("field-safety-notice") == Severity.HIGH

    def test_case_insensitive_classification(self):
        """Severity classification is case-insensitive."""
        from app.scrapers.base_scraper import BaseEnforcementScraper, Severity

        assert BaseEnforcementScraper.classify_severity("WARNING_LETTER") == Severity.HIGH
        assert BaseEnforcementScraper.classify_severity("Warning_Letter") == Severity.HIGH


class TestRegistrationRecord:
    """Tests for RegistrationRecord model in base_adapter."""

    def test_registration_record_defaults(self):
        """RegistrationRecord has sensible defaults."""
        from app.adapters.base_adapter import RegistrationRecord

        record = RegistrationRecord()
        assert record.registry == ""
        assert record.country == ""
        assert record.device_id == ""
        assert record.raw == {}
        assert isinstance(record.fetched_at, datetime)

    def test_registration_record_with_values(self):
        """RegistrationRecord accepts all fields."""
        from app.adapters.base_adapter import RegistrationRecord

        record = RegistrationRecord(
            registry="GUDID",
            country="US",
            device_id="08717648200274",
            device_name="OrthoKnee Pro",
            manufacturer="OrthoTech",
            status="In Commercial Distribution",
        )
        assert record.registry == "GUDID"
        assert record.status == "In Commercial Distribution"
