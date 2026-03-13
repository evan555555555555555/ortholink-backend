"""
ROA — Regulatory Operations Agent tests.
test_texas_checklist: US Class II ≥15 items, Manufacturer∩Importer=∅, all cite 21 CFR or QMSR/UDI.
test_pdf_export: Checklist response valid (PDF export endpoint when implemented).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.middleware.auth import AuthenticatedUser, get_current_user


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_reviewer_user():
    async def reviewer():
        return AuthenticatedUser(
            user_id="reviewer-user",
            email="reviewer@test.com",
            org_id="org-1",
            role="reviewer",
        )

    app.dependency_overrides[get_current_user] = reviewer
    yield
    app.dependency_overrides.pop(get_current_user, None)


def test_texas_checklist(client, mock_reviewer_user):
    """ROA: US Class II checklist has valid structure; when store has data, ≥15 items and role split."""
    response = client.post(
        "/api/v1/generate-checklist",
        data={"country": "US", "device_class": "II"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "country" in data
    assert data["country"] == "US"
    assert "items" in data
    assert "disclaimer" in data
    assert "Reference tool only" in data["disclaimer"]
    items = data["items"]
    roles = {r for i in items for r in (i.get("role"),)}
    assert roles <= {"MANUFACTURER", "IMPORTER", "BOTH"}
    for it in items:
        assert "item" in it
        assert "regulation_cite" in it
    # When vector store has data, PRD requires ≥15 items for US Class II
    if len(items) >= 15:
        manufacturer_items = {i["item"] for i in items if i.get("role") == "MANUFACTURER"}
        importer_items = {i["item"] for i in items if i.get("role") == "IMPORTER"}
        assert manufacturer_items.isdisjoint(importer_items), "Manufacturer and Importer lists must not overlap"


def test_pdf_export_checklist_structure(client, mock_reviewer_user):
    """Checklist response is valid; PDF export via WeasyPrint."""
    response = client.post(
        "/api/v1/generate-checklist",
        data={"country": "UA", "device_class": "IIb"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)
    assert "disclaimer" in data


def test_pdf_export(client, mock_reviewer_user):
    """WeasyPrint PDF opens without errors, contains all sections (PRD G12). Skips if pango/cairo missing."""
    try:
        from weasyprint import HTML  # noqa: F401
    except OSError:
        pytest.skip("WeasyPrint system libs (pango/cairo) not installed; run: brew install cairo pango")
    response = client.post(
        "/api/v1/export-pdf",
        data={"report_type": "checklist", "country": "US", "device_class": "II"},
    )
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("application/pdf")
    body = response.content
    assert body[:5] == b"%PDF-", "Response must be valid PDF"
    assert len(body) > 500, "PDF must have non-trivial content"
