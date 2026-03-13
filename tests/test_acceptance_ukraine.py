"""
PRD §3.1 — Ukraine DVA acceptance test.
Hits LIVE server (http://localhost:8000) with real FAISS and real OpenAI.
Requires: vector store populated with UA data, server running, ORTHOLINK_TEST_JWT set.
"""

import os
import httpx
import pytest

BASE_URL = os.environ.get("ORTHOLINK_ACCEPTANCE_BASE_URL", "http://localhost:8000")

UKRAINE_CSV = """document
Bank Statement
Certificate of Free Sale
Technical File
ISO 13485 Certificate
Instructions for Use
Declaration of Conformity
Clinical Evaluation Report
Pet Photo of Distributor's Cat
Letter from the Mayor
Distributor Agreement
Quality Manual
Risk Management File
Post-Market Surveillance Plan
Labeling Samples"""


def _get_test_jwt() -> str:
    """JWT that the live server will accept (reviewer role). Set ORTHOLINK_TEST_JWT or create via auth."""
    jwt = os.environ.get("ORTHOLINK_TEST_JWT", "").strip()
    if not jwt:
        pytest.skip(
            "ORTHOLINK_TEST_JWT not set. Start server with test secret and set ORTHOLINK_TEST_JWT for acceptance test."
        )
    return jwt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ukraine_dva_acceptance():
    """PRD §3.1: Real Ukraine DVA with real FAISS data and real OpenAI calls."""
    token = _get_test_jwt()
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            f"{BASE_URL}/api/v1/verify-distributor",
            data={"country": "UA", "device_class": "IIb"},
            files={"file": ("ukraine.csv", UKRAINE_CSV.encode(), "text/csv")},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200, response.text
    data = response.json()

    items = {i["distributor_item"]: i for i in data["items"]}

    # PRD critical semantic test
    assert items["Bank Statement"]["status"] == "REQUIRED"
    assert "753" in items["Bank Statement"]["citation"]
    assert items["Bank Statement"]["confidence"] >= 0.82

    # PRD critical classification test — previous contractor failed this
    assert items["Certificate of Free Sale"]["status"] == "REQUIRED"

    # Obvious EXTRA items
    assert items["Pet Photo of Distributor's Cat"]["status"] == "EXTRA"
    assert items["Pet Photo of Distributor's Cat"].get("citation", "") == ""
    assert items["Letter from the Mayor"]["status"] == "EXTRA"

    # MISSING items must exist (real regulations require more than submitted)
    assert len(data.get("missing_requirements", [])) > 0

    # Fraud risk must be non-zero (we submitted EXTRA items)
    assert data["summary"]["fraud_risk_score"] > 0

    # Disclaimer present
    assert "Verify with official sources" in data["disclaimer"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_country_isolation_real_data():
    """Ukraine query returns 0 FDA chunks. FDA query returns 0 Ukraine chunks."""
    from app.tools.vector_store import get_vector_store

    store = get_vector_store()
    # Queries that match ingested content so we get results per country
    ua_results = store.search(
        "Certificate of Free Sale bank statement proof financial",
        country="UA",
        top_k=10,
    )
    us_results = store.search(
        "device labeling 21 CFR reporting",
        country="US",
        top_k=10,
    )

    for r in ua_results:
        assert r["country"] == "UA", "UA query returned non-UA chunk"
    for r in us_results:
        assert r["country"] == "US", "US query returned non-US chunk"

    assert len(ua_results) > 0, "No Ukraine chunks found — ingestion may have failed"
    assert len(us_results) > 0, "No US chunks found — ingestion may have failed"
