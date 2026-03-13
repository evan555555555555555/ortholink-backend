"""
Tests for the 5 new regulatory department agents:
TDA (Technical Documentation), PMS (Post-Market Surveillance),
CAPA (Corrective/Preventive Action), GCO (Global Compliance Orchestrator), Verify (Truth Checker).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.middleware.auth import AuthenticatedUser, get_current_user


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=False)
def mock_reviewer(client):
    async def reviewer():
        return AuthenticatedUser(
            user_id="test-reviewer",
            email="reviewer@test.com",
            org_id="org-test",
            role="reviewer",
        )
    app.dependency_overrides[get_current_user] = reviewer
    yield
    app.dependency_overrides.pop(get_current_user, None)


# ── TDA ─────────────────────────────────────────────────────────────────────

class TestTDA:
    def test_returns_202_async(self, client, mock_reviewer):
        """TDA: async_mode=True returns 202 with job_id."""
        r = client.post(
            "/api/v1/technical-dossier",
            data={"country": "US", "device_class": "II", "async_mode": "true"},
        )
        assert r.status_code == 202
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["country"] == "US"

    def test_sync_returns_dossier(self, client, mock_reviewer):
        """TDA: synchronous mode returns TechnicalDossierPlan structure (via job store)."""
        r = client.post(
            "/api/v1/technical-dossier",
            data={"country": "EU", "device_class": "IIb", "async_mode": "false"},
        )
        assert r.status_code == 200
        outer = r.json()
        # Sync path returns job-store-wrapped payload: {result: {...}, status: "completed"}
        data = outer.get("result", outer)
        assert "sections" in data
        assert isinstance(data["sections"], list)
        assert "device_class" in data
        assert data["device_class"] == "IIb"
        assert "disclaimer" in data
        # Each section should have required fields
        for section in data["sections"]:
            assert "section_title" in section
            assert "regulation_cite" in section
            assert "required" in section

    def test_requires_country(self, client, mock_reviewer):
        """TDA: missing country returns 422."""
        r = client.post(
            "/api/v1/technical-dossier",
            data={"device_class": "II", "async_mode": "false"},
        )
        assert r.status_code == 422

    def test_requires_auth(self, client):
        """TDA: unauthenticated returns 401 or 403."""
        r = client.post(
            "/api/v1/technical-dossier",
            data={"country": "US", "device_class": "II"},
        )
        assert r.status_code in (401, 403)


# ── PMS ─────────────────────────────────────────────────────────────────────

class TestPMS:
    def test_returns_202_async(self, client, mock_reviewer):
        """PMS: async_mode=True returns 202 with job_id."""
        r = client.post(
            "/api/v1/pms-plan",
            data={"country": "EU", "device_class": "III", "async_mode": "true"},
        )
        assert r.status_code == 202
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_sync_returns_pms_plan(self, client, mock_reviewer):
        """PMS: synchronous mode returns PMSPlan structure (via job store)."""
        r = client.post(
            "/api/v1/pms-plan",
            data={"country": "US", "device_class": "II", "async_mode": "false"},
        )
        assert r.status_code == 200
        outer = r.json()
        data = outer.get("result", outer)
        assert "activities" in data
        assert "reporting_requirements" in data
        assert isinstance(data["activities"], list)
        assert isinstance(data["reporting_requirements"], list)
        assert "country" in data
        assert "disclaimer" in data
        for activity in data["activities"]:
            assert "activity" in activity
            assert "frequency" in activity
            assert "regulation_cite" in activity

    def test_requires_country(self, client, mock_reviewer):
        r = client.post("/api/v1/pms-plan", data={"device_class": "II", "async_mode": "false"})
        assert r.status_code == 422

    def test_requires_auth(self, client):
        r = client.post("/api/v1/pms-plan", data={"country": "US", "device_class": "II"})
        assert r.status_code in (401, 403)


# ── CAPA ────────────────────────────────────────────────────────────────────

class TestCAPA:
    def test_returns_202_async(self, client, mock_reviewer):
        """CAPA: async_mode=True returns 202 with job_id."""
        r = client.post(
            "/api/v1/capa",
            data={
                "problem_statement": "Customer complaint: implant loosening reported after 6 months",
                "country": "US",
                "device_class": "III",
                "async_mode": "true",
            },
        )
        assert r.status_code == 202
        data = r.json()
        assert "job_id" in data

    def test_sync_returns_capa_analysis(self, client, mock_reviewer):
        """CAPA: synchronous mode returns CAPAAnalysis structure (via job store)."""
        r = client.post(
            "/api/v1/capa",
            data={
                "problem_statement": "Device labeling error: incorrect sterilization instructions",
                "country": "EU",
                "device_class": "IIb",
                "async_mode": "false",
            },
        )
        assert r.status_code == 200
        outer = r.json()
        data = outer.get("result", outer)
        assert "root_cause_categories" in data
        assert "corrective_actions" in data
        assert "regulatory_obligations" in data
        assert "requires_regulatory_notification" in data
        assert isinstance(data["root_cause_categories"], list)
        assert isinstance(data["corrective_actions"], list)
        assert "severity" in data
        assert data["severity"] in ("Critical", "Major", "Minor")
        for action in data["corrective_actions"]:
            assert "action_type" in action
            assert "description" in action
            assert "regulation_cite" in action

    def test_requires_problem_statement(self, client, mock_reviewer):
        r = client.post(
            "/api/v1/capa",
            data={"problem_statement": "", "country": "US", "device_class": "II", "async_mode": "false"},
        )
        assert r.status_code == 422

    def test_requires_auth(self, client):
        r = client.post(
            "/api/v1/capa",
            data={"problem_statement": "test", "country": "US", "device_class": "II"},
        )
        assert r.status_code in (401, 403)


# ── GCO (Global Compliance Orchestrator) ─────────────────────────────────────

class TestGco:
    def test_returns_202_async(self, client, mock_reviewer):
        """GCO: async_mode=True returns 202 with job_id and agent list."""
        r = client.post(
            "/api/v1/gco-analysis",
            data={"country": "US", "device_class": "II", "async_mode": "true"},
        )
        assert r.status_code == 202
        data = r.json()
        assert "job_id" in data
        assert "agents" in data
        assert "TDA" in data["agents"]
        assert "PMS" in data["agents"]
        assert "ROA" in data["agents"]

    def test_includes_capa_when_problem_given(self, client, mock_reviewer):
        """GCO: CAPA agent included when problem_statement provided."""
        r = client.post(
            "/api/v1/gco-analysis",
            data={
                "country": "EU",
                "device_class": "IIb",
                "problem_statement": "adverse event reported",
                "async_mode": "true",
            },
        )
        assert r.status_code == 202
        data = r.json()
        assert "CAPA" in data["agents"]

    def test_requires_auth(self, client):
        r = client.post(
            "/api/v1/gco-analysis",
            data={"country": "US", "device_class": "II"},
        )
        assert r.status_code in (401, 403)


# ── VERIFY ────────────────────────────────────────────────────────────────────

class TestVerify:
    def test_verifies_valid_claims(self, client, mock_reviewer):
        """Verify: returns VerificationReport with per-claim verdicts."""
        claims = "\n".join([
            "Manufacturers must maintain a quality management system",
            "Medical devices must be registered before market placement",
            "Post-market surveillance is required for all device classes",
        ])
        r = client.post(
            "/api/v1/verify-claims",
            data={"claims_text": claims, "country": "US", "device_class": "II"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "claim_results" in data
        assert "overall_verdict" in data
        assert "overall_confidence" in data
        assert data["total_claims"] == 3
        assert data["overall_verdict"] in ("RELIABLE", "REVIEW_REQUIRED", "UNRELIABLE")
        assert 0.0 <= data["overall_confidence"] <= 1.0
        for result in data["claim_results"]:
            assert "claim" in result
            assert "verdict" in result
            assert result["verdict"] in ("VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED", "CONTRADICTED")
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0

    def test_requires_claims(self, client, mock_reviewer):
        """Verify: empty claims text returns 422."""
        r = client.post(
            "/api/v1/verify-claims",
            data={"claims_text": "", "country": "US", "device_class": "II"},
        )
        assert r.status_code == 422

    def test_caps_at_20_claims(self, client, mock_reviewer):
        """Verify: more than 20 claims are silently truncated to 20."""
        claims = "\n".join([f"Claim number {i}" for i in range(25)])
        r = client.post(
            "/api/v1/verify-claims",
            data={"claims_text": claims, "country": "EU", "device_class": "IIb"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["total_claims"] <= 20

    def test_requires_auth(self, client):
        r = client.post(
            "/api/v1/verify-claims",
            data={"claims_text": "test claim", "country": "US", "device_class": "II"},
        )
        assert r.status_code in (401, 403)

    def test_synchronous_fast_response(self, client, mock_reviewer):
        """Verify endpoint is synchronous (no job_id needed)."""
        r = client.post(
            "/api/v1/verify-claims",
            data={
                "claims_text": "Manufacturers of Class II devices must obtain 510(k) clearance",
                "country": "US",
                "device_class": "II",
            },
        )
        assert r.status_code == 200
        # Not async — returns result directly, no job_id
        data = r.json()
        assert "job_id" in data  # VerificationReport has job_id field
        assert "claim_results" in data


# ── RMA — Risk Management Agent (ISO 14971:2019) ─────────────────────────────


class TestRMA:
    def test_returns_202_async(self, client, mock_reviewer):
        """RMA: async_mode=True returns 202 with job_id and country/class."""
        r = client.post(
            "/api/v1/risk-analysis",
            data={
                "device_description": "Titanium tibial knee implant with UHMWPE bearing",
                "intended_use": "Total knee replacement in skeletally mature adults",
                "country": "US",
                "device_class": "III",
                "async_mode": "true",
            },
        )
        assert r.status_code == 202
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["country"] == "US"

    def test_sync_returns_risk_report(self, client, mock_reviewer):
        """RMA: synchronous mode returns RiskManagementReport structure via job store."""
        r = client.post(
            "/api/v1/risk-analysis",
            data={
                "device_description": "Silicone breast implant for cosmetic augmentation",
                "intended_use": "Aesthetic breast augmentation in adult women",
                "country": "EU",
                "device_class": "IIb",
                "async_mode": "false",
            },
        )
        assert r.status_code == 200
        outer = r.json()
        data = outer.get("result", outer)

        # Required top-level fields
        assert "hazard_analysis" in data
        assert "overall_verdict" in data
        assert "total_hazards" in data
        assert "device_class" in data
        assert "applicable_standards" in data
        assert "disclaimer" in data

        # Verdict must be valid
        assert data["overall_verdict"] in ("ACCEPTABLE", "ALARP", "UNACCEPTABLE")

        # Hazard table structure
        assert isinstance(data["hazard_analysis"], list)
        assert len(data["hazard_analysis"]) >= 1

        for hazard in data["hazard_analysis"]:
            # Required hazard fields
            assert "hazard_id" in hazard
            assert "hazard" in hazard
            assert "hazardous_situation" in hazard
            assert "harm" in hazard
            # Severity must be 1-5
            assert 1 <= hazard["severity"] <= 5
            # Probability must be 1-5
            assert 1 <= hazard["probability"] <= 5
            # Risk level must be valid
            assert hazard["risk_level"] in ("ACCEPTABLE", "ALARP", "UNACCEPTABLE")
            # Residual risk must be valid
            assert hazard["residual_risk_level"] in ("ACCEPTABLE", "ALARP", "UNACCEPTABLE")
            # Controls must be present
            assert "control_measures" in hazard
            assert isinstance(hazard["control_measures"], list)

    def test_risk_matrix_computed_server_side(self, client, mock_reviewer):
        """
        RMA: risk_level is always computed from canonical ISO 14971 matrix,
        never from LLM output. Verify the computation is deterministic.
        """
        r = client.post(
            "/api/v1/risk-analysis",
            data={
                "device_description": "Spinal fixation rod",
                "intended_use": "Thoracolumbar spinal stabilisation in adults",
                "country": "AU",
                "device_class": "III",
                "async_mode": "false",
            },
        )
        assert r.status_code == 200
        outer = r.json()
        data = outer.get("result", outer)

        for hazard in data["hazard_analysis"]:
            sev = hazard["severity"]
            prob = hazard["probability"]
            expected_score = sev * prob
            assert hazard["risk_score"] == expected_score, (
                f"risk_score should be sev×prob={expected_score} "
                f"but got {hazard['risk_score']}"
            )

    def test_requires_device_description(self, client, mock_reviewer):
        """RMA: empty device_description returns 422."""
        r = client.post(
            "/api/v1/risk-analysis",
            data={
                "device_description": "",
                "intended_use": "Some intended use",
                "country": "US",
                "device_class": "II",
                "async_mode": "false",
            },
        )
        assert r.status_code == 422

    def test_requires_intended_use(self, client, mock_reviewer):
        """RMA: empty intended_use returns 422."""
        r = client.post(
            "/api/v1/risk-analysis",
            data={
                "device_description": "Some device",
                "intended_use": "",
                "country": "US",
                "device_class": "II",
                "async_mode": "false",
            },
        )
        assert r.status_code == 422

    def test_requires_auth(self, client):
        """RMA: unauthenticated returns 401 or 403."""
        r = client.post(
            "/api/v1/risk-analysis",
            data={
                "device_description": "test",
                "intended_use": "test",
                "country": "US",
                "device_class": "II",
            },
        )
        assert r.status_code in (401, 403)

    def test_hazards_hint_accepted(self, client, mock_reviewer):
        """RMA: optional hazards_hint parameter accepted without error."""
        r = client.post(
            "/api/v1/risk-analysis",
            data={
                "device_description": "Cobalt-chrome hip implant",
                "intended_use": "Total hip arthroplasty",
                "country": "JP",
                "device_class": "III",
                "hazards_hint": "metal ion release, fretting corrosion, dislocation",
                "async_mode": "true",
            },
        )
        assert r.status_code == 202
        assert "job_id" in r.json()

    def test_briefing_routes_exist(self, client, mock_reviewer):
        """Briefing: POST /run returns a brief with required fields."""
        r = client.post("/api/v1/briefing/run")
        assert r.status_code == 200
        data = r.json()
        assert "risk_level" in data
        assert "alert_summary" in data
        assert "coverage_audit" in data
        assert data["risk_level"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        assert "verdict" in data["coverage_audit"]

    def test_briefing_latest_404_initially(self, client, mock_reviewer):
        """Briefing: GET /latest returns 200 or 404 (never 500)."""
        r = client.get("/api/v1/briefing/latest")
        assert r.status_code in (200, 404)

    def test_briefing_coverage_returns_audit(self, client, mock_reviewer):
        """Briefing: GET /coverage returns Reality Checker verdict."""
        r = client.get("/api/v1/briefing/coverage")
        assert r.status_code == 200
        data = r.json()
        assert "verdict" in data
        assert data["verdict"] in (
            "COVERAGE_OK", "LOW_COVERAGE", "CRITICAL_GAPS", "AUDIT_FAILED"
        )
