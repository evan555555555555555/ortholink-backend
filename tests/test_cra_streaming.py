"""
CRA — Compliance Review Agent tests.
test_cra_streaming: SSE stream produces valid AIComment[] with severity + citations (or structured refusal).
"""

import json
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


def test_cra_streaming(client, mock_reviewer_user):
    """SSE stream produces valid response: job_id and either comments (with severity + citation) or refused."""
    # Minimal text file for review
    body = b"This is a quality management system procedure. We use ALARP for risk assessment."
    response = client.post(
        "/api/v1/review-document",
        files={"file": ("doc.txt", body, "text/plain")},
        data={"standard": "FDA 21 CFR 820", "country": "US"},
    )
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    # May be SSE or JSON (if refused)
    if "text/event-stream" in content_type:
        lines = response.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data:")]
        assert len(data_lines) >= 1
        first = json.loads(data_lines[0].replace("data: ", ""))
        assert "job_id" in first
        for line in data_lines[1:]:
            try:
                obj = json.loads(line.replace("data: ", ""))
                if "comment" in obj:
                    c = obj["comment"]
                    assert "severity" in c
                    assert c["severity"] in ("CRITICAL", "MAJOR", "MINOR", "INFO")
                    assert "citation" in c or "standard_reference" in c
            except json.JSONDecodeError:
                pass
    else:
        data = response.json()
        assert "job_id" in data
        if data.get("refused"):
            assert "refusal_reason" in data
            assert "disclaimer" in data
        else:
            assert "comments" in data or "disclaimer" in data
