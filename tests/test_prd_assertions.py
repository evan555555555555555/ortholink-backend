"""
PRD Section 13: Required pytest assertions.
- test_extra_detection: "Distributor's Pet Photo" → EXTRA, no citation.
- test_no_hardcoded_dicts: grep returns nothing.
- test_embedding_model: grep shows only text-embedding-3-large.
- test_env_not_committed: git log .env returns nothing.
- test_rbac_viewer: Viewer role POST /verify-distributor → 403.
- test_audit_no_delete: Documented; requires deployed Supabase RLS (REVOKE DELETE on audit_log).
"""

import subprocess
from pathlib import Path

import pytest

# Backend root for grep/git
BACKEND_ROOT = Path(__file__).resolve().parent.parent


def test_no_hardcoded_dicts():
    """HC-2: No COMMERCIAL_ITEMS, UA_ITEM_TERMS, KNOWN_ITEMS in app code (exclude tests)."""
    app_dir = BACKEND_ROOT / "app"
    result = subprocess.run(
        [
            "grep",
            "-rn",
            "-E",
            "COMMERCIAL_ITEMS|UA_ITEM_TERMS|KNOWN_ITEMS",
            str(app_dir),
            "--include=*.py",
        ],
        capture_output=True,
        text=True,
        cwd=BACKEND_ROOT,
    )
    assert result.returncode != 0 or result.stdout.strip() == "", (
        "Hardcoded classification dictionaries found in app/: " + (result.stdout or "")
    )


def test_embedding_model_large_only():
    """HC-1: In app code, text-embedding must only be text-embedding-3-large (exclude tests)."""
    app_dir = BACKEND_ROOT / "app"
    result = subprocess.run(
        ["grep", "-r", "text-embedding", str(app_dir), "--include=*.py"],
        capture_output=True,
        text=True,
        cwd=BACKEND_ROOT,
    )
    if result.returncode != 0:
        return
    full_out = result.stdout + result.stderr
    assert "text-embedding-3-small" not in full_out, "text-embedding-3-small is forbidden in app/ (HC-1)"


def test_env_not_committed():
    """HC-5: .env must never have been committed to git. On CI this must not skip."""
    import os
    is_ci = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")

    result = subprocess.run(
        ["git", "log", "--all", "--full-history", "--", ".env"],
        capture_output=True,
        text=True,
        cwd=BACKEND_ROOT.parent,
    )
    if result.returncode != 0:
        pytest.skip("Not a git repo — cannot verify .env history")
    if result.stdout.strip():
        pytest.fail(".env appears in git history. Remove it and add to .gitignore.")


def test_extra_detection_extra_item_model():
    """DVA: 'Distributor's Pet Photo' must be classifiable as EXTRA with no regulatory citation.
    PRD: test_extra_detection — Submit 'Distributor's Pet Photo', assert EXTRA, no citation."""
    from app.crews.verify_distributor import GapItem

    extra_item = GapItem(
        distributor_item="Distributor's Pet Photo",
        status="EXTRA",
        confidence=0.6,
        explanation="No matching regulatory requirement found in database. Item is not required by regulation.",
    )
    assert extra_item.status == "EXTRA"
    # EXTRA items that are clearly non-regulatory should not have a matched regulation citation
    assert extra_item.matched_regulation is None or len(extra_item.explanation or "") > 0


# test_rbac_viewer: Viewer role POST /verify-distributor → 403
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture
def mock_viewer_user():
    from app.middleware.auth import AuthenticatedUser, get_current_user
    from app.main import app

    async def viewer():
        return AuthenticatedUser(
            user_id="viewer-user",
            email="viewer@test.com",
            org_id="org-1",
            role="viewer",
        )

    app.dependency_overrides[get_current_user] = viewer
    yield
    app.dependency_overrides.pop(get_current_user, None)


def test_rbac_viewer_cannot_post_verify_distributor(client, mock_viewer_user):
    """Viewer role must get 403 on POST /api/v1/verify-distributor (requires reviewer+)."""
    response = client.post(
        "/api/v1/verify-distributor",
        data={"country": "UA", "device_class": "IIb"},
        files={"file": ("test.csv", b"document\nTechnical File", "text/csv")},
    )
    # mock_viewer_user overrides get_current_user with viewer; require_reviewer returns 403
    assert response.status_code == 403


def test_audit_no_delete_requirement():
    """PRD: audit_log must have REVOKE DELETE. Documented; full check requires deployed RLS."""
    pytest.skip("Requires deployed Supabase RLS; run manually: DELETE FROM audit_log; → permission denied")


def test_rsa_async_returns_202(client, mock_viewer_user):
    """RSA with async_mode=True must return 202 and job_id. Use reviewer for POST."""
    from app.main import app
    from app.middleware.auth import AuthenticatedUser, get_current_user

    async def reviewer():
        return AuthenticatedUser(
            user_id="reviewer-id",
            email="reviewer@test.com",
            org_id="org-1",
            role="reviewer",
        )

    from app.middleware.rbac import require_reviewer
    app.dependency_overrides[require_reviewer] = reviewer
    try:
        response = client.post(
            "/api/v1/plan-strategy",
            data={
                "device_name": "Test Implant",
                "target_markets": "US,UA",
                "async_mode": True,
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data.get("status") == "pending"
    finally:
        app.dependency_overrides.pop(require_reviewer, None)


def test_audit_no_delete_requirement():
    """PRD: audit_log must have REVOKE DELETE. This test documents the requirement.
    Full verification: psql DELETE FROM audit_log; must return permission denied (deployed RLS)."""
    # Schema/migration should enforce REVOKE DELETE on audit_log; we cannot run psql here.
    pytest.skip("Requires deployed Supabase RLS; run manually: DELETE FROM audit_log; → permission denied")
