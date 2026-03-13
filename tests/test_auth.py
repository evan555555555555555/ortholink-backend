"""
Tests for authentication middleware.
"""

import time

import jwt
import pytest
from fastapi import HTTPException

from app.middleware.auth import AuthenticatedUser, verify_jwt

# Test JWT secret for unit tests
TEST_JWT_SECRET = "test-jwt-secret-for-unit-tests-only"


def _make_token(
    user_id: str = "test-user-123",
    email: str = "test@example.com",
    org_id: str = "test-org-456",
    role: str = "admin",
    secret: str = TEST_JWT_SECRET,
    expired: bool = False,
) -> str:
    """Create a test JWT token."""
    now = int(time.time())
    payload = {
        "sub": user_id,
        "email": email,
        "aud": "authenticated",
        "iat": now,
        "exp": now - 3600 if expired else now + 3600,
        "user_metadata": {
            "org_id": org_id,
            "role": role,
        },
    }
    return jwt.encode(payload, secret, algorithm="HS256")


class TestVerifyJWT:
    """Test JWT token verification."""

    def test_valid_token(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_JWT_SECRET", TEST_JWT_SECRET)
        # Clear cached settings
        from app.core.config import get_settings
        get_settings.cache_clear()

        token = _make_token()
        payload = verify_jwt(token)

        assert payload["sub"] == "test-user-123"
        assert payload["email"] == "test@example.com"

    def test_expired_token(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_JWT_SECRET", TEST_JWT_SECRET)
        from app.core.config import get_settings
        get_settings.cache_clear()

        token = _make_token(expired=True)

        with pytest.raises(HTTPException) as exc_info:
            verify_jwt(token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_invalid_token(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_JWT_SECRET", TEST_JWT_SECRET)
        from app.core.config import get_settings
        get_settings.cache_clear()

        with pytest.raises(HTTPException) as exc_info:
            verify_jwt("invalid-token-string")
        assert exc_info.value.status_code == 401

    def test_wrong_secret(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_JWT_SECRET", TEST_JWT_SECRET)
        from app.core.config import get_settings
        get_settings.cache_clear()

        token = _make_token(secret="wrong-secret")

        with pytest.raises(HTTPException) as exc_info:
            verify_jwt(token)
        assert exc_info.value.status_code == 401

    def test_missing_jwt_secret(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_JWT_SECRET", "")
        from app.core.config import get_settings
        get_settings.cache_clear()

        token = _make_token()
        with pytest.raises(HTTPException) as exc_info:
            verify_jwt(token)
        assert exc_info.value.status_code == 500


class TestAuthenticatedUser:
    """Test AuthenticatedUser model."""

    def test_create_user(self):
        user = AuthenticatedUser(
            user_id="user-1",
            email="test@example.com",
            org_id="org-1",
            role="admin",
        )
        assert user.user_id == "user-1"
        assert user.email == "test@example.com"
        assert user.org_id == "org-1"
        assert user.role == "admin"

    def test_default_values(self):
        user = AuthenticatedUser(
            user_id="user-1",
            email="test@example.com",
        )
        assert user.org_id is None
        assert user.role is None
