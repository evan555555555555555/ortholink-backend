"""
OrthoLink Auth Middleware
JWT verification via Supabase.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


def create_test_jwt(
    org_id: str = "test-org",
    user_id: str = "test-user",
    role: str = "reviewer",
    email: str = "test@ortholink.local",
) -> str:
    """
    Create a JWT that the server will accept when SUPABASE_JWT_SECRET is set.
    For acceptance tests: run server with same .env, then export ORTHOLINK_TEST_JWT from this.
    """
    settings = get_settings()
    if not settings.supabase_jwt_secret:
        raise ValueError("SUPABASE_JWT_SECRET not set. Set it in .env for acceptance tests.")
    now = int(time.time())
    exp_ts = int((datetime.now(timezone.utc) + timedelta(days=36500)).timestamp())
    payload = {
        "sub": user_id,
        "email": email,
        "aud": "authenticated",
        "iat": now,
        "exp": exp_ts,
        "user_metadata": {"org_id": org_id, "role": role},
    }
    return jwt.encode(payload, settings.supabase_jwt_secret, algorithm="HS256")


class AuthenticatedUser:
    """Represents an authenticated user extracted from JWT."""

    def __init__(
        self,
        user_id: str,
        email: str,
        org_id: Optional[str] = None,
        role: Optional[str] = None,
    ):
        self.user_id = user_id
        self.email = email
        self.org_id = org_id
        self.role = role


def verify_jwt(token: str) -> dict:
    """
    Verify a Supabase JWT token and return the payload.
    Raises HTTPException if token is invalid.
    """
    settings = get_settings()

    if not settings.supabase_jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT secret not configured",
        )

    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
            options={
                "require": ["exp", "iat", "sub", "aud"],
                "leeway": 0,                        # No clock drift tolerance
                "verify_exp": True,
                "verify_iat": True,
                "verify_aud": True,
            },
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        # Do NOT leak internal JWT error details (timing / enumeration attacks)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or malformed token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> AuthenticatedUser:
    """
    FastAPI dependency to extract and verify the current user from JWT.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_jwt(credentials.credentials)

    user_id = payload.get("sub")
    email = payload.get("email", "")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID",
        )

    # Extract org_id and role from app_metadata or user_metadata
    app_metadata = payload.get("app_metadata", {})
    user_metadata = payload.get("user_metadata", {})

    org_id = app_metadata.get("org_id") or user_metadata.get("org_id")
    role = app_metadata.get("role") or user_metadata.get("role", "viewer")

    return AuthenticatedUser(
        user_id=user_id,
        email=email,
        org_id=org_id,
        role=role,
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[AuthenticatedUser]:
    """
    FastAPI dependency that returns the current user if authenticated, or None.
    Used for endpoints that work both authenticated and unauthenticated.
    """
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
