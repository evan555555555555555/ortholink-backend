"""
OrthoLink Auth Routes
POST /api/v1/auth/signup      — create account via Supabase (email+password).
POST /api/v1/auth/login       — sign in via Supabase (email+password → JWT).
POST /api/v1/auth/magic-link  — trigger Supabase magic link for login.
GET  /api/v1/auth/me           — return current user info from JWT.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field

from app.middleware.auth import AuthenticatedUser, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])

logger = logging.getLogger(__name__)


# ── Signup ───────────────────────────────────────────────────────────────────


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class AuthTokenResponse(BaseModel):
    access_token: str
    refresh_token: str = ""
    user_id: str = ""
    email: str = ""


@router.post("/signup", response_model=AuthTokenResponse)
async def signup(request: SignupRequest):
    """
    Create a new user account via Supabase email+password.
    PRD: POST /api/v1/auth/signup → Supabase session.
    """
    try:
        from app.services.supabase_client import get_supabase_client
        client = get_supabase_client()
        result = client.auth.sign_up({"email": request.email, "password": request.password})
        session = getattr(result, "session", None)
        user = getattr(result, "user", None)
        if session:
            return AuthTokenResponse(
                access_token=session.access_token,
                refresh_token=getattr(session, "refresh_token", ""),
                user_id=getattr(user, "id", "") if user else "",
                email=request.email,
            )
        # Supabase may return user without session if email confirmation is required
        return AuthTokenResponse(
            access_token="",
            refresh_token="",
            user_id=getattr(user, "id", "") if user else "",
            email=request.email,
        )
    except Exception as e:
        logger.warning(f"Signup failed: {e}")
        raise HTTPException(status_code=400, detail="Signup failed. Please check your details and try again.")


# ── Login ────────────────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


@router.post("/login", response_model=AuthTokenResponse)
async def login(request: LoginRequest):
    """
    Sign in with email+password via Supabase.
    PRD: POST /api/v1/auth/login → JWT tokens.
    """
    try:
        from app.services.supabase_client import get_supabase_client
        client = get_supabase_client()
        result = client.auth.sign_in_with_password({"email": request.email, "password": request.password})
        session = getattr(result, "session", None)
        user = getattr(result, "user", None)
        if not session:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return AuthTokenResponse(
            access_token=session.access_token,
            refresh_token=getattr(session, "refresh_token", ""),
            user_id=getattr(user, "id", "") if user else "",
            email=request.email,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Login failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid email or password")


class MagicLinkRequest(BaseModel):
    email: EmailStr


class MagicLinkResponse(BaseModel):
    sent: bool
    message: str


@router.post("/magic-link", response_model=MagicLinkResponse)
async def send_magic_link(request: MagicLinkRequest):
    """
    Trigger Supabase magic link for passwordless login.
    PRD: POST /api/v1/auth/magic-link → triggers Supabase magic link.
    """
    try:
        from app.services.supabase_client import get_supabase_client
        client = get_supabase_client()
        result = client.auth.sign_in_with_otp({"email": request.email})
        if result and getattr(result, "session", None) is None:
            return MagicLinkResponse(sent=True, message="If an account exists, a magic link was sent.")
        return MagicLinkResponse(sent=True, message="Magic link sent to your email.")
    except Exception as e:
        logger.warning(f"Magic link request failed: {e}")
        return MagicLinkResponse(
            sent=False,
            message="Could not send magic link. Please try again or contact support.",
        )


class MeResponse(BaseModel):
    user_id: str
    email: str
    org_id: Optional[str] = None
    role: Optional[str] = None


@router.get("/me", response_model=MeResponse)
async def get_me(user: AuthenticatedUser = Depends(get_current_user)):
    """Return current authenticated user's profile from JWT claims."""
    return MeResponse(
        user_id=user.user_id,
        email=user.email,
        org_id=user.org_id,
        role=user.role,
    )
