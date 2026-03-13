"""
Billing routes — Stripe Checkout + Webhook + Subscription Status.

POST /api/v1/billing/checkout          — create Stripe checkout session
GET  /api/v1/billing/subscription      — current subscription status for org
GET  /api/v1/billing/plans             — list available plans and pricing
POST /api/v1/billing/webhook           — Stripe webhook receiver (no auth)
"""

import logging

from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from app.middleware.auth import AuthenticatedUser, get_current_user
from app.services.billing_service import (
    create_checkout_session,
    get_subscription_status,
    handle_webhook,
    list_plans,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["Billing"])


# ── Request schemas ───────────────────────────────────────────────────────────

_ALLOWED_REDIRECT_HOSTS = frozenset({"app.ortholink.ai", "localhost", "127.0.0.1"})


class CheckoutBody(BaseModel):
    plan: str
    success_url: str = "https://app.ortholink.ai/billing/success"
    cancel_url: str = "https://app.ortholink.ai/billing/cancel"

    @field_validator("success_url", "cancel_url")
    @classmethod
    def _validate_redirect_url(cls, v: str) -> str:
        """Prevent open redirect: only allow known safe hosts."""
        parsed = urlparse(v)
        if parsed.hostname not in _ALLOWED_REDIRECT_HOSTS:
            raise ValueError(f"Redirect URL host not allowed: {parsed.hostname!r}")
        return v


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/plans")
def get_plans():
    """List available plans and pricing. No auth required."""
    return {"plans": list_plans()}


@router.post("/checkout")
def checkout(
    body: CheckoutBody,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """
    Create a Stripe Checkout session for the authenticated user's org.
    Returns {session_id, checkout_url} — redirect the user to checkout_url.
    """
    org_id = user.org_id or ""
    if not org_id:
        raise HTTPException(status_code=400, detail="No org associated with this user")

    user_email = getattr(user, "email", "") or ""
    result = create_checkout_session(
        plan=body.plan,
        org_id=org_id,
        customer_email=user_email,
        success_url=body.success_url,
        cancel_url=body.cancel_url,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.get("/subscription")
def subscription_status(
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Get current subscription status for the user's org."""
    org_id = user.org_id or ""
    if not org_id:
        return {"status": "no_org"}
    return get_subscription_status(org_id)


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """
    Stripe webhook endpoint — receives events for subscription lifecycle.
    Must be public (no auth middleware); Stripe signs each request.
    Register this URL in your Stripe Dashboard → Webhooks.

    Events handled:
      checkout.session.completed
      customer.subscription.created / updated / deleted
      invoice.payment_failed
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    result = handle_webhook(payload, sig_header)

    if "error" in result:
        # Return 400 so Stripe retries the event
        return JSONResponse(status_code=400, content=result)

    return JSONResponse(status_code=200, content=result)
