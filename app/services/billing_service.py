"""
OrthoLink Billing Service — Stripe integration.

Plans:
  starter   — $299/month  — 1 seat,  50 agent runs/mo,  5 countries
  pro       — $799/month  — 5 seats, 200 agent runs/mo, 15 countries, RAA alerts
  enterprise— $1,999/month— unlimited seats, unlimited runs, priority support

HC-7: No AI provider names in customer-facing copy.
"""

import logging
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── Plan definitions ──────────────────────────────────────────────────────────

PLANS: dict[str, dict] = {
    "starter": {
        "name": "Starter",
        "price_usd_month": 299,
        "seats": 1,
        "agent_runs_per_month": 50,
        "countries": 5,
        "raa_alerts": False,
        "description": "For individual regulatory affairs professionals.",
    },
    "pro": {
        "name": "Pro",
        "price_usd_month": 799,
        "seats": 5,
        "agent_runs_per_month": 200,
        "countries": 15,
        "raa_alerts": True,
        "description": "For regulatory teams entering multiple markets.",
    },
    "enterprise": {
        "name": "Enterprise",
        "price_usd_month": 1999,
        "seats": -1,          # unlimited
        "agent_runs_per_month": -1,
        "countries": 15,
        "raa_alerts": True,
        "description": "For large organizations requiring unlimited access and priority support.",
    },
}


# ── Stripe client factory ─────────────────────────────────────────────────────

def _get_stripe():
    """Lazy-import stripe; returns None if not configured."""
    settings = get_settings()
    if not settings.stripe_secret_key:
        logger.warning("STRIPE_SECRET_KEY not set; billing unavailable")
        return None
    try:
        import stripe as _stripe
        _stripe.api_key = settings.stripe_secret_key
        return _stripe
    except ImportError:
        logger.error("stripe package not installed; run: pip install stripe")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def create_checkout_session(
    plan: str,
    org_id: str,
    customer_email: str,
    success_url: str,
    cancel_url: str,
) -> dict:
    """
    Create a Stripe Checkout session for the given plan.
    Returns {"session_id": ..., "checkout_url": ...} or {"error": ...}.
    """
    if plan not in PLANS:
        return {"error": f"Unknown plan '{plan}'. Valid plans: {list(PLANS.keys())}"}

    stripe = _get_stripe()
    if stripe is None:
        return {"error": "Billing not configured (STRIPE_SECRET_KEY missing)"}

    settings = get_settings()
    price_id = settings.stripe_price_ids.get(plan)
    if not price_id:
        return {
            "error": (
                f"STRIPE_PRICE_ID_{plan.upper()} not set in environment. "
                "Create Stripe products/prices and set the price IDs."
            )
        }

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            customer_email=customer_email,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={"org_id": org_id, "plan": plan},
            subscription_data={"metadata": {"org_id": org_id, "plan": plan}},
        )
        return {
            "session_id": session.id,
            "checkout_url": session.url,
            "plan": plan,
            "plan_details": PLANS[plan],
        }
    except Exception as e:
        logger.error(f"Stripe checkout session creation failed: {e}")
        return {"error": str(e)}


def get_subscription_status(org_id: str) -> dict:
    """
    Look up the current Stripe subscription for an org.
    Returns subscription details or {"status": "no_subscription"}.
    """
    stripe = _get_stripe()
    if stripe is None:
        return {"status": "billing_not_configured"}

    try:
        subscriptions = stripe.Subscription.search(
            query=f"metadata['org_id']:'{org_id}'",
            limit=1,
        )
        if not subscriptions.data:
            return {"status": "no_subscription"}

        sub = subscriptions.data[0]
        plan = sub.metadata.get("plan", "unknown")
        return {
            "status": sub.status,             # active | trialing | past_due | canceled
            "plan": plan,
            "plan_details": PLANS.get(plan, {}),
            "current_period_end": sub.current_period_end,
            "cancel_at_period_end": sub.cancel_at_period_end,
            "subscription_id": sub.id,
        }
    except Exception as e:
        logger.error(f"Stripe subscription lookup failed for org={org_id}: {e}")
        return {"status": "error", "error": str(e)}


def handle_webhook(payload: bytes, sig_header: str) -> dict:
    """
    Verify and process a Stripe webhook event.
    Returns {"handled": True, "event_type": ...} or {"error": ...}.
    """
    stripe = _get_stripe()
    if stripe is None:
        return {"error": "Billing not configured"}

    settings = get_settings()
    if not settings.stripe_webhook_secret:
        return {"error": "STRIPE_WEBHOOK_SECRET not set"}

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.stripe_webhook_secret
        )
    except stripe.error.SignatureVerificationError as e:
        logger.warning(f"Stripe webhook signature verification failed: {e}")
        return {"error": "Invalid signature"}
    except Exception as e:
        logger.error(f"Stripe webhook parsing failed: {e}")
        return {"error": str(e)}

    event_type = event["type"]
    logger.info(f"Stripe webhook received: {event_type}")

    if event_type == "checkout.session.completed":
        session = event["data"]["object"]
        org_id = session.get("metadata", {}).get("org_id")
        plan = session.get("metadata", {}).get("plan")
        logger.info(f"Checkout completed: org={org_id} plan={plan}")
        _handle_subscription_activated(org_id, plan, session.get("subscription"))

    elif event_type in ("customer.subscription.updated", "customer.subscription.created"):
        sub = event["data"]["object"]
        org_id = sub.get("metadata", {}).get("org_id")
        plan = sub.get("metadata", {}).get("plan")
        status = sub.get("status")
        logger.info(f"Subscription {event_type}: org={org_id} plan={plan} status={status}")

    elif event_type == "customer.subscription.deleted":
        sub = event["data"]["object"]
        org_id = sub.get("metadata", {}).get("org_id")
        logger.info(f"Subscription canceled: org={org_id}")
        _handle_subscription_canceled(org_id)

    elif event_type == "invoice.payment_failed":
        invoice = event["data"]["object"]
        customer_email = invoice.get("customer_email")
        logger.warning(f"Payment failed for {customer_email}")

    return {"handled": True, "event_type": event_type}


def _handle_subscription_activated(
    org_id: Optional[str],
    plan: Optional[str],
    subscription_id: Optional[str],
) -> None:
    """Persist subscription activation to Supabase (best-effort)."""
    if not org_id:
        return
    try:
        import os
        from supabase import create_client
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            return
        sb = create_client(url, key)
        sb.table("organizations").update({
            "plan": plan,
            "stripe_subscription_id": subscription_id,
            "subscription_status": "active",
        }).eq("id", org_id).execute()
        logger.info(f"Organization {org_id} upgraded to {plan}")
    except Exception as e:
        logger.warning(f"Failed to persist subscription activation: {e}")


def _handle_subscription_canceled(org_id: Optional[str]) -> None:
    """Downgrade org to free/no subscription in Supabase (best-effort)."""
    if not org_id:
        return
    try:
        import os
        from supabase import create_client
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            return
        sb = create_client(url, key)
        sb.table("organizations").update({
            "plan": "free",
            "subscription_status": "canceled",
        }).eq("id", org_id).execute()
        logger.info(f"Organization {org_id} subscription canceled")
    except Exception as e:
        logger.warning(f"Failed to persist subscription cancellation: {e}")


def list_plans() -> list[dict]:
    """Return all available plans."""
    return [{"id": k, **v} for k, v in PLANS.items()]
