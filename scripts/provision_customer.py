#!/usr/bin/env python3
"""
OrthoLink Customer Provisioning Script

Usage:
    python scripts/provision_customer.py \
        --org-name "BRM Extremities" \
        --slug "brm-extremities" \
        --admin-email "helton@brmextremities.com" \
        --plan-tier "professional"

Creates:
1. organizations row
2. Supabase Auth user
3. org_members (role=admin)
4. org_provider_config (default: gpt-4o)
5. Sends branded welcome email via Resend
"""

import argparse
import sys
import uuid
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.services.email_service import get_email_service


def provision(
    org_name: str,
    slug: str,
    admin_email: str,
    plan_tier: str = "free_trial",
) -> dict:
    """Provision a new customer organization."""
    settings = get_settings()

    from app.services.supabase_client import get_supabase_client
    client = get_supabase_client()

    org_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    print(f"Provisioning organization: {org_name} ({slug})")

    # 1. Create organization
    client.table("organizations").insert({
        "id": org_id,
        "org_name": org_name,
        "slug": slug,
        "plan_tier": plan_tier,
    }).execute()
    print(f"  [1/5] Organization created: {org_id}")

    # 2. Create auth user (via Supabase admin API)
    try:
        auth_response = client.auth.admin.create_user({
            "email": admin_email,
            "email_confirm": True,
            "user_metadata": {
                "org_id": org_id,
                "role": "admin",
            },
        })
        user_id = auth_response.user.id
        print(f"  [2/5] Auth user created: {user_id}")
    except Exception as e:
        print(f"  [2/5] Auth user creation skipped: {e}")

    # 3. Insert org_members
    client.table("org_members").insert({
        "id": str(uuid.uuid4()),
        "org_id": org_id,
        "user_id": user_id,
        "role": "admin",
        "email": admin_email,
    }).execute()
    print(f"  [3/5] Org member added (admin)")

    # 4. Seed org_provider_config
    client.table("org_provider_config").insert({
        "id": str(uuid.uuid4()),
        "org_id": org_id,
        "provider": "openai",
        "model": "gpt-4o",
        "embedding_model": "text-embedding-3-large",
    }).execute()
    print(f"  [4/5] Provider config seeded (gpt-4o)")

    # 5. Generate magic link and send email
    magic_link = f"https://app.ortholink.ai/auth/callback?org={slug}"
    try:
        email_svc = get_email_service()
        sent = email_svc.send_welcome_email(admin_email, org_name, magic_link)
        print(f"  [5/5] Welcome email {'sent' if sent else 'not sent (no API key)'}")
    except Exception as e:
        print(f"  [5/5] Email failed: {e}")

    print(f"\nProvisioning complete!")
    print(f"  Org ID: {org_id}")
    print(f"  Magic Link: {magic_link}")

    return {
        "org_id": org_id,
        "user_id": user_id,
        "slug": slug,
        "magic_link": magic_link,
    }


def main():
    parser = argparse.ArgumentParser(description="Provision a new OrthoLink customer")
    parser.add_argument("--org-name", required=True, help="Organization name")
    parser.add_argument("--slug", required=True, help="URL slug (lowercase, hyphens)")
    parser.add_argument("--admin-email", required=True, help="Admin email address")
    parser.add_argument(
        "--plan-tier",
        default="free_trial",
        choices=["free_trial", "starter", "professional", "enterprise"],
        help="Subscription tier",
    )

    args = parser.parse_args()
    provision(args.org_name, args.slug, args.admin_email, args.plan_tier)


if __name__ == "__main__":
    main()
