"""
OrthoLink Admin Routes
Organization provisioning and management.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field

from app.middleware.auth import AuthenticatedUser
from app.middleware.rbac import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


class ProvisionRequest(BaseModel):
    """Request to provision a new organization."""

    org_name: str = Field(..., min_length=2, max_length=100)
    slug: str = Field(..., min_length=2, max_length=50, pattern=r"^[a-z0-9-]+$")
    admin_email: EmailStr
    plan_tier: str = Field(default="free_trial", pattern=r"^(free_trial|starter|professional|enterprise)$")


class ProvisionResponse(BaseModel):
    """Response after provisioning an organization."""

    org_id: str
    org_name: str
    slug: str
    admin_email: str
    plan_tier: str
    magic_link_sent: bool
    message: str


@router.post("/provision", response_model=ProvisionResponse)
async def provision_organization(
    request: ProvisionRequest,
    user: AuthenticatedUser = Depends(require_admin),
):
    """
    Provision a new organization.

    PRD: Total time 30 seconds. Customer receives one email. Clicks one link.
    """
    try:
        from app.services.supabase_client import get_supabase_client

        client = get_supabase_client()

        # 1. Create organization
        import uuid
        org_id = str(uuid.uuid4())

        client.table("organizations").insert({
            "id": org_id,
            "org_name": request.org_name,
            "slug": request.slug,
            "plan_tier": request.plan_tier,
        }).execute()

        # 2. Create Supabase Auth user (via admin API)
        # Note: In production, use supabase admin client
        # For now, create the org_members entry

        # 3. Insert org_members
        client.table("org_members").insert({
            "id": str(uuid.uuid4()),
            "org_id": org_id,
            "user_id": user.user_id,  # Will be updated with new user's ID
            "role": "admin",
            "email": request.admin_email,
        }).execute()

        # 4. Seed org_provider_config (default: gpt-4o)
        client.table("org_provider_config").insert({
            "id": str(uuid.uuid4()),
            "org_id": org_id,
            "provider": "openai",
            "model": "gpt-4o",
            "embedding_model": "text-embedding-3-large",
        }).execute()

        # 5. Send welcome email
        magic_link_sent = False
        try:
            from app.services.email_service import get_email_service
            email_svc = get_email_service()
            magic_link_sent = email_svc.send_welcome_email(
                to_email=request.admin_email,
                org_name=request.org_name,
                magic_link=f"https://app.ortholink.ai/auth/callback?org={request.slug}",
            )
        except Exception as e:
            logger.warning(f"Failed to send welcome email: {e}")

        # Audit log
        from app.services.audit_logger import get_audit_logger
        audit = get_audit_logger()
        audit.log(
            action="org_provisioned",
            org_id=org_id,
            user_id=user.user_id,
            resource_type="organization",
            resource_id=org_id,
            details={
                "org_name": request.org_name,
                "plan_tier": request.plan_tier,
                "admin_email": request.admin_email,
            },
        )

        return ProvisionResponse(
            org_id=org_id,
            org_name=request.org_name,
            slug=request.slug,
            admin_email=request.admin_email,
            plan_tier=request.plan_tier,
            magic_link_sent=magic_link_sent,
            message=f"Organization '{request.org_name}' provisioned successfully.",
        )

    except Exception as e:
        logger.error(f"Provisioning failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to provision organization. Please try again.",
        )


class OrgSummary(BaseModel):
    """Summary of an organization for admin listing."""
    org_id: str
    org_name: str
    slug: str
    plan_tier: str
    member_count: int = 0


class OrgsListResponse(BaseModel):
    organizations: list[OrgSummary]
    total: int


@router.get("/orgs", response_model=OrgsListResponse)
async def list_organizations(
    user: AuthenticatedUser = Depends(require_admin),
):
    """List all organizations. Admin only."""
    try:
        from app.services.supabase_client import get_supabase_client
        client = get_supabase_client()
        result = client.table("organizations").select("*").execute()
        orgs = []
        for org in (result.data or []):
            orgs.append(OrgSummary(
                org_id=org.get("id", ""),
                org_name=org.get("org_name", org.get("name", "")),
                slug=org.get("slug", ""),
                plan_tier=org.get("plan_tier", "free_trial"),
            ))
        return OrgsListResponse(organizations=orgs, total=len(orgs))
    except Exception as e:
        logger.error(f"Failed to list organizations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list organizations.")
