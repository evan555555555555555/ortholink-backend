"""
OrthoLink RBAC Middleware
Role-Based Access Control: Admin / Reviewer / Viewer.
"""

import logging
from typing import Callable

from fastapi import Depends, HTTPException, status

from app.middleware.auth import AuthenticatedUser, get_current_user

logger = logging.getLogger(__name__)

# Role hierarchy: admin > reviewer > viewer
ROLE_HIERARCHY = {
    "admin": 3,
    "reviewer": 2,
    "viewer": 1,
}


def get_role_level(role: str) -> int:
    """Get the numeric level of a role."""
    return ROLE_HIERARCHY.get(role.lower(), 0)


def require_role(minimum_role: str) -> Callable:
    """
    FastAPI dependency factory that requires a minimum role level.

    Usage:
        @router.post("/admin-only", dependencies=[Depends(require_role("admin"))])
        async def admin_endpoint():
            ...
    """
    minimum_level = get_role_level(minimum_role)

    async def role_checker(
        user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
        user_level = get_role_level(user.role or "viewer")

        if user_level < minimum_level:
            logger.warning(
                f"Access denied: user {user.user_id} with role '{user.role}' "
                f"attempted action requiring '{minimum_role}'"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires '{minimum_role}' role or higher. "
                f"Your current role is '{user.role}'.",
            )

        return user

    return role_checker


def require_org_membership(
    user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """
    Dependency that ensures the user belongs to an organization.
    HC-8: Multi-tenant isolation.
    """
    if not user.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You must belong to an organization to access this resource.",
        )
    return user


# Convenience dependencies
require_admin = require_role("admin")
require_reviewer = require_role("reviewer")
require_viewer = require_role("viewer")
