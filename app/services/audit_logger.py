"""
OrthoLink Audit Logger Service
HC-6: Append-only audit log. Never delete, never update.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from supabase import Client

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Append-only audit logger that writes to Supabase audit_log table.
    HC-6: Records are NEVER deleted or updated.
    """

    def __init__(self, supabase_client: Optional[Client] = None):
        self._client = supabase_client

    def _get_client(self) -> Client:
        """Lazy-initialize Supabase client."""
        if self._client is None:
            from app.services.supabase_client import get_supabase_client
            self._client = get_supabase_client()
        return self._client

    def log(
        self,
        action: str,
        org_id: str,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> str:
        """
        Write an audit log entry. Returns the log entry ID.

        HC-6: This is APPEND-ONLY. No update or delete methods exist.
        """
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        entry = {
            "id": entry_id,
            "org_id": org_id,
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": json.dumps(details) if details else None,
            "ip_address": ip_address,
            "created_at": timestamp,
        }

        try:
            client = self._get_client()
            client.table("audit_log").insert(entry).execute()
            logger.info(f"Audit log: {action} by {user_id} on {resource_type}/{resource_id}")
        except Exception as e:
            # Fallback to local logging if Supabase is unavailable
            logger.error(f"Failed to write audit log to Supabase: {e}")
            logger.info(f"AUDIT_FALLBACK: {json.dumps(entry)}")

        return entry_id

    def log_dva_analysis(
        self,
        org_id: str,
        user_id: str,
        analysis_id: str,
        country: str,
        device_class: str,
        item_count: int,
        fraud_risk_score: float,
    ) -> str:
        """Log a DVA analysis event."""
        return self.log(
            action="dva_analysis",
            org_id=org_id,
            user_id=user_id,
            resource_type="analysis",
            resource_id=analysis_id,
            details={
                "country": country,
                "device_class": device_class,
                "item_count": item_count,
                "fraud_risk_score": fraud_risk_score,
            },
        )

    def log_auth_event(
        self,
        org_id: str,
        user_id: str,
        event_type: str,
        ip_address: Optional[str] = None,
    ) -> str:
        """Log an authentication event."""
        return self.log(
            action=f"auth_{event_type}",
            org_id=org_id,
            user_id=user_id,
            resource_type="auth",
            ip_address=ip_address,
        )


# Module-level singleton
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create the singleton audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
