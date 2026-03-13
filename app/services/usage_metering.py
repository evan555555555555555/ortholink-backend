"""
OrthoLink Usage Metering Service
Tracks API usage for billing and free trial limits.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Free trial limit
FREE_TRIAL_LIMIT = 15


class UsageMeter:
    """Tracks usage events for billing and trial management."""

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            from app.services.supabase_client import get_supabase_client
            self._client = get_supabase_client()
        return self._client

    def record_usage(
        self,
        org_id: str,
        user_id: str,
        agent_type: str,
        tokens_used: int = 0,
    ) -> str:
        """Record a usage event."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "org_id": org_id,
            "user_id": user_id,
            "agent_type": agent_type,
            "tokens_used": tokens_used,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            client = self._get_client()
            client.table("usage_events").insert(event).execute()
        except Exception as e:
            logger.error(f"Failed to record usage event: {e}")

        return event_id

    def get_usage_count(self, org_id: str) -> int:
        """Get total usage count for an organization."""
        try:
            client = self._get_client()
            result = (
                client.table("usage_events")
                .select("id", count="exact")
                .eq("org_id", org_id)
                .execute()
            )
            return result.count or 0
        except Exception as e:
            logger.error(f"Failed to get usage count: {e}")
            return 0

    def check_trial_limit(self, org_id: str) -> dict:
        """Check if org has exceeded free trial limit."""
        count = self.get_usage_count(org_id)
        return {
            "usage_count": count,
            "limit": FREE_TRIAL_LIMIT,
            "remaining": max(0, FREE_TRIAL_LIMIT - count),
            "exceeded": count >= FREE_TRIAL_LIMIT,
        }


_meter: Optional[UsageMeter] = None


def get_usage_meter() -> UsageMeter:
    global _meter
    if _meter is None:
        _meter = UsageMeter()
    return _meter
