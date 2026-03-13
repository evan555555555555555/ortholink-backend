"""
OrthoLink Supabase Client
Singleton Supabase client for database operations.
"""

import logging
from typing import Optional

from supabase import Client, create_client

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """Get or create the singleton Supabase client."""
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.supabase_url or not settings.supabase_service_role_key:
            raise RuntimeError(
                "Supabase URL and service role key must be configured. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env"
            )
        _client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
        logger.info("Supabase client initialized")
    return _client


def get_supabase_anon_client() -> Client:
    """Get a Supabase client using the anon key (for RLS-enforced queries)."""
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise RuntimeError(
            "Supabase URL and anon key must be configured. "
            "Set SUPABASE_URL and SUPABASE_ANON_KEY in .env"
        )
    return create_client(
        settings.supabase_url,
        settings.supabase_anon_key,
    )
