"""
OrthoLink Configuration — Pydantic Settings loading from .env
"""

import logging
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings

_config_logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "OrthoLink"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # API
    api_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:3000,http://localhost:3005,http://localhost:3006,http://localhost:3007,http://localhost:3008,http://localhost:3009,http://localhost:3010,http://localhost:5173,http://localhost:8000"

    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""

    # OpenAI
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimensions: int = 3072
    openai_generation_model: str = "gpt-4o"

    # FAISS Vector Store
    faiss_index_path: str = "data/embeddings"

    # Sentry (optional)
    sentry_dsn: Optional[str] = None

    # PostHog (optional)
    posthog_api_key: Optional[str] = None
    posthog_host: str = "https://app.posthog.com"

    # Resend (email)
    resend_api_key: Optional[str] = None
    resend_from_email: str = "noreply@ortholink.ai"

    # Stripe (billing)
    stripe_secret_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    # Price IDs map: plan slug → Stripe price ID (set via STRIPE_PRICE_ID_STARTER etc.)
    stripe_price_id_starter: Optional[str] = None
    stripe_price_id_pro: Optional[str] = None
    stripe_price_id_enterprise: Optional[str] = None

    @property
    def stripe_price_ids(self) -> dict[str, str]:
        ids = {}
        if self.stripe_price_id_starter:
            ids["starter"] = self.stripe_price_id_starter
        if self.stripe_price_id_pro:
            ids["pro"] = self.stripe_price_id_pro
        if self.stripe_price_id_enterprise:
            ids["enterprise"] = self.stripe_price_id_enterprise
        return ids

    # Anti-hallucination
    confidence_threshold: float = 0.7
    similarity_threshold: float = 0.55

    # RSA Strategy scoring (PRD §4.5.4) — configurable weights for country ranking
    # Optimal entry sequence: weighted sum of normalized dimensions (0–1)
    strategy_weight_reuse_pct: float = 0.30
    strategy_weight_time: float = 0.25
    strategy_weight_cost: float = 0.25
    strategy_weight_revenue: float = 0.20

    # Rate limiting
    max_rpm: int = 60
    ai_rpm: int = 20  # Stricter limit for AI/LLM endpoints

    # Vault — AES-256-GCM encryption key (overrides SUPABASE_JWT_SECRET for vault ops)
    # Set VAULT_KEY in .env for dedicated encryption key; otherwise falls back to JWT secret
    vault_key: Optional[str] = None

    # GMDN Agency API key (https://www.gmdnagency.org)
    gmdn_api_key: str = ""

    # DVA: use CrewAI crew when True (PRD C1); else use direct pipeline.
    # Direct pipeline is more reliable — classifies items individually via FAISS+LLM.
    # CrewAI path relies on LLM generating a complete parseable JSON for the full report,
    # which silently falls back to empty report (0 items) when parsing fails.
    use_dva_crew: bool = False

    # RAA: scheduled monitoring (PRD §4.6.2). When enabled, run RAA at interval_hours.
    raa_scheduler_enabled: bool = False
    raa_scheduler_interval_hours: float = 24.0

    # Registry API keys
    swissdamed_api_key: str = ""

    # Ingestion pipeline settings
    ingestion_enabled: bool = True
    ingestion_registry_batch_size: int = 100
    ingestion_enforcement_days: int = 30

    # Redis (optional — used for FAISS result caching)
    redis_url: str = "redis://localhost:6379/0"
    faiss_cache_ttl: int = 3600  # seconds — FAISS result cache TTL (1 hour default)
    faiss_cache_enabled: bool = True  # Set False to disable Redis cache entirely

    # HIPAA / Business Continuity
    hipaa_recovery_audit_enabled: bool = True

    @property
    def cors_origins_list(self) -> list[str]:
        origins = [origin.strip() for origin in self.cors_origins.split(",")]
        for origin in origins:
            if origin == "*":
                _config_logger.warning(
                    "SECURITY WARNING: CORS_ORIGINS contains a wildcard ('*'). "
                    "This allows any origin to make cross-origin requests and is "
                    "unsafe in production. Set CORS_ORIGINS to explicit allowed "
                    "origins (e.g. https://app.ortholink.ai)."
                )
                break
        return origins

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
