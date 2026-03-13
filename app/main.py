"""
OrthoLink FastAPI Application Factory
Main entry point for the backend API.
"""

import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import get_settings
from app.middleware.rate_limiter import RateLimiterMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.routes import (
    admin, alerts, audit_distributor, auth, billing, briefing, capa, countries, cra,
    dashboard, dva, enforcement, export_pdf, health, history, ingestion, integrity, jobs,
    market_entry, pms, prep_submission, query, registries, risk, roa, strategy, surveillance,
    swarm as gco, tda, verify,
)

logger = logging.getLogger(__name__)

_raa_scheduler = None

# Per-country schedules (PRD §4.6.2 — match change velocity of each authority)
_RAA_SCHEDULE_DAILY = {"US", "EU"}
_RAA_SCHEDULE_3DAY = {"UK", "CA", "JP", "AU", "IN", "BR"}
_RAA_SCHEDULE_WEEKLY = {"UA", "CN", "RU", "CH", "MX", "KR", "SA"}


def _make_raa_job(countries: list[str]):
    """Return a callable that runs RAA for a fixed list of countries."""
    def _job():
        from app.routes.alerts import _run_check_changes_sync
        for country in countries:
            try:
                _run_check_changes_sync(country=country, document_id=None)
            except Exception as e:
                logger.warning("RAA scheduled run failed for %s: %s", country, e)
    return _job


def _run_daily_brief():
    """
    Cron entry point: ComplianceAuditor + Reality Checker daily brief.
    Agency pattern: automated evidence collection, no manual input required.
    """
    try:
        from app.services.daily_brief import run_daily_brief_cron
        run_daily_brief_cron()
    except Exception as e:
        logger.warning("Daily brief cron failed: %s", e)


def _run_ingestion_cron():
    """
    Cron entry point: priority-source ingestion pipeline.
    Ingests monitored regulatory docs + enforcement scrapers for US, EU, UK, AU.
    Runs in a separate thread so it never blocks the APScheduler event loop.
    """
    try:
        from app.ingestion.pipeline import run_priority_ingestion_sync
        run_priority_ingestion_sync()
    except Exception as e:
        logger.warning("Ingestion cron failed: %s", e)


def _run_recovery_audit_cron():
    """
    Cron entry point: daily 72-hour recovery audit check.
    Runs RecoveryAudit.run() if the service is installed; otherwise a no-op.
    """
    try:
        from app.services.recovery_audit import RecoveryAudit
        import asyncio as _asyncio

        async def _run():
            audit = RecoveryAudit()
            fn = getattr(audit, "run", None)
            if fn is None:
                return
            if _asyncio.iscoroutinefunction(fn):
                result = await fn()
            else:
                result = await _asyncio.to_thread(fn)
            # Store result in job_store so /surveillance/recovery-audit/latest picks it up
            from app.services.job_store import create_job, set_completed
            job_id = create_job(agent="recovery_audit")
            if hasattr(result, "model_dump"):
                payload = result.model_dump()
            elif isinstance(result, dict):
                payload = result
            else:
                payload = {"raw": str(result)}
            payload["job_id"] = job_id
            set_completed(job_id, payload)

        # Run in a thread-scoped event loop (APScheduler thread context)
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_asyncio.run, _run())
            future.result(timeout=7200)  # 2h hard timeout
    except ImportError:
        pass  # RecoveryAudit not installed — silent no-op
    except Exception as e:
        logger.warning("Recovery audit cron failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    global _raa_scheduler
    settings = get_settings()

    # Startup — fail fast on missing critical configuration
    # Skip in test environment (conftest monkeypatches fake keys before app loads)
    if settings.environment not in {"test", "testing"}:
        missing = []
        if not settings.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not settings.supabase_jwt_secret:
            missing.append("SUPABASE_JWT_SECRET")
        if missing:
            logger.critical(
                "FATAL: Missing required environment variables: %s — refusing to start.",
                ", ".join(missing),
            )
            raise RuntimeError(
                f"Missing required env vars: {', '.join(missing)}. "
                "Set them in backend/.env and restart."
            )

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Log vault status
    try:
        from app.services.vault import is_vault_enabled
        logger.info("Vault encryption: %s", "ENABLED" if is_vault_enabled() else "disabled (dev mode)")
    except Exception:
        pass

    # Initialize Sentry if configured
    if settings.sentry_dsn:
        try:
            import sentry_sdk
            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                environment=settings.environment,
                traces_sample_rate=0.1,
            )
            logger.info("Sentry initialized")
        except ImportError:
            logger.warning("sentry-sdk not installed, skipping Sentry init")

    # Initialize PostHog if configured
    if settings.posthog_api_key:
        try:
            import posthog
            posthog.api_key = settings.posthog_api_key
            posthog.host = settings.posthog_host
            logger.info("PostHog initialized")
        except ImportError:
            logger.warning("posthog not installed, skipping PostHog init")

    # RAA scheduler (PRD §4.6.2): per-country schedule tiers
    # US/EU: daily; UK/CA/JP/AU/IN/BR: every 3 days; UA/CN/RU/CH/MX/KR/SA: weekly
    if getattr(settings, "raa_scheduler_enabled", False):
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            _raa_scheduler = BackgroundScheduler(timezone="UTC")

            _raa_scheduler.add_job(
                _make_raa_job(list(_RAA_SCHEDULE_DAILY)),
                "interval", hours=24, id="raa_daily",
                misfire_grace_time=3600,
            )
            _raa_scheduler.add_job(
                _make_raa_job(list(_RAA_SCHEDULE_3DAY)),
                "interval", hours=72, id="raa_3day",
                misfire_grace_time=3600,
            )
            _raa_scheduler.add_job(
                _make_raa_job(list(_RAA_SCHEDULE_WEEKLY)),
                "interval", hours=168, id="raa_weekly",
                misfire_grace_time=3600,
            )

            # ── Agency-pattern cron jobs ──────────────────────────────────────
            # ComplianceAuditor + Reality Checker daily regulatory intelligence brief
            # Runs 1h after app start, then every 24h (offset avoids RAA collision)
            _raa_scheduler.add_job(
                _run_daily_brief,
                "interval", hours=24, id="daily_brief",
                misfire_grace_time=1800,
            )

            # Ingestion pipeline — priority sources (US, EU, UK, AU) every 24h
            # Offset by 2h from daily_brief to spread I/O load
            if getattr(settings, "ingestion_enabled", False):
                _raa_scheduler.add_job(
                    _run_ingestion_cron,
                    "interval", hours=24, id="ingestion_priority",
                    misfire_grace_time=3600,
                )
                logger.info("Ingestion cron scheduled: priority sources every 24h")

            # Recovery audit — daily check (runs if RecoveryAudit service is installed)
            if getattr(settings, "hipaa_recovery_audit_enabled", False):
                _raa_scheduler.add_job(
                    _run_recovery_audit_cron,
                    "interval", hours=24, id="recovery_audit_daily",
                    misfire_grace_time=3600,
                )
                logger.info("Recovery audit cron scheduled: every 24h")

            _raa_scheduler.start()
            logger.info(
                "RAA scheduler started — daily: %s | 3-day: %s | weekly: %s | brief: 24h",
                sorted(_RAA_SCHEDULE_DAILY),
                sorted(_RAA_SCHEDULE_3DAY),
                sorted(_RAA_SCHEDULE_WEEKLY),
            )
        except ImportError:
            logger.warning("apscheduler not installed, RAA scheduler disabled")

    yield

    # Shutdown
    if _raa_scheduler:
        try:
            _raa_scheduler.shutdown(wait=False)
            logger.info("RAA scheduler stopped")
        except Exception as e:
            logger.warning("RAA scheduler shutdown: %s", e)
    logger.info(f"Shutting down {settings.app_name}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Regulatory Intelligence Platform for Medical Devices. "
            "12 AI Agents. 18 Markets. FAISS vector store with 34K+ chunks."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # ── Security hardening ────────────────────────────────────────────────────
    # Middleware order: last-added = outermost (applied first on request,
    # last on response). SecurityHeaders must be outermost to cover all responses.

    # Rate limiter — AI endpoints: 20 RPM; all other /api/: 60 RPM
    app.add_middleware(
        RateLimiterMiddleware,
        default_rpm=settings.max_rpm,
        ai_rpm=settings.ai_rpm,
    )

    # CORS — explicit allow-list (no wildcard headers)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "X-Request-ID",
            "X-Requested-With",
            "Cache-Control",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ],
    )

    # Security headers — outermost: runs last on request, first on response
    app.add_middleware(SecurityHeadersMiddleware)

    # ── Global error sanitizer ────────────────────────────────────────────────
    # Strip internal stack traces from unhandled 500s (never leak to client)
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception for %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please try again."},
        )

    # Register routes
    app.include_router(health.router)
    app.include_router(dva.router, prefix=settings.api_prefix)
    app.include_router(countries.router, prefix=settings.api_prefix)
    app.include_router(auth.router, prefix=settings.api_prefix)
    app.include_router(query.router, prefix=settings.api_prefix)
    app.include_router(cra.router, prefix=settings.api_prefix)
    app.include_router(roa.router, prefix=settings.api_prefix)
    app.include_router(export_pdf.router, prefix=settings.api_prefix)
    app.include_router(jobs.router, prefix=settings.api_prefix)
    app.include_router(strategy.router, prefix=settings.api_prefix)
    app.include_router(market_entry.router, prefix=settings.api_prefix)
    app.include_router(alerts.router, prefix=settings.api_prefix)
    app.include_router(audit_distributor.router, prefix=settings.api_prefix)
    app.include_router(prep_submission.router, prefix=settings.api_prefix)
    app.include_router(admin.router, prefix=settings.api_prefix)
    app.include_router(history.router, prefix=settings.api_prefix)
    app.include_router(billing.router, prefix=settings.api_prefix)
    # New regulatory department agents
    app.include_router(tda.router, prefix=settings.api_prefix)
    app.include_router(pms.router, prefix=settings.api_prefix)
    app.include_router(capa.router, prefix=settings.api_prefix)
    app.include_router(gco.router, prefix=settings.api_prefix)
    app.include_router(verify.router, prefix=settings.api_prefix)
    app.include_router(integrity.router, prefix=settings.api_prefix)
    app.include_router(briefing.router, prefix=settings.api_prefix)
    app.include_router(risk.router, prefix=settings.api_prefix)
    app.include_router(dashboard.router, prefix=settings.api_prefix)
    # Enforcement + Market Surveillance + Global Registries + Ingestion Pipeline
    app.include_router(enforcement.router, prefix=settings.api_prefix)
    app.include_router(surveillance.router, prefix=settings.api_prefix)
    app.include_router(registries.router, prefix=settings.api_prefix)
    app.include_router(ingestion.router, prefix=settings.api_prefix)

    # ── SPA static file serving ────────────────────────────────────────────────
    # Serve the Vite-built frontend from frontend-vite/dist/ on the same port.
    _FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend-vite" / "dist"
    if _FRONTEND_DIST.is_dir():
        # Catch-all: serve static files from dist/ or fall back to index.html (SPA)
        # IMPORTANT: Must NOT intercept /api/, /health, /docs, /redoc, /openapi.json
        @app.get("/{full_path:path}")
        async def _spa_catch_all(full_path: str):
            # Never intercept API or system routes — let FastAPI 404 them properly
            if full_path.startswith(("api/", "health", "docs", "redoc", "openapi.json")):
                return JSONResponse(status_code=404, content={"detail": "Not found"})
            # Serve actual static files if they exist (JS/CSS bundles, favicon, images)
            candidate = _FRONTEND_DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(_FRONTEND_DIST / "index.html")

        logger.info("SPA frontend mounted from %s", _FRONTEND_DIST)
    else:
        logger.warning("Frontend dist not found at %s — API-only mode", _FRONTEND_DIST)

    return app


# Application instance
app = create_app()
